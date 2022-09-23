import os
import sys
import numpy as np
import re
from shaderBuffers import *
from numba import njit

here = os.path.dirname(os.path.abspath(__file__))

localtest = True
if localtest == True:
    vkpath = os.path.join(here, "..", "..", "vulkanese")
    # sys.path.append(vkpath)
    sys.path = [vkpath] + sys.path
    print(vkpath)
    print(sys.path)
    from vulkanese import Instance
    from vulkanese import *
else:
    from vulkanese.vulkanese import *


class SynthShader:
    def __init__(self, parent, paramsDict, runtype):
        self.parent = parent
        # Runtime Parameters
        self.GRAPH = False
        self.PYSOUND = False
        self.SOUND = False
        self.DEBUG = False
        exec("self." + runtype + " = True")

        self.paramsDict = paramsDict
        # device selection and instantiation
        self.instance_inst = Instance()
        print("available Devices:")
        # for i, d in enumerate(self.instance_inst.getDeviceList()):
        #    print("    " + str(i) + ": " + d)
        # die
        print("")

        # choose a device
        print("naively choosing device 0")
        device = self.instance_inst.getDevice(0)
        self.device = device

        for k, v in self.paramsDict.items():
            exec("self." + k + " = " + str(v))

        header = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_ARB_separate_shader_objects : enable
"""
        for k, v in self.paramsDict.items():
            header += "#define " + k + " " + str(v) + "\n"
        header += "layout (local_size_x = SAMPLES_PER_DISPATCH, local_size_y = SHADERS_PER_SAMPLE, local_size_z = 1 ) in;"

        with open("synthshader.c", "r") as f:
            main = f.read()

        # compute the size of each shader
        for buffList in [
            shaderOutputBuffers,
            debuggableVars,
            shaderInputBuffers,
            shaderInputBuffersNoDebug,
        ]:
            for s in buffList:
                TOTALSIZEBYTES = 4 * 4
                for d in s["dims"]:
                    TOTALSIZEBYTES *= eval("self." + d)
                s["SIZEBYTES"] = TOTALSIZEBYTES

                # optional: convert everything to float
                # if s["type"] == "float64_t":
                #    s["type"] = "float"

        # generate a compute cmd buffer
        self.computeShader = ComputeShader(
            main=main,
            header=header,
            device=device,
            shaderOutputBuffers=shaderOutputBuffers,
            debuggableVars=debuggableVars,
            shaderInputBuffers=shaderInputBuffers,
            shaderInputBuffersNoDebug=shaderInputBuffersNoDebug,
            DEBUG=self.DEBUG,
        )

        # print the object hierarchy
        print("Object tree:")
        print(json.dumps(device.asDict(), indent=4))

        # initialize some of them
        # self.noteStrikeTime.setBuffer(np.ones((4 * self.POLYPHONY)) * time.time())
        # self.noteReleaseTime.setBuffer(
        #    np.ones((4 * self.POLYPHONY)) * (time.time() + 0.1)
        # )
        print(len(self.computeShader.freqFilter.pmap))
        print(4 * self.FILTER_STEPS)
        self.computeShader.freqFilter.setBuffer(np.ones((4 * self.FILTER_STEPS)))

        # "ATTACK_TIME" : 0, ALL TIME AS A FLOAT OF SECONDS
        self.computeShader.attackEnvelope.setBuffer(np.ones((4 * self.ENVELOPE_LENGTH)))

        # initialize the Sine LookUp Table
        skipval = self.computeShader.SLUT.skipval
        self.computeShader.SLUT.setBuffer(
            np.sin(
                2
                * 3.1415926
                * np.arange(skipval * self.SLUTLEN)
                / (skipval * self.SLUTLEN)
            )
        )

        # value of 1 means 1 second attack. 2 means 1/2 second attack
        ENVTIME_SECONDS = 1
        factor = self.ENVELOPE_LENGTH * 4 / ENVTIME_SECONDS
        # self.attackSpeedMultiplier.setBuffer(np.ones((4 * self.POLYPHONY)) * factor) ???
        skipval = self.computeShader.attackSpeedMultiplier.skipval
        self.computeShader.attackSpeedMultiplier.setBuffer(
            np.ones((skipval * self.POLYPHONY)) * factor
        )

        # value of 1 means 1 second attack. 2 means 1/2 second attack
        ENVTIME_SECONDS = 1
        factor = self.ENVELOPE_LENGTH * 4 / ENVTIME_SECONDS
        # self.releaseSpeedMultiplier.setBuffer(np.ones((4 * self.POLYPHONY)) * factor) ???
        skipval = self.computeShader.releaseSpeedMultiplier.skipval
        self.computeShader.releaseSpeedMultiplier.setBuffer(
            np.ones((skipval * self.POLYPHONY)) * factor
        )

        # precompute some arrays
        skipval = self.computeShader.noteBasePhase.skipval
        self.fullAddArray = np.zeros((skipval * self.POLYPHONY), dtype=np.float64)

        self.updatePartials()

        self.postBendArray = self.fullAddArray.copy()

        sineFilt = 4 + np.sin(
            3.141592 * 16 * np.arange(4 * self.FILTER_STEPS) / (4 * self.FILTER_STEPS)
        )
        sineFilt /= np.max(sineFilt)

        # also load the release env starting at the final value
        releaseEnv = np.linspace(
            start=1.0,
            stop=0,
            num=4 * self.ENVELOPE_LENGTH,
            endpoint=True,
            retstep=False,
            dtype=np.float32,
            axis=0,
        )
        self.computeShader.releaseEnvelope.setBuffer(releaseEnv)

        # read all memory needed for simult postprocess
        self.buffView = np.frombuffer(self.computeShader.pcmBufferOut.pmap, np.float32)[
            ::4
        ]

    def range2unity(self, maxi):
        if maxi == 1:
            unity = [0]
        else:
            ar = np.arange(maxi)
            unity = ar - np.mean(ar)
            unity /= max(unity)
        return unity

    # update the partial scheme according to PARTIALS_PER_HARMONIC
    def updatePartials(self):
        hm = np.ones((4 * self.PARTIALS_PER_VOICE), dtype=np.float32)
        pv = np.ones((4 * self.PARTIALS_PER_VOICE), dtype=np.float32)
        # for i in range(int(self.PARTIALS_PER_VOICE/2)):
        #    hm[4*i*2]= 1.5
        np.set_printoptions(threshold=sys.maxsize)
        OVERTONE_COUNT = int(self.PARTIALS_PER_VOICE / self.PARTIALS_PER_HARMONIC)
        # simple equation to return value on range (-1,1)
        for harmonic in range(OVERTONE_COUNT):
            harmonic_unity = self.range2unity(OVERTONE_COUNT)[harmonic]
            for partial_in_harmonic in range(self.PARTIALS_PER_HARMONIC):
                partial_in_harmonic_unity = self.range2unity(
                    self.PARTIALS_PER_HARMONIC
                )[partial_in_harmonic]
                # simple equation to return value on range (-1,1)

                pv[
                    (harmonic * self.PARTIALS_PER_HARMONIC + partial_in_harmonic) * 4
                ] = 1 - abs(partial_in_harmonic_unity)

                hm[
                    (harmonic * self.PARTIALS_PER_HARMONIC + partial_in_harmonic) * 4
                ] = (
                    1
                    + harmonic
                    + partial_in_harmonic_unity
                    * self.PARTIAL_SPREAD
                    * np.log2(harmonic + 2)
                )  # i hope log2 of the harmonic is the octave  # + partial_in_harmonic*0.0001

        self.computeShader.partialMultiplier.setBuffer(hm)
        self.computeShader.partialVolume.setBuffer(pv)

    def run(self):

        # UPDATE PMAP MEMORY
        self.postBendArray += self.fullAddArray * self.parent.mm.pitchwheelReal
        self.computeShader.currTime.setByIndex(0, time.time())
        self.computeShader.noteBasePhase.setBuffer(self.postBendArray)
        self.computeShader.pitchFactor.setBuffer(
            self.parent.POLYLEN_ONES * self.parent.mm.pitchwheelReal
        )
        self.computeShader.run()

        # do CPU tings NOT simultaneous with GPU process

        pa = np.ascontiguousarray(self.buffView)
        pa = np.reshape(pa, (self.SAMPLES_PER_DISPATCH, self.SHADERS_PER_SAMPLE))
        pa = np.sum(pa, axis=1)
        return pa

    def dumpMemory(self):

        outdict = {}
        for debugVar in shaderInputBuffers + debuggableVars + shaderOutputBuffers:
            if "64" in debugVar["type"]:
                # skipindex = 8
                skipindex = 2
            else:
                skipindex = 4

            # glsl to python
            newt = glsltype2pythonstring(debugVar["type"])
            runString = (
                "np.frombuffer(self.computeShader."
                + debugVar["name"]
                + ".pmap, "
                + newt
                + ")"
            )
            # print(runString)

            # calculate the dims we need to turn the array into
            newDims = []
            for d in debugVar["dims"]:
                newDims += [eval("self." + d)]
                # print([eval("self." + d)])

            # first retrieve the array with a simple eval
            rcvdArray = eval(runString)
            rcvdArray = list(rcvdArray.astype(float))[
                ::skipindex
            ]  # apply the skip index
            rcvdArray = np.array(rcvdArray).reshape(newDims)
            outdict[
                debugVar["name"]
            ] = rcvdArray.tolist()  # nested lists with same data, indices
        with open("debug.json", "w+") as f:
            json.dump(outdict, f, indent=4)
        sys.exit()

    # if self.GRAPH:
    #    self.updatingGraph(currFilt)

    def release(self):
        vkDestroyFence(self.device.vkDevice, self.fence, None)
