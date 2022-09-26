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
    from vulkanese import *
else:
    from vulkanese.vulkanese import *


class Add:
    def __init__(self, parent, constantsDict, runtype):

        # any parameters set here will be available as
        # self.param within this Python class
        # and as a defined value within the shader
        self.constantsDict = {"NUMBERS_TO_SUM": 2 ** 28}

        # Input buffers to the shader
        # These are Uniform Buffers normally,
        # Storage Buffers in DEBUG Mode
        shaderInputBuffers = [
            {"name": "bufferToSum", "type": "float64_t", "dims": ["NUMBERS_TO_SUM"]}
        ]

        # any input buffers you want to exclude from debug
        # for example, a sine lookup table
        shaderInputBuffersNoDebug = [{}]

        # variables that are usually intermediate variables in the shader
        # but in DEBUG mode they are made visible to the CPU (as Storage Buffers)
        # so that you can view shader intermediate values :)
        debuggableVars = [{}]

        # the output of the compute shader,
        # which in our case is always a Storage Buffer
        shaderOutputBuffers = [
            {
                "name": "sumOut",
                "type": "float64",
                # "dims": ["SAMPLES_PER_DISPATCH", "SHADERS_PER_SAMPLE", "CHANNELS"],
                "dims": ["SAMPLES_PER_DISPATCH", "SHADERS_PER_SAMPLE"],
            }
        ]

        # derived values
        self.constantsDict["SHADERS_PER_SAMPLE"] = int(
            self.constantsDict["POLYPHONY"] / self.constantsDict["POLYPHONY_PER_SHADER"]
        )
        for k, v in self.constantsDict.items():
            exec("self." + k + " = " + str(v))

        self.spectralAdd = engineSpectral.Add(self, self.constantsDict, runtype)

        self.parent = parent
        # Runtime Parameters
        self.GRAPH = False
        self.PYSOUND = False
        self.SOUND = False
        self.DEBUG = False
        exec("self." + runtype + " = True")

        self.constantsDict = constantsDict
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

        for k, v in self.constantsDict.items():
            exec("self." + k + " = " + str(v))

        header = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_ARB_separate_shader_objects : enable
"""
        for k, v in self.constantsDict.items():
            header += "#define " + k + " " + str(v) + "\n"
        header += "layout (local_size_x = SAMPLES_PER_DISPATCH, local_size_y = SHADERS_PER_SAMPLE, local_size_z = 1 ) in;"

        with open("shaderSpectral.c", "r") as f:
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

    def run(self):

        # UPDATE PMAP MEMORY
        self.computeShader.currTime.setByIndex(0, time.time())
        self.computeShader.noteBasePhase.setBuffer(self.postBendArray)

        # with artiphon, only bend recent note
        ARTIPHON = True
        if ARTIPHON:
            self.parent.POLYLEN_ONES_POST64[:] = self.fullAddArray[:]
            self.parent.POLYLEN_ONES_POST64[
                self.parent.mm.mostRecentlyStruckNoteIndex * 2
            ] = (
                self.parent.POLYLEN_ONES_POST64[
                    self.parent.mm.mostRecentlyStruckNoteIndex * 2
                ]
                * self.parent.mm.pitchwheelReal
            )
            self.postBendArray += self.parent.POLYLEN_ONES_POST64

            self.parent.POLYLEN_ONES_POST64[:] = self.parent.POLYLEN_ONES64[:]
            self.parent.POLYLEN_ONES_POST64[
                self.parent.mm.mostRecentlyStruckNoteIndex * 2
            ] = self.parent.mm.pitchwheelReal
            self.computeShader.pitchFactor.setBuffer(self.parent.POLYLEN_ONES_POST64)

        else:
            self.postBendArray += self.fullAddArray * self.parent.mm.pitchwheelReal
            self.computeShader.pitchFactor.setBuffer(
                self.parent.POLYLEN_ONES64 * self.parent.mm.pitchwheelReal
            )
        self.computeShader.run()

        # do CPU tings NOT simultaneous with GPU process

        pa = np.ascontiguousarray(self.buffView)
        pa = np.reshape(pa, (self.SAMPLES_PER_DISPATCH, self.SHADERS_PER_SAMPLE))
        pa = np.sum(pa, axis=1)
        return pa

    # if self.GRAPH:
    #    self.updatingGraph(currFilt)

    def release(self):
        vkDestroyFence(self.device.vkDevice, self.fence, None)
