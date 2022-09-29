import os
import sys
import numpy as np
import re
from shaderBuffers import *
from numba import njit

here = os.path.dirname(os.path.abspath(__file__))


def noteToFreq(note):
    a = 440.0  # frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12.0))


localtest = True
if localtest == True:
    vkpath = os.path.join(here, "..", "..", "vulkanese")
    sys.path = [vkpath] + sys.path
    from vulkanese import *
else:
    from vulkanese.vulkanese import *


class Interface:
    def __init__(self, parent, constantsDict, runtype):
        self.parent = parent
        # Runtime Parameters
        self.GRAPH = False
        self.PYSOUND = False
        self.SOUND = False
        self.DEBUG = False
        exec("self." + runtype + " = True")

        # given dimension A, return index
        self.dim2index = {
            "SHADERS_PER_SAMPLE": "shaderIndexInSample",
            "SAMPLES_PER_DISPATCH": "sampleNo",
            "POLYPHONY": "noteNo",
            "PARTIALS_PER_VOICE": "partialNo",
        }

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

        with open("shaderSpectral.c", "r") as f:
            glslCode = f.read()
            
        # generate a compute cmd buffer
        self.computePipeline = ComputePipeline(
            glslCode=glslCode,
            device=device,
            constantsDict=constantsDict,
            shaderOutputBuffers=shaderOutputBuffers,
            debuggableVars=debuggableVars,
            shaderInputBuffers=shaderInputBuffers,
            shaderInputBuffersNoDebug=shaderInputBuffersNoDebug,
            DEBUG=self.DEBUG,
            dim2index=self.dim2index,
        )

        # print the object hierarchy
        print("Object tree:")
        print(json.dumps(device.asDict(), indent=4))

        # initialize some of them
        # self.noteStrikeTime.setBuffer(np.ones((4 * self.POLYPHONY)) * time.time())
        # self.noteReleaseTime.setBuffer(
        #    np.ones((4 * self.POLYPHONY)) * (time.time() + 0.1)
        # )
        print(len(self.computePipeline.freqFilter.pmap))
        print(4 * self.FILTER_STEPS)
        self.computePipeline.freqFilter.setBuffer(np.ones((4 * self.FILTER_STEPS)))

        # "ATTACK_TIME" : 0, ALL TIME AS A FLOAT OF SECONDS
        self.computePipeline.attackEnvelope.setBuffer(
            np.ones((4 * self.ENVELOPE_LENGTH))
        )

        # initialize the Sine LookUp Table
        skipval = self.computePipeline.SLUT.skipval
        self.computePipeline.SLUT.setBuffer(
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
        skipval = self.computePipeline.attackSpeedMultiplier.skipval
        self.computePipeline.attackSpeedMultiplier.setBuffer(
            np.ones((skipval * self.POLYPHONY)) * factor
        )

        # value of 1 means 1 second attack. 2 means 1/2 second attack
        ENVTIME_SECONDS = 1
        factor = self.ENVELOPE_LENGTH * 4 / ENVTIME_SECONDS
        # self.releaseSpeedMultiplier.setBuffer(np.ones((4 * self.POLYPHONY)) * factor) ???
        skipval = self.computePipeline.releaseSpeedMultiplier.skipval
        self.computePipeline.releaseSpeedMultiplier.setBuffer(
            np.ones((skipval * self.POLYPHONY)) * factor
        )

        # precompute some arrays
        skipval = self.computePipeline.noteBasePhase.skipval
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
        self.computePipeline.releaseEnvelope.setBuffer(releaseEnv)

        # read all memory needed for simult postprocess
        self.buffView = np.frombuffer(
            self.computePipeline.pcmBufferOut.pmap, np.float32
        )[::4]

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
        print("Updating Partials")
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
                    + partial_in_harmonic_unity * self.PARTIAL_SPREAD
                    # * np.log2(harmonic + 2)
                )  # i hope log2 of the harmonic is the octave  # + partial_in_harmonic*0.0001

        self.computePipeline.partialMultiplier.setBuffer(hm)
        self.computePipeline.partialVolume.setBuffer(pv)

    def updatePitchBend(self):
        # with artiphon, only bend recent note
        ARTIPHON = False
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
            self.computePipeline.pitchFactor.setBuffer(self.parent.POLYLEN_ONES_POST64)

        else:
            self.postBendArray += self.fullAddArray * self.parent.mm.pitchwheelReal
            self.computePipeline.pitchFactor.setBuffer(
                self.parent.POLYLEN_ONES64 * self.parent.mm.pitchwheelReal
            )
            
    def run(self):

        # UPDATE PMAP MEMORY
        self.computePipeline.currTime.setByIndex(0, time.time())
        self.computePipeline.noteBasePhase.setBuffer(self.postBendArray)
        self.updatePitchBend()
        self.computePipeline.run()

        # do CPU tings NOT simultaneous with GPU process

        pa = np.ascontiguousarray(self.buffView)
        pa = np.reshape(pa, (self.SAMPLES_PER_DISPATCH, self.SHADERS_PER_SAMPLE))
        pa = np.sum(pa, axis=1)
        return pa

    def noteOff(self, note):

        # we need to mulitiply by 16
        # because it seems minimum GPU read is 16 bytes
        # self.noteBaseIncrement.setByIndex(note.index, 0)
        # self.fullAddArray[note.index * 4] = 0
        index = note.index
        self.computePipeline.noteReleaseTime.setByIndex(index, time.time())

        if hasattr(self, "envelopeAmplitude"):
            currVol = self.computePipeline.envelopeAmplitude.getByIndex(index)
        # compute what current volume is
        else:
            secondsSinceStrike = time.time() - note.strikeTime
            currVolIndex = (
                secondsSinceStrike
                * self.computePipeline.attackSpeedMultiplier.getByIndex(index)
            )
            currVolIndex = min(currVolIndex, self.ENVELOPE_LENGTH - 1)
            currVol = self.computePipeline.attackEnvelope.getByIndex(int(currVolIndex))

        self.computePipeline.noteVolume.setByIndex(index, currVol)

    def noteOn(self, note):

        # UNITY FREQS! (Phase on range [0,1) )
        incrementPerSample = (
            # 2 * 3.141592 * noteToFreq(msg.note) / self.SAMPLE_FREQUENCY
            noteToFreq(note.midiIndex)
            / self.SAMPLE_FREQUENCY
        )
        newIndex = note.index
        self.computePipeline.noteVolume.setByIndex(newIndex, 1)
        self.computePipeline.noteBaseIncrement.setByIndex(newIndex, incrementPerSample)
        self.computePipeline.noteBasePhase.setByIndex(newIndex, 0)
        self.computePipeline.noteStrikeTime.setByIndex(newIndex, time.time())
        self.fullAddArray[newIndex * 2] = incrementPerSample * self.SAMPLES_PER_DISPATCH

        # print("NOTE ON" + str(newIndex) + ", " + str(incrementPerSample) + ", " + str(note.midiIndex))
        # print(str(msg))
        # print(note.index)
        # print(incrementPerSample)

    def pitchWheel(self, val):
        pass

    def modWheel(self, val):
        pass

    def release(self):
        vkDestroyFence(self.device.vkDevice, self.fence, None)
