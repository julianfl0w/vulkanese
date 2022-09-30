import os
import sys
import numpy as np
import re
from buffers_sample_synth import *
from numba import njit
from tqdm import tqdm
import librosa

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
#!/bin/env python
import ctypes
import os
import time
import gc
import sys
import numpy as np

# import cupy as cp
import json
import cv2 as cv

# import matplotlib
import matplotlib.pyplot as plt
import sounddevice as sd
import jmidi
import pickle as pkl
import re
from jsynth import JSynth

print(sys.path)


# from vulkanese.vulkanese import *
import time
import rtmidi
from rtmidi.midiutil import *
import mido

here = os.path.dirname(os.path.abspath(__file__))

#######################################################


# WORKGROUP_SIZE = 1  # Workgroup size in compute shader.
# SAMPLES_PER_DISPATCH = 512
class SampleSynth(JSynth):
    def __init__(self, q, runtype):
        # Runtime Parameters
        self.GRAPH = False
        self.PYSOUND = False
        self.SOUND = False
        self.DEBUG = False
        exec("self." + runtype + " = True")

        self.q = q

        if self.DEBUG:
            self.constantsDict = {
                "POLYPHONY": 4,
                "POLYPHONY_PER_SHADER": 1,
                "SAMPLE_FREQUENCY": 44100,
                "OVERVOLUME": 1,
                "CHANNELS": 1,
                "SAMPLES_PER_DISPATCH": 4,
                "LATENCY_SECONDS": 0.010,
                "ENVELOPE_LENGTH": 16,
                "FILTER_STEPS": 16,
                "SAMPLE_SET_COUNT": 1,
                "MIDI_COUNT": 128,
                "SAMPLE_MAX_TIME_SECONDS": 5,
            }
        else:
            self.constantsDict = {
                "POLYPHONY": 128,
                "POLYPHONY_PER_SHADER": 8,
                "SAMPLE_FREQUENCY": 44100,
                "OVERVOLUME": 16,
                "CHANNELS": 1,
                "SAMPLES_PER_DISPATCH": 64,
                "LATENCY_SECONDS": 0.007,
                "ENVELOPE_LENGTH": 512,
                "FILTER_STEPS": 512,
                "SAMPLE_SET_COUNT": 1,
                "MIDI_COUNT": 128,
                "SAMPLE_MAX_TIME_SECONDS": 5,
            }

        # derived values
        self.constantsDict["SHADERS_PER_SAMPLE"] = int(
            self.constantsDict["POLYPHONY"] / self.constantsDict["POLYPHONY_PER_SHADER"]
        )
        self.constantsDict["SAMPLE_MAX_SAMPLE_COUNT"] = int(
            self.constantsDict["SAMPLE_MAX_TIME_SECONDS"]
            * self.constantsDict["SAMPLE_FREQUENCY"]
        )
        for k, v in self.constantsDict.items():
            exec("self." + k + " = " + str(v))

        JSynth.__init__(self)

        self.mm = jmidi.MidiManager(polyphony=self.POLYPHONY, synthInterface=self)

        # for delay effects
        self.RING_BUFFER_TIME_SECONDS = 2
        self.RING_BUFFER_SIZE = int(self.RING_BUFFER_TIME_SECONDS*self.SAMPLE_FREQUENCY / self.SAMPLES_PER_DISPATCH)
        self.ringBufferWriteIndex = 0
        self.ringBuffer = np.zeros((self.RING_BUFFER_SIZE, self.SAMPLES_PER_DISPATCH))
        self.delayTimeSeconds = 0.1
        self.delayTimeSamples = self.delayTimeSeconds*self.SAMPLE_FREQUENCY
        self.delayTimeBuffers = self.delayTimeSamples/self.SAMPLES_PER_DISPATCH
        self.decayFactor = 0.8
        self.delayOn = False
        
        # preallocate
        self.POLYLEN_ONES = np.ones((4 * self.POLYPHONY), dtype=np.float32)
        self.POLYLEN_ONES_POST = np.ones((4 * self.POLYPHONY), dtype=np.float32)
        self.POLYLEN_ONES64 = np.ones((2 * self.POLYPHONY), dtype=np.float64)
        self.POLYLEN_ONES_POST64 = np.ones((2 * self.POLYPHONY), dtype=np.float64)

        # given dimension A, return index
        self.dim2index = {
            "SHADERS_PER_SAMPLE": "shaderIndexInSample",
            "SAMPLES_PER_DISPATCH": "sampleNo",
            "POLYPHONY": "noteNo",
            "PARTIALS_PER_VOICE": "partialNo",
            "SAMPLE_SET_COUNT": "sampleSet",
        }

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

        with open("sample_shader.c", "r") as f:
            glslCode = f.read()

        # generate a compute cmd buffer
        self.computePipeline = ComputePipeline(
            glslCode=glslCode,
            device=device,
            constantsDict=self.constantsDict,
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

        # "ATTACK_TIME" : 0, ALL TIME AS A FLOAT OF SECONDS
        attackEnvelope = np.ones((4 * self.ENVELOPE_LENGTH))
        # smooth fadein
        fadeInTime = 100
        for i in range(fadeInTime):
            attackEnvelope[i] = float(i)/fadeInTime
        self.computePipeline.attackEnvelope.setBuffer(
            attackEnvelope
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
        skipval = self.computePipeline.noteBaseIndex.skipval
        self.fullAddArray = (
            np.ones((skipval * self.POLYPHONY), dtype=np.float64)
            * self.SAMPLES_PER_DISPATCH
        )
        self.postBendArray = self.fullAddArray.copy()

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

        # load all the samples
        if self.DEBUG:
            midiRange = np.arange(20) + 40
        else:
            midiRange = range(self.MIDI_COUNT)

        sampleDir = "samples/rhodes/"
        for m in tqdm(midiRange):
            y, samplerate = librosa.load(
                os.path.join(sampleDir, "midi" + str(m) + ".wav"), sr=None
            )
            self.computePipeline.sampleBuffer.pmap[
                self.SAMPLE_MAX_SAMPLE_COUNT
                * 4
                * 4
                * m : self.SAMPLE_MAX_SAMPLE_COUNT
                * 4
                * 4
                * m
                + len(y) * 4
            ] = y
            # ] = np.ones(len(y), dtype=np.float32)

        y, samplerate = librosa.load(os.path.join(sampleDir, "midipercussive.wav"), sr=None)
        self.computePipeline.sampleBuffer.pmap[
            self.SAMPLE_SET_COUNT * 4 * 4 * m : self.SAMPLE_SET_COUNT * 4 * 4 * m
            + len(y) * 4
        ] = y

    def run(self):

        # here we go
        gc.disable()

        # into the loop
        # for i in range(int(1024 * 128 / self.SAMPLES_PER_DISPATCH)):
        while 1:
            # do all cpu stuff first
            self.checkQ()

            # process MIDI
            self.mm.eventLoop(self)

            # UPDATE PMAP MEMORY
            self.updatePitchBend()
            self.computePipeline.noteBaseIndex.setBuffer(self.postBendArray)
            self.computePipeline.currTime.setByIndex(0, time.time())
            self.computePipeline.run()

            # do CPU tings NOT simultaneous with GPU process

            pa = np.ascontiguousarray(self.buffView)
            pa = np.reshape(pa, (self.SAMPLES_PER_DISPATCH, self.SHADERS_PER_SAMPLE))
            pa = np.sum(pa, axis=1)
            
            if self.delayOn:
                ringBufferReadIndex = int(self.ringBufferWriteIndex + self.RING_BUFFER_SIZE - self.delayTimeBuffers) % self.RING_BUFFER_SIZE   
                pa += self.ringBuffer[ringBufferReadIndex]*self.decayFactor
                self.ringBuffer[self.ringBufferWriteIndex] = pa
                self.ringBufferWriteIndex = (self.ringBufferWriteIndex + 1) % self.RING_BUFFER_SIZE   

            if self.PYSOUND:
                self.stream.write(pa)
            # we do CPU tings simultaneously?
            if self.DEBUG:
                self.computePipeline.dumpMemory()
                sys.exit()
            if self.GRAPH:
                self.updatingGraph(self.newVal)
                # eval("self." + varName + ".pmap[:] = newVal")

        self.release()
        # elegantly free all memory
        self.instance_inst.release()

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
        # self.computePipeline.noteBaseIncrement.setByIndex(newIndex, incrementPerSample)
        self.computePipeline.noteBaseIndex.setByIndex(newIndex, 0)
        self.postBendArray[newIndex * 2] = 0
        self.computePipeline.midiNoteNo.setByIndex(newIndex, note.midiIndex)
        self.computePipeline.noteStrikeTime.setByIndex(newIndex, time.time())

        # print("NOTE ON" + str(newIndex) + ", " + str(incrementPerSample) + ", " + str(note.midiIndex))
        # print(str(msg))
        # print(note.index)
        # print(incrementPerSample)

    def pitchWheel(self, val):
        pass

    def modWheel(self, val):
        self.computePipeline.tremAmount.setByIndex(0, val)

    def release(self):
        vkDestroyFence(self.device.vkDevice, self.fence, None)


def runSampleSynth(q, runtype="PYSOUND"):
    s = SampleSynth(q, runtype)
    if runtype == "DEBUG":
        s.runTest()
    s.run()


if __name__ == "__main__":
    runSampleSynth(q=None, runtype=sys.argv[1])
