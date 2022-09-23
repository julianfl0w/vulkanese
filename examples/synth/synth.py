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
import shaders

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
class Synth:
    def __init__(self, q, runtype="PYSOUND"):
        self.q = q

        # Runtime Parameters
        self.GRAPH = False
        self.PYSOUND = False
        self.SOUND = False
        self.DEBUG = False
        exec("self." + runtype + " = True")

        if self.DEBUG:
            self.paramsDict = {
                "POLYPHONY": 16,
                "POLYPHONY_PER_SHADER": 1,
                "SLUTLEN": 2 ** 18,
                "PARTIALS_PER_VOICE": 1,
                "SAMPLE_FREQUENCY": 44100,
                "PARTIALS_PER_HARMONIC": 1,
                "PARTIAL_SPREAD": 0.02,
                "UNDERVOLUME": 3,
                "CHANNELS": 1,
                "SAMPLES_PER_DISPATCH": 32,
                "LATENCY_SECONDS": 0.010,
                "ENVELOPE_LENGTH": 16,
                "FILTER_STEPS": 16,
            }
        else:
            self.paramsDict = {
                "POLYPHONY": 32,
                "POLYPHONY_PER_SHADER": 1,
                "SLUTLEN": 2 ** 18,
                "PARTIALS_PER_VOICE": 128,
                "SAMPLE_FREQUENCY": 44100,
                "PARTIALS_PER_HARMONIC": 3,
                "PARTIAL_SPREAD": 0.001,
                "UNDERVOLUME": 3,
                "CHANNELS": 1,
                "SAMPLES_PER_DISPATCH": 64,
                "LATENCY_SECONDS": 0.025,
                "ENVELOPE_LENGTH": 512,
                "FILTER_STEPS": 512,
            }

        # derived values
        self.paramsDict["SHADERS_PER_SAMPLE"] = int(
            self.paramsDict["POLYPHONY"] / self.paramsDict["POLYPHONY_PER_SHADER"]
        )
        for k, v in self.paramsDict.items():
            exec("self." + k + " = " + str(v))
        self.synthShader = shaders.SynthShader(self, self.paramsDict, runtype)
        self.mm = jmidi.MidiManager(polyphony=self.POLYPHONY, synthInterface=self)

        # preallocate
        self.POLYLEN_ONES = np.ones((4 * self.POLYPHONY), dtype=np.float32)
        self.POLYLEN_ONES64 = np.ones((2 * self.POLYPHONY), dtype=np.float64)

        # Start the sound server
        if self.PYSOUND:
            self.stream = sd.Stream(
                samplerate=self.SAMPLE_FREQUENCY,
                blocksize=self.SAMPLES_PER_DISPATCH,
                device=None,
                channels=self.CHANNELS,
                dtype=np.float32,
                latency=self.LATENCY_SECONDS,
                extra_settings=None,
                callback=None,
                finished_callback=None,
                clip_off=None,
                dither_off=None,
                never_drop_input=None,
                prime_output_buffers_using_stream_callback=None,
            )

        if self.GRAPH:
            # to run GUI event loop
            plt.ion()

            # here we are creating sub plots
            self.figure, ax = plt.subplots(figsize=(10, 8))
            self.newVal = np.ones((4 * self.ENVELOPE_LENGTH * self.POLYPHONY))
            self.plot, = ax.plot(self.newVal)
            plt.ylabel("some numbers")
            plt.show()
            plt.ylim(-2, 2)

        if self.PYSOUND:
            self.stream.start()

    def noteOff(self, note):

        # we need to mulitiply by 16
        # because it seems minimum GPU read is 16 bytes
        # self.noteBaseIncrement.setByIndex(note.index, 0)
        # self.fullAddArray[note.index * 4] = 0
        index = note.index
        self.synthShader.computeShader.noteReleaseTime.setByIndex(index, time.time())

        if hasattr(self, "envelopeAmplitude"):
            currVol = self.synthShader.computeShader.envelopeAmplitude.getByIndex(index)
        # compute what current volume is
        else:
            secondsSinceStrike = time.time() - note.strikeTime
            currVolIndex = (
                secondsSinceStrike
                * self.synthShader.computeShader.attackSpeedMultiplier.getByIndex(index)
            )
            currVolIndex = min(currVolIndex, self.ENVELOPE_LENGTH - 1)
            currVol = self.synthShader.computeShader.attackEnvelope.getByIndex(
                int(currVolIndex)
            )

        self.synthShader.computeShader.noteVolume.setByIndex(index, currVol)

    def noteOn(self, note):

        # UNITY FREQS! (Phase on range [0,1) )
        incrementPerSample = (
            # 2 * 3.141592 * noteToFreq(msg.note) / self.SAMPLE_FREQUENCY
            jmidi.noteToFreq(note.midiIndex)
            / self.SAMPLE_FREQUENCY
        )
        newIndex = note.index
        self.synthShader.computeShader.noteVolume.setByIndex(newIndex, 1)
        self.synthShader.computeShader.noteBaseIncrement.setByIndex(
            newIndex, incrementPerSample
        )
        self.synthShader.computeShader.noteBasePhase.setByIndex(newIndex, 0)
        self.synthShader.computeShader.noteStrikeTime.setByIndex(newIndex, time.time())
        self.synthShader.fullAddArray[newIndex * 2] = (
            incrementPerSample * self.SAMPLES_PER_DISPATCH
        )

        print("NOTE ON" + str(newIndex) + ", " + str(incrementPerSample) + ", " + str(note.midiIndex))
        # print(str(msg))
        # print(note.index)
        # print(incrementPerSample)

    def pitchWheel(self, val):
        pass

    def modWheel(self, val):
        pass

    def updatingGraph(self, data):
        # print(pa2)
        # updating data values
        self.plot.set_ydata(data)

        # drawing updated values
        self.figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.figure.canvas.flush_events()

    def runTest(self):

        # start middle A note
        self.mm.processMidi(mido.Message("note_on", note=50, velocity=64, time=6.2))
        self.mm.processMidi(mido.Message("note_on", note=60, velocity=64, time=6.2))
        # self.midi2commands(mido.Message("note_on", note=70, velocity=64, time=6.2))

    def checkQ(self):
        # check the queue (q) for incoming commands
        if self.q is not None:
            if self.q.qsize():
                recvd = self.q.get()
                # print(recvd)
                varName, self.newVal = recvd
                if varName == "attackEnvelope":
                    self.synthShader.computeShader.attackEnvelope.setBuffer(self.newVal)
                elif varName == "attackLifespan":
                    mini = 0.25  # minimum lifespan, seconds
                    maxi = 5  # maximim lifespan, seconds
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    multiplier = self.ENVELOPE_LENGTH / (self.newVal)
                    print(multiplier)
                    self.synthShader.computeShader.attackSpeedMultiplier.setBuffer(
                        self.POLYLEN_ONES64 * multiplier
                    )

                elif varName == "releaseLifespan":
                    mini = 0.25  # minimum lifespan, seconds
                    maxi = 5  # maximim lifespan, seconds
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    multiplier = self.ENVELOPE_LENGTH / (self.newVal)
                    print(multiplier)
                    self.synthShader.computeShader.releaseSpeedMultiplier.setBuffer(
                        self.POLYLEN_ONES64 * multiplier
                    )

                elif varName == "partialSpread":
                    mini = 0.001  # minimum
                    maxi = 0.1  # maximim
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    self.synthShader.PARTIAL_SPREAD = self.newVal
                    self.synthShader.updatePartials()

    def run(self):

        # here we go
        # gc.disable()

        # into the loop
        # for i in range(int(1024 * 128 / self.SAMPLES_PER_DISPATCH)):
        while 1:
            # do all cpu stuff first
            self.checkQ()

            # process MIDI
            self.mm.eventLoop(self)

            pa = self.synthShader.run()

            if self.PYSOUND:
                self.stream.write(pa)
            # we do CPU tings simultaneously
            # apply pitch bend

            # NO MEMORY ACCESS
            # NO PMAP
            # startPoint = int((1 - self.modWheelReal) * 4 * self.FILTER_STEPS)
            # currFilt = self.freqFilterTotal[
            #    startPoint : startPoint + self.FILTER_STEPS * 4
            # ]
            # sineFilt = np.roll(sineFilt, 10)
            # currFilt += sineFilt
            # cfm = np.max(currFilt)
            # if cfm:
            #    currFilt /= cfm

            if self.DEBUG:
                self.synthShader.dumpMemory()
            if self.GRAPH:
                self.updatingGraph(self.newVal)
                # eval("self." + varName + ".pmap[:] = newVal")

        self.synthShader.release()
        # elegantly free all memory
        self.instance_inst.release()


def runSynth(q, runtype="PYSOUND"):
    s = Synth(q, runtype)
    if runtype == "DEBUG":
        s.runTest()
    s.run()


if __name__ == "__main__":
    runSynth(q=None, runtype=sys.argv[1])
