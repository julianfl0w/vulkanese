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
import engineSpectral

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
            self.constantsDict = {
                "POLYPHONY": 16,
                "POLYPHONY_PER_SHADER": 1,
                "SLUTLEN": 2 ** 18,
                "PARTIALS_PER_VOICE": 1,
                "SAMPLE_FREQUENCY": 44100,
                "PARTIALS_PER_HARMONIC": 1,
                "PARTIAL_SPREAD": 0.02,
                "OVERVOLUME": 1,
                "CHANNELS": 1,
                "SAMPLES_PER_DISPATCH": 32,
                "LATENCY_SECONDS": 0.010,
                "ENVELOPE_LENGTH": 16,
                "FILTER_STEPS": 16,
            }
        else:
            self.constantsDict = {
                "POLYPHONY": 32,
                "POLYPHONY_PER_SHADER": 2,
                "SLUTLEN": 2 ** 18,
                "PARTIALS_PER_VOICE": 128,
                "SAMPLE_FREQUENCY": 44100,
                "PARTIALS_PER_HARMONIC": 3,
                "PARTIAL_SPREAD": 0.001,
                "OVERVOLUME": 8,
                "CHANNELS": 1,
                "SAMPLES_PER_DISPATCH": 64,
                "LATENCY_SECONDS": 0.007,
                "ENVELOPE_LENGTH": 512,
                "FILTER_STEPS": 512,
            }

        # derived values
        self.constantsDict["SHADERS_PER_SAMPLE"] = int(
            self.constantsDict["POLYPHONY"] / self.constantsDict["POLYPHONY_PER_SHADER"]
        )
        for k, v in self.constantsDict.items():
            exec("self." + k + " = " + str(v))
        self.spectralInterface = engineSpectral.Interface(
            self, self.constantsDict, runtype
        )
        self.mm = jmidi.MidiManager(
            polyphony=self.POLYPHONY, synthInterface=self.spectralInterface
        )

        # preallocate
        self.POLYLEN_ONES = np.ones((4 * self.POLYPHONY), dtype=np.float32)
        self.POLYLEN_ONES_POST = np.ones((4 * self.POLYPHONY), dtype=np.float32)
        self.POLYLEN_ONES64 = np.ones((2 * self.POLYPHONY), dtype=np.float64)
        self.POLYLEN_ONES_POST64 = np.ones((2 * self.POLYPHONY), dtype=np.float64)

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
                    self.spectralInterface.computeShader.attackEnvelope.setBuffer(
                        self.newVal
                    )
                elif varName == "attackLifespan":
                    mini = 0.25  # minimum lifespan, seconds
                    maxi = 5  # maximim lifespan, seconds
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    multiplier = self.ENVELOPE_LENGTH / (self.newVal)
                    print(multiplier)
                    self.spectralInterface.computeShader.attackSpeedMultiplier.setBuffer(
                        self.POLYLEN_ONES64 * multiplier
                    )

                elif varName == "releaseLifespan":
                    mini = 0.25  # minimum lifespan, seconds
                    maxi = 5  # maximim lifespan, seconds
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    multiplier = self.ENVELOPE_LENGTH / (self.newVal)
                    print(multiplier)
                    self.spectralInterface.computeShader.releaseSpeedMultiplier.setBuffer(
                        self.POLYLEN_ONES64 * multiplier
                    )

                elif varName == "partialSpread":
                    mini = 0.0001  # minimum
                    maxi = 0.001  # maximim
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    self.spectralInterface.PARTIAL_SPREAD = self.newVal
                    self.spectralInterface.updatePartials()

                elif varName == "partialCount":
                    mini = 1  # minimum
                    maxi = 15  # maximim
                    self.newVal = mini + (self.newVal * (maxi - mini))
                    self.spectralInterface.PARTIALS_PER_HARMONIC = int(self.newVal)
                    self.spectralInterface.updatePartials()

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

            pa = self.spectralInterface.run()

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
                self.spectralInterface.computePipeline.dumpMemory()
                sys.exit()
            if self.GRAPH:
                self.updatingGraph(self.newVal)
                # eval("self." + varName + ".pmap[:] = newVal")

        self.spectralInterface.release()
        # elegantly free all memory
        self.instance_inst.release()


def runSynth(q, runtype="PYSOUND"):
    s = Synth(q, runtype)
    if runtype == "DEBUG":
        s.runTest()
    s.run()


if __name__ == "__main__":
    runSynth(q=None, runtype=sys.argv[1])
