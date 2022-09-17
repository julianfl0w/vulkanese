#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np

# import cupy as cp
import json
import cv2 as cv

# import matplotlib
import matplotlib.pyplot as plt
import sounddevice as sd
import midiManager
import zmq
import pickle as pkl
import re
from numba import njit

print(sys.path)

here = os.path.dirname(os.path.abspath(__file__))

localtest = True
if localtest == True:
    vkpath = os.path.join(here, "..", "vulkanese")
    # sys.path.append(vkpath)
    sys.path = [vkpath] + sys.path
    print(vkpath)
    print(sys.path)
    from vulkanese import Instance
    from vulkanese import *
else:
    from vulkanese.vulkanese import *

# from vulkanese.vulkanese import *
import time
import rtmidi
from rtmidi.midiutil import *
import mido

here = os.path.dirname(os.path.abspath(__file__))

#######################################################


def noteToFreq(note):
    a = 440.0  # frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12.0))


class Note:
    def __init__(self, index):
        self.index = index
        self.voices = []
        self.velocity = 0
        self.velocityReal = 0
        self.held = False
        self.polytouch = 0
        self.midiIndex = 0
        self.msg = None
        self.releaseTime = -index
        self.strikeTime = -index


# WORKGROUP_SIZE = 1  # Workgroup size in compute shader.
# SAMPLES_PER_DISPATCH = 512
class Synth:

    # random ass fast cpu proc in random ass place
    # need to remove njit to profile
    # @njit
    def audioPostProcessAccelerated(pa, SAMPLES_PER_DISPATCH, SHADERS_PER_SAMPLE):
        #print(pa)
        pa = np.ascontiguousarray(pa)
        pa = np.reshape(pa, (SAMPLES_PER_DISPATCH, SHADERS_PER_SAMPLE))
        print(pa)
        pa = np.sum(pa, axis=1)
        #print(pa)
        # pa2 = pa #np.ascontiguousarray(pa)
        # pa3 = np.vstack((pa2, pa2))
        # pa4 = np.swapaxes(pa3, 0, 1)
        # pa5 = np.ascontiguousarray(pa4)
        # print(np.shape(pa2))
        # print(pa2)
        return pa

    def __init__(self, q, runtype = "PYSOUND"):
        self.q = q
        self.mm = midiManager.MidiManager()

        self.GRAPH = False
        self.PYSOUND = False
        self.SOUND = False
        self.DEBUG = False
        
        exec("self." + runtype + " = True")

        context = zmq.Context()
        # recieve work
        self.consumer_receiver = context.socket(zmq.PULL)
        self.consumer_receiver.connect("tcp://127.0.0.1:5557")

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

        if self.DEBUG:
            self.replaceDict = {
                "POLYPHONY": 8,
                "POLYPHONY_PER_SHADER": 2,
                "PARTIALS_PER_VOICE": 1,
                "MINIMUM_FREQUENCY_HZ": 20,
                "MAXIMUM_FREQUENCY_HZ": 20000,
                # "SAMPLE_FREQUENCY"     : 48000,
                "SAMPLE_FREQUENCY": 44100,
                "PARTIALS_PER_HARMONIC": 1,
                "UNDERVOLUME": 3,
                "CHANNELS": 1,
                "SAMPLES_PER_DISPATCH": 64,
                "LATENCY_SECONDS": 0.010,
                "ENVELOPE_LENGTH": 16,
                "FILTER_STEPS": 16,
            }
        else:
            self.replaceDict = {
                "POLYPHONY": 8,
                "POLYPHONY_PER_SHADER": 2,
                "PARTIALS_PER_VOICE": 1,
                "MINIMUM_FREQUENCY_HZ": 20,
                "MAXIMUM_FREQUENCY_HZ": 20000,
                # "SAMPLE_FREQUENCY"     : 48000,
                "SAMPLE_FREQUENCY": 44100,
                "PARTIALS_PER_HARMONIC": 1,
                "UNDERVOLUME": 3,
                "CHANNELS": 1,
                "SAMPLES_PER_DISPATCH": 64,
                "LATENCY_SECONDS": 0.010,
                "ENVELOPE_LENGTH": 16,
                "FILTER_STEPS": 16,
            }
        # derived values
        self.replaceDict["SHADERS_PER_SAMPLE"] = int(
            self.replaceDict["POLYPHONY"] / self.replaceDict["POLYPHONY_PER_SHADER"]
        )
        for k, v in self.replaceDict.items():
            exec("self." + k + " = " + str(v))

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

        self.shaderOutputBuffers = [
            {
                "name": "pcmBufferOut",
                "type": "float",
                "dims": ["SAMPLES_PER_DISPATCH", "SHADERS_PER_SAMPLE", "CHANNELS"],
            }
        ]

        self.debuggableVars = [
            # per timeslice, per polyslice (per shader)
            {
                "name": "currTimeWithSampleOffset",
                "type": "float64_t",
                "dims": ["SAMPLES_PER_DISPATCH", "SHADERS_PER_SAMPLE"],
            },
            {
                "name": "shadersum",
                "type": "float",
                "dims": ["SAMPLES_PER_DISPATCH", "SHADERS_PER_SAMPLE"],
            },
            {
                "name": "envelopeAmplitude",
                "type": "float",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            {
                "name": "envelopeIndexFloat64",
                "type": "float64_t",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            {
                "name": "envelopeIndex",
                "type": "int",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            # // per timeslice, per note  # make these vals 64bit if possible
            {
                "name": "secondsSinceStrike",
                "type": "float64_t",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            {
                "name": "secondsSinceRelease",
                "type": "float64_t",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            {
                "name": "fractional",
                "type": "float64_t",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            {
                "name": "basePhaseThisNote",
                "type": "float64_t",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            {
                "name": "noteVol",
                "type": "float",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            {
                "name": "increment",
                "type": "float",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            {
                "name": "innersum",
                "type": "float",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY"],
            },
            # // per timeslice, per note, per partial
            {
                "name": "thisIncrement",
                "type": "float",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY", "PARTIALS_PER_VOICE"],
            },
            {
                "name": "indexInFilter",
                "type": "int",
                "dims": ["SAMPLES_PER_DISPATCH", "POLYPHONY", "PARTIALS_PER_VOICE"],
            },
        ]

        dim2index = {
            "SHADERS_PER_SAMPLE": "shaderIndexInSample",
            "SAMPLES_PER_DISPATCH": "sampleNo",
            "POLYPHONY": "noteNo",
            "PARTIALS_PER_VOICE": "partialNo",
        }

        self.shaderInputBuffers = [
            {"name": "noteBaseIncrement", "type": "float", "dims": ["POLYPHONY"]},
            {
                "name": "partialMultiplier",
                "type": "float",
                "dims": ["PARTIALS_PER_VOICE"],
            },
            {"name": "partialVolume", "type": "float", "dims": ["PARTIALS_PER_VOICE"]},
            {"name": "noteVolume", "type": "float", "dims": ["POLYPHONY"]},
            {
                "name": "noteStrikeTime",
                "type": "float64_t",
                "dims": ["POLYPHONY"],
            },  # make these vals 64bit if possible
            {
                "name": "noteReleaseTime",
                "type": "float64_t",
                "dims": ["POLYPHONY"],
            },  # make these vals 64bit if possible
            {
                "name": "currTime",
                "type": "float64_t",
                "dims": ["POLYPHONY"],
            },  # make these vals 64bit if possible
            {
                "name": "noteBasePhase",
                "type": "float",
                "dims": ["POLYPHONY"],
            },  # make these vals 64bit if possible
            {"name": "attackEnvelope", "type": "float", "dims": ["ENVELOPE_LENGTH"]},
            {"name": "releaseEnvelope", "type": "float", "dims": ["ENVELOPE_LENGTH"]},
            {"name": "attackSpeedMultiplier", "type": "float64_t", "dims": ["POLYPHONY"]},
            {"name": "releaseSpeedMultiplier", "type": "float64_t", "dims": ["POLYPHONY"]},
            {"name": "freqFilter", "type": "float", "dims": ["FILTER_STEPS"]},
            {"name": "pitchFactor", "type": "float", "dims": ["POLYPHONY"]},
        ]

        bindingUniform = 0
        bindingStorage = 0
        allBuffers = []

        # if we're debugging, all intermediate variables become output buffers
        if self.DEBUG:
            allBufferDescriptions = (
                self.shaderOutputBuffers + self.debuggableVars + self.shaderInputBuffers
            )
        else:
            allBufferDescriptions = self.shaderOutputBuffers + self.shaderInputBuffers

        # create all buffers according to their description
        for s in allBufferDescriptions:
            format = VK_FORMAT_R32_SFLOAT
            if s["type"] == "float64_t":
                # format = VK_FORMAT_R64_SFLOAT
                #itemsize = 4 * 8
                itemsize = 4 * 4
            else:
                itemsize = 4 * 4

            TOTALSIZEBYTES = itemsize
            for d in s["dims"]:
                TOTALSIZEBYTES *= eval("self." + d)

            if s in self.shaderOutputBuffers or s in self.debuggableVars:
                descriptorSet = device.descriptorPool.descSetGlobal
                usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                binding = bindingStorage
                bindingStorage += 1
                # bindingUniform += int(itemsize/16)
            else:
                descriptorSet = device.descriptorPool.descSetUniform
                usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                binding = bindingUniform
                # bindingStorage += int(itemsize/16)
                bindingUniform += 1

            newBuff = Buffer(
                binding=binding,
                device=device,
                type=s["type"],
                descriptorSet=descriptorSet,
                qualifier="out",
                name=s["name"],
                readFromCPU=True,
                SIZEBYTES=TOTALSIZEBYTES,
                usage=usage,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                location=0,
                format=format,
            )
            allBuffers += [newBuff]
            exec("self." + s["name"] + " = newBuff")

        # initialize some of them
        # self.noteStrikeTime.setBuffer(np.ones((4 * self.POLYPHONY)) * time.time())
        # self.noteReleaseTime.setBuffer(
        #    np.ones((4 * self.POLYPHONY)) * (time.time() + 0.1)
        # )
        self.freqFilter.setBuffer(np.ones((4 * self.FILTER_STEPS)))

        # "ATTACK_TIME" : 0, ALL TIME AS A FLOAT OF SECONDS
        self.attackEnvelope.setBuffer(np.ones((4 * self.ENVELOPE_LENGTH)))

        # value of 1 means 1 second attack. 2 means 1/2 second attack
        ENVTIME_SECONDS = 1
        factor = self.ENVELOPE_LENGTH * 4 / ENVTIME_SECONDS
        #self.attackSpeedMultiplier.setBuffer(np.ones((4 * self.POLYPHONY)) * factor) ???
        self.attackSpeedMultiplier.setBuffer(np.ones((2 * self.POLYPHONY)) * factor)

        # value of 1 means 1 second attack. 2 means 1/2 second attack
        ENVTIME_SECONDS = 1
        factor = self.ENVELOPE_LENGTH * 4 / ENVTIME_SECONDS
        #self.releaseSpeedMultiplier.setBuffer(np.ones((4 * self.POLYPHONY)) * factor) ???
        self.releaseSpeedMultiplier.setBuffer(np.ones((2 * self.POLYPHONY)) * factor)

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

        header = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_ARB_separate_shader_objects : enable
        """
        for k, v in self.replaceDict.items():
            header += "#define " + k + " " + str(v) + "\n"
        header += "layout (local_size_x = SAMPLES_PER_DISPATCH, local_size_y = SHADERS_PER_SAMPLE, local_size_z = 1 ) in;"

        with open("synthshader.c", "r") as f:
            main = f.read()

        VARSDECL = ""
        if self.DEBUG:
            # if this is debug mode, all vars have already been declared as output buffers
            # intermediate variables must be indexed
            for var in self.debuggableVars:
                indexedVarString = var["name"] + "["
                for i, d in enumerate(var["dims"]):
                    indexedVarString += dim2index[d]
                    # since GLSL doesnt allow multidim arrays (kinda, see gl_arb_arrays_of_arrays)
                    # ok i dont understand Vulkan multidim
                    for j, rd in enumerate(var["dims"][i + 1 :]):
                        indexedVarString += "*" + rd
                    if i < (len(var["dims"]) - 1):
                        indexedVarString += "+"
                indexedVarString += "]"

                main2 = ""
                for line in main.split("\n"):
                    # replace all non-comments
                    if not line.strip().startswith("//"):
                        # whole-word replacement
                        # main2 += line.replace(var["name"], indexedVarString) + "\n"
                        main2 += (
                            re.sub(
                                r"\b{}\b".format(var["name"]), indexedVarString, line
                            )
                            + "\n"
                        )
                    # keep the comments
                    else:
                        main2 += line + "\n"

                main = main2
        else:
            # otherwise, just declare the variable type
            # INITIALIZE TO 0 ! 
            for var in self.debuggableVars:
                VARSDECL += var["type"] + " " + var["name"] + " = 0;\n"

        main = main.replace("VARIABLEDECLARATIONS", VARSDECL)

        # Stage
        existingBuffers = []
        mandleStage = Stage(
            device=device,
            name="mandlebrot.comp",
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            existingBuffers=existingBuffers,
            outputWidthPixels=700,
            outputHeightPixels=700,
            header=header,
            main=main,
            buffers=allBuffers,
        )

        #######################################################
        # Pipeline
        device.descriptorPool.finalize()

        self.computePipeline = ComputePipeline(
            device=device,
            workgroupShape=[self.SAMPLES_PER_DISPATCH, self.SHADERS_PER_SAMPLE, 1],
            stages=[mandleStage],
        )
        device.children += [self.computePipeline]

        # print the object hierarchy
        print("Object tree:")

        print(json.dumps(device.asDict(), indent=4))

        # Now we shall finally submit the recorded command buffer to a queue.
        self.submitInfo = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,  # submit a single command buffer
            pCommandBuffers=[
                self.computePipeline.commandBuffer.vkCommandBuffers[0]
            ],  # the command buffer to submit.
        )

        self.allNotes = [Note(index=i) for i in range(self.POLYPHONY)]
        self.sustain = False
        self.toRelease = []
        self.modWheelReal = 0.25
        self.pitchwheelReal = 1

        # We create a fence.
        fenceCreateInfo = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=0
        )
        self.fence = vkCreateFence(self.device.vkDevice, fenceCreateInfo, None)

        # precompute some arrays
        self.fullAddArray = np.zeros((4 * self.POLYPHONY), dtype=np.float64)

        if self.PYSOUND:
            self.stream.start()

        self.updatePartials()

        self.postBendArray = self.fullAddArray.copy()
        self.replaceDict["FENCEADDR"] = hex(eval(str(self.fence).split(" ")[-1][:-1]))
        self.replaceDict["DEVADDR"] = str(self.device.vkDevice).split(" ")[-1][:-1]
        self.replaceDict["SUBMITINFOADDR"] = str(ffi.addressof(self.submitInfo)).split(
            " "
        )[-1][:-1]
        # compile the C-code
        if self.SOUND:
            if os.path.exists("./alsatonic"):
                os.remove("./alsatonic")
            header = ""
            for k, v in self.replaceDict.items():
                header += "#define " + k + " " + str(v) + "\n"
            with open(os.path.join(here, "resources", "alsatonic.template"), "r") as f:
                at = header + f.read()
            cfilename = os.path.join(here, "resources", "alsatonic.c")
            with open(cfilename, "w+") as f:
                f.write(at)
            os.system("g++ " + cfilename + " -o alsatonic -lm -lasound")
            for i in range(1):
                if os.path.exists("./alsatonic"):
                    os.system("taskset -c 15 ./alsatonic")
                else:
                    die
                break

        sineFilt = 4 + np.sin(
            3.141592 * 16 * np.arange(4 * self.FILTER_STEPS) / (4 * self.FILTER_STEPS)
        )
        sineFilt /= np.max(sineFilt)

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
                    + partial_in_harmonic_unity * 0.04 * np.log2(harmonic + 2)
                )  # i hope log2 of the harmonic is the octave  # + partial_in_harmonic*0.0001

        self.partialMultiplier.setBuffer(hm)
        self.partialVolume.setBuffer(pv)

    def spawnVoice(self):
        unheldNotes = []
        for n in self.allNotes:
            if not n.held:
                unheldNotes += [n]
        if len(unheldNotes):
            return sorted(unheldNotes, key=lambda x: x.strikeTime, reverse=True)[0]
        else:
            return self.allNotes[0]

    def getNoteFromMidi(self, num):
        for n in self.allNotes:
            if n.midiIndex == num:
                return n
        return self.allNotes[0]

    def midi2commands(self, msg):

        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if self.sustain:
                self.toRelease += [msg.note]
                return

            note = self.getNoteFromMidi(msg.note)
            note.velocity = 0
            note.velocityReal = 0
            note.midiIndex = -1
            # if note.cluster is not None:
            #    note.cluster.silenceAllOps()
            # note.cluster = None
            note.held = False
            note.releaseTime = time.time()

            # we need to mulitiply by 16
            # because it seems minimum GPU read is 16 bytes
            self.noteBaseIncrement.setByIndex(note.index, 0)
            self.fullAddArray[note.index * 4] = 0
            self.noteReleaseTime.setByIndex(note.index, time.time())

            # print("NOTE OFF" + str(note.index))
            # print(str(msg))
        # if note on, spawn voices
        elif msg.type == "note_on":
            # print(msg)

            # if its already going, despawn it
            # if self.getNoteFromMidi(msg.note).held:
            #    self.midi2commands(mido.Message('note_off', note=note, velocity=0, time=6.2))

            note = self.spawnVoice()
            note.strikeTime = time.time()
            note.velocity = msg.velocity
            note.velocityReal = (msg.velocity / 127.0) ** 2
            note.held = True
            note.msg = msg
            note.midiIndex = msg.note

            incrementPerSample = (
                2 * 3.141592 * noteToFreq(msg.note) / self.SAMPLE_FREQUENCY
            )
            self.noteVolume.setByIndex(note.index, 1)
            self.noteBaseIncrement.setByIndex(note.index, incrementPerSample)
            # self.noteBasePhase.setByIndex(note.index, 0)
            self.noteStrikeTime.setByIndex(note.index, time.time())
            self.fullAddArray[note.index * 4] = (
                incrementPerSample * self.SAMPLES_PER_DISPATCH
            )

            # print("NOTE ON" + str(note.index))
            # print(str(msg))
            # print(note.index)
            # print(incrementPerSample)

        elif msg.type == "pitchwheel":
            # print("PW: " + str(msg.pitch))
            self.pitchwheel = msg.pitch
            ARTIPHON = 0
            if ARTIPHON:
                self.pitchwheel *= 2
            amountchange = self.pitchwheel / 8192.0
            octavecount = 2 / 12
            self.pitchwheelReal = pow(2, amountchange * octavecount)
            # print("PWREAL " + str(self.pitchwheelReal))
            # self.setAllIncrements()

        elif msg.type == "control_change":

            event = "control[" + str(msg.control) + "]"

            # print(event)
            # sustain pedal
            if msg.control == 64:
                print(msg.value)
                if msg.value:
                    self.sustain = True
                else:
                    self.sustain = False
                    for note in self.toRelease:
                        self.midi2commands(
                            mido.Message("note_off", note=note, velocity=0, time=6.2)
                        )
                    self.toRelease = []

            # mod wheel
            elif msg.control == 1:
                valReal = msg.value / 127.0
                print(valReal)
                self.modWheelReal = valReal

        elif msg.type == "polytouch":
            self.polytouch = msg.value
            self.polytouchReal = msg.value / 127.0

        elif msg.type == "aftertouch":
            self.aftertouch = msg.value
            self.aftertouchReal = msg.value / 127.0

        # if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
        #    # implement rising mono rate
        #    for heldnote in self.allNotes[::-1]:
        #        if heldnote.held and self.polyphony == self.voicesPerCluster :
        #            self.midi2commands(heldnote.msg)
        #            break

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
        self.midi2commands(mido.Message("note_on", note=50, velocity=64, time=6.2))
        #self.midi2commands(mido.Message("note_on", note=60, velocity=64, time=6.2))
                                      
    def run(self):

        # into the loop
        # for i in range(int(1024 * 128 / self.SAMPLES_PER_DISPATCH)):
        while 1:
            # do all cpu stuff first

            # check the queue (q) for incoming commands
            if False and self.q is not None:
                while self.q.qsize():
                    recvd = self.q.get()
                    # print(recvd)
                    varName, self.newVal = recvd
                    if varName == "attackEnvelope":
                        pass
                        # self.attackEnvelope.pmap[:] = self.newVal
                        # also load the release env starting at the final value
                        # self.releaseEnvelope.pmap[:] = numpy.linspace(start=self.newVal[-1], stop=0,
                        #                                              num=4 * self.ENVELOPE_LENGTH,
                        #                                              endpoint=True, retstep=False, dtype=np.float32, axis=0)

                    elif varName == "attackLifespan":
                        mini = 0.25  # minimum lifespan, seconds
                        maxi = 5  # maximim lifespan, seconds
                        self.newVal = mini + (self.newVal * (maxi - mini))
                        multiplier = self.ENVELOPE_LENGTH / (self.newVal)
                        print(multiplier)
                        # self.attackSpeedMultiplier.pmap[:] = np.ones((4*self.POLYPHONY), dtype=np.float32) * multiplier

                    elif varName == "releaseLifespan":
                        mini = 0.25  # minimum lifespan, seconds
                        maxi = 5  # maximim lifespan, seconds
                        self.newVal = mini + (self.newVal * (maxi - mini))
                        multiplier = self.ENVELOPE_LENGTH / (self.newVal)
                        print(multiplier)
                        # self.releaseSpeedMultiplier.pmap[:] = np.ones((4*self.POLYPHONY), dtype=np.float32) * multiplier

            # UPDATE PMAP MEMORY
            self.currTime.setBuffer(np.ones((2 * self.POLYPHONY)) * time.time()) # ???
            self.noteBasePhase.setBuffer(self.postBendArray)
            self.pitchFactor.setBuffer(
                np.ones((4 * self.POLYPHONY)) * self.pitchwheelReal
            )

            # process MIDI
            self.mm.eventLoop(self)

            # We submit the command buffer on the queue, at the same time giving a fence.
            vkQueueSubmit(self.device.compute_queue, 1, self.submitInfo, self.fence)

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

            # The command will not have finished executing until the fence is signalled.
            # So we wait here.
            # We will directly after this read our buffer from the GPU,
            # and we will not be sure that the command has finished executing unless we wait for the fence.
            # Hence, we use a fence here.
            vkWaitForFences(
                self.device.vkDevice, 1, [self.fence], VK_TRUE, 100000000000
            )


            # read all memory needed for simult postprocess
            pa = np.frombuffer(self.pcmBufferOut.pmap, np.float32)[::4]

            # do CPU tings simultaneous with GPU process
            pa = Synth.audioPostProcessAccelerated(
                pa, self.SAMPLES_PER_DISPATCH, self.SHADERS_PER_SAMPLE
            )
            sys.exit()
            if self.PYSOUND:
                self.stream.write(pa)
            # we do CPU tings simultaneously
            # apply pitch bend
            self.postBendArray += self.fullAddArray * self.pitchwheelReal

            vkResetFences(
                device=self.device.vkDevice, fenceCount=1, pFences=[self.fence]
            )
            
            if self.DEBUG:
                outdict = {}
                for debugVar in (
                    self.shaderInputBuffers
                    + self.debuggableVars
                    + self.shaderOutputBuffers
                ):
                    if "64" in debugVar["type"]:
                        #skipindex = 8
                        skipindex = 2
                    else:
                        skipindex = 4

                    # glsl to python
                    newt = glsltype2pythonstring(debugVar["type"])
                    runString = (
                        "np.frombuffer(self."
                        + debugVar["name"]
                        + ".pmap, "
                        + newt
                        + ")"
                    )
                    #print(runString)

                    # calculate the dims we need to turn the array into
                    newDims = []
                    for d in debugVar["dims"]:
                        newDims += [eval("self." + d)]
                        #print([eval("self." + d)])

                    # first retrieve the array with a simple eval
                    rcvdArray = eval(runString)
                    rcvdArray = list(rcvdArray.astype(float))[
                        ::skipindex
                    ]  # apply the skip index
                    rcvdArray = np.array(rcvdArray).reshape(newDims)
                    outdict[debugVar["name"]] = rcvdArray.tolist() # nested lists with same data, indices
                with open("debug.json", "w+") as f:
                    json.dump(outdict, f, indent=4)
                sys.exit()
            # if self.GRAPH:
            #    self.updatingGraph(currFilt)

            if self.GRAPH:
                self.updatingGraph(self.newVal)
                # eval("self." + varName + ".pmap[:] = newVal")

        vkDestroyFence(self.device.vkDevice, self.fence, None)
        # elegantly free all memory
        self.instance_inst.release()


def runSynth(q, runtype = "PYSOUND"):
    s = Synth(q, runtype)
    s.runTest()
    s.run()


if __name__ == "__main__":
    runSynth(q=None, runtype = sys.argv[1])
