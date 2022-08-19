#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import cv2 as cv

# import matplotlib
import matplotlib.pyplot as plt
import sounddevice as sd
import midiManager

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
    a = 440.0 #frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12.0))

class Note:
    def __init__(self, index):
        self.index  = index
        self.voices = []
        self.velocity = 0
        self.velocityReal = 0
        self.held  = False
        self.polytouch = 0
        self.midiIndex = 0
        self.msg  = None
        self.releaseTime = -index

# WORKGROUP_SIZE = 1  # Workgroup size in compute shader.
# SAMPLES_PER_DISPATCH = 512
class Synth:
    def __init__(self, GRAPH):

        self.mm = midiManager.MidiManager()

        self.GRAPH = False
        self.PYSOUND = True
        self.SOUND = False

        # device selection and instantiation
        self.instance_inst = Instance()
        print("available Devices:")
        #for i, d in enumerate(self.instance_inst.getDeviceList()):
        #    print("    " + str(i) + ": " + d)
        #die
        print("")

        # choose a device
        print("naively choosing device 0")
        device = self.instance_inst.getDevice(0)
        self.device = device

        self.replaceDict = {
            "POLYPHONY": 128,
            "PARTIALS_PER_VOICE": 256,
            "MINIMUM_FREQUENCY_HZ": 20,
            "MAXIMUM_FREQUENCY_HZ": 20000,
            # "SAMPLE_FREQUENCY"     : 48000,
            "SAMPLE_FREQUENCY": 44100,
            "UNDERVOLUME": 3,
            "CHANNELS": 1,
            "SAMPLES_PER_DISPATCH": 64,
            "LATENCY_SECONDS": 0.100,
        }
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

        self.pcmBufferOut = Buffer(
            binding=0,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetGlobal,
            qualifier="in",
            name="pcmBufferOut",
            readFromCPU=True,
            SIZEBYTES=4
            * 4
            * self.SAMPLES_PER_DISPATCH * self.CHANNELS,
            initData=np.zeros((4 * self.SAMPLES_PER_DISPATCH * self.CHANNELS), dtype=np.float32),
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.noteBasePhase = Buffer(
            binding=1,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteBasePhase",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            initData=np.zeros(
                (4 * self.POLYPHONY), dtype=np.float32
            ),
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.noteBaseIncrement = Buffer(
            binding=2,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteBaseIncrement",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            initData=np.zeros((4 * self.POLYPHONY), dtype=np.float32),
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.partialMultiplier = Buffer(
            binding=3,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="partialMultiplier",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.PARTIALS_PER_VOICE,
            initData=np.ones((4 * self.PARTIALS_PER_VOICE), dtype=np.float32),
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.partialVolume = Buffer(
            binding=4,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            readFromCPU=True,
            initData=np.ones((4 * self.POLYPHONY), dtype=np.float32),
            name="partialVolume",
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )
        
        self.noteVolume = Buffer(
            binding=5,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteVolume",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            initData=np.zeros((4 * self.POLYPHONY), dtype=np.float32),
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        # noteAge = Buffer(
        #    binding=5,
        #    device=device,
        #    type="float",
        #    descriptorSet=device.descriptorPool.descSetGlobal,
        #    qualifier="",
        #    name="noteAge",
        #    SIZEBYTES=4 * 4 * PARTIALS_PER_VOICE,
        #    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        #    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
        #    location=0,
        #    format=VK_FORMAT_R32_SFLOAT,
        # )
        #
        # ADSR = Buffer(
        #    binding=6,
        #    device=device,
        #    type="Pixel",
        #    descriptorSet=device.descriptorPool.descSetGlobal,
        #    qualifier="",
        #    name="ADSR",
        #    SIZEBYTES=4 * 4 * PARTIALS_PER_VOICE,
        #    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        #    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
        #    location=0,
        #    format=VK_FORMAT_R32_SFLOAT,
        # )

        header = """#version 450
        #extension GL_ARB_separate_shader_objects : enable
        """
        for k, v in self.replaceDict.items():
            header += "#define " + k + " " + str(v) + "\n"
        header += "layout (local_size_x = 1, local_size_y = PARTIALS_PER_VOICE, local_size_z = 1 ) in;"

        main = """

        void main() {

          uint timeSlice = gl_GlobalInvocationID.y;

          float sum = 0;
          
          for (uint noteNo = 0; noteNo<POLYPHONY; noteNo++){
              float noteVol = noteVolume[noteNo];
              float increment = noteBaseIncrement[noteNo];
              float phase = noteBasePhase[noteNo] + (timeSlice * increment);

              float innersum = 0;
              for (uint partialNo = 0; partialNo<PARTIALS_PER_VOICE; partialNo++)
              {
                float vol = partialVolume[partialNo];

                float harmonicRatio   = partialMultiplier[partialNo];
                innersum += vol * sin(phase*harmonicRatio);

              }
              sum+=innersum*noteVol;
          }
          
          pcmBufferOut[timeSlice] = sum/(PARTIALS_PER_VOICE*POLYPHONY);
        }
        """

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
            buffers=[
                self.pcmBufferOut,
                self.noteBasePhase,
                self.noteBaseIncrement,
                self.partialMultiplier,
                self.partialVolume,
                self.noteVolume,
            ],
        )

        #######################################################
        # Pipeline
        device.descriptorPool.finalize()

        self.computePipeline = ComputePipeline(
            device=device,
            workgroupShape=[1, self.SAMPLES_PER_DISPATCH, 1],
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
        
        self.allNotes = [Note(index = i) for i in range(self.POLYPHONY)] 
        
    def spawnVoice(self):
        unheldNotes = []
        for n in self.allNotes:
            if not n.held:
                unheldNotes += [n]
        if len(unheldNotes):
            return sorted(unheldNotes, key=lambda x: x.releaseTime, reverse=True)[0]
        else:
            return self.allNotes[0]
    
    def getNoteFromMidi(self, num):
        for n in self.allNotes:
            if n.midiIndex == num:
                return n
        return self.allNotes[0]
    
    def midi2commands(self, msg):
        
        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            #if self.sustain:
            #    self.toRelease[msg.note] = True
            #    return

            note = self.getNoteFromMidi(msg.note)
            note.velocity = 0 
            note.velocityReal = 0 
            #if note.cluster is not None:
            #    note.cluster.silenceAllOps()
            #note.cluster = None
            note.held = False
            note.releaseTime = time.time()
            #self.partialVolume.pmap[msg.note*4:(msg.note+1)*4] = np.array([0                                                      ],dtype=np.float32) 

            self.noteBaseIncrement.pmap[note.index*16:note.index*16+4]   = \
                np.array([0],dtype=np.float32)
            self.noteBasePhase.pmap[note.index*16:note.index*16+4]   = \
                np.array([0],dtype=np.float32)
            self.fullAddArray[note.index*4] = 0
            self.noteVolume.pmap[note.index*16:note.index*16+4]   = \
                np.array([0],dtype=np.float32)
            print("NOTE OFF" + str(note.index))
        # if note on, spawn voices
        elif msg.type == "note_on":
            #print(msg)
            note = self.spawnVoice()
            note.velocity     = msg.velocity
            note.velocityReal = (msg.velocity/127.0)**2
            note.held = True
            note.msg = msg
            note.midiIndex = msg.note
            #self.partialVolume.pmap[msg.note*4:(msg.note+1)*4] = np.array([1                                                      ],dtype=np.float32) 
            #self.noteBaseIncrement.pmap[msg.note*4:(msg.note+1)*4]   = np.array([2*3.141592*noteToFreq(msg.note) / self.SAMPLE_FREQUENCY],dtype=np.float32)

            incrementPerSample = 2*3.141592*noteToFreq(msg.note) / self.SAMPLE_FREQUENCY
            self.noteBaseIncrement.pmap[note.index*16:note.index*16+4]   = \
                np.array([incrementPerSample],dtype=np.float32)
            self.noteBasePhase.pmap[note.index*16:note.index*16+4]   = \
                np.array([0],dtype=np.float32)
            self.noteVolume.pmap[note.index*16:note.index*16+4]   = \
                np.array([1],dtype=np.float32)
            self.fullAddArray[note.index*4] = incrementPerSample*self.SAMPLES_PER_DISPATCH

            print("NOTE ON" + str(note.index))
            #print(note.index)
            #print(incrementPerSample)

        elif msg.type == 'pitchwheel':
            print("PW: " + str(msg.pitch))
            self.pitchwheel = msg.pitch
            ARTIPHON = 1
            if ARTIPHON:
                self.pitchwheel *= 2
            amountchange = self.pitchwheel / 8192.0
            self.pitchwheelReal = pow(2, amountchange)
            print("PWREAL " + str(self.pitchwheelReal))
            #self.setAllIncrements()

        elif msg.type == 'control_change':

            event = "control[" + str(msg.control) + "]"

        elif msg.type == 'polytouch':
            self.polytouch = msg.value
            self.polytouchReal = msg.value/127.0

        elif msg.type == 'aftertouch':
            self.aftertouch = msg.value
            self.aftertouchReal = msg.value/127.0

            self.setAllIncrements()
            #for voice in self.voices:
            #	if time.time() - voice.note.releaseTime > max(voice.envTimeSeconds[3,:]):
            #		voice.setAllIncrements(self.pitchwheelReal * (1 + self.aftertouchReal))


        #if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
        #    # implement rising mono rate
        #    for heldnote in self.allNotes[::-1]:
        #        if heldnote.held and self.polyphony == self.voicesPerCluster :
        #            self.midi2commands(heldnote.msg)
        #            break

    def run(self):
        timer = 0
        # We create a fence.
        fenceCreateInfo = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=0
        )
        self.fence = vkCreateFence(self.device.vkDevice, fenceCreateInfo, None)

        # precompute some arrays
        self.fullAddArray = (
            np.zeros((int(self.noteBasePhase.size / 4)), dtype=np.float32))

        if self.PYSOUND:
            self.stream.start()

        hm = np.ones((4 * self.PARTIALS_PER_VOICE), dtype=np.float32)
        #for i in range(int(self.PARTIALS_PER_VOICE/2)):
        #    hm[4*i*2]= 1.5
        hm[4]  = 1
        hm[8]  = 1
        hm[12] = 1
        hm[16] = 2.01
        hm[20] = 2.01
        hm[24] = 2.01
        hm[28] = 2.01
        hm[32] = 3.01
        hm[36] = 3.01
        hm[40] = 3.01
        hm[44] = 3.01
        hm[48] = 4.01
        hm[52] = 4.01
        hm[56] = 4.01
        hm[60] = 4.01
        hm[64] = 5.01
        hm[68] = 5.01
        hm[72] = 5.01
        hm[76] = 5.01
        hm[80] = 6.01
        hm[84] = 6.01
        hm[88] = 6.01
        hm[92] = 6.01
        self.partialMultiplier.setBuffer(hm)

        hm2 = np.ones((4 * self.POLYPHONY), dtype=np.float32)
        self.partialVolume.setBuffer(hm2)

        newArray = self.fullAddArray.copy()
        self.replaceDict["FENCEADDR"] = hex(eval(str(self.fence).split(' ')[-1][:-1]))
        self.replaceDict["DEVADDR"]   = str(self.device.vkDevice).split(' ')[-1][:-1]
        self.replaceDict["SUBMITINFOADDR"]   = str(ffi.addressof(self.submitInfo)).split(' ')[-1][:-1]
        # compile the C-code
        if self.SOUND:
            if os.path.exists("./alsatonic"):
                os.remove("./alsatonic")
            header = ""
            for k, v in self.replaceDict.items():
                header += "#define " + k + " " + str(v) + "\n"
            with open(os.path.join(here, "resources", "alsatonic.template"), 'r') as f:
                at = header + f.read()
            cfilename = os.path.join(here, "resources", "alsatonic.c")
            with open(cfilename, 'w+') as f:
                f.write(at)
            os.system("g++ " + cfilename + " -o alsatonic -lm -lasound")
            for i in range(1):
                if os.path.exists("./alsatonic"):
                    os.system("taskset -c 15 ./alsatonic")
                else:
                    die
                break
        
        # start middle A note
        #self.midi2commands(mido.Message('note_on', note=50, velocity=64, time=6.2))
        #print(np.frombuffer(self.partialVolume.pmap   , np.float32)[::4])
        #print(np.frombuffer(self.partialMultiplier.pmap, np.float32)[::4])
        #print(np.frombuffer(self.noteBaseIncrement.pmap     , np.float32)[::4])
        
        
        np.set_printoptions(threshold=sys.maxsize)
        # into the loop
        #for i in range(int(1024 * 128 / self.SAMPLES_PER_DISPATCH)):
        while(1):
            # We submit the command buffer on the queue, at the same time giving a fence.
            vkQueueSubmit(self.device.compute_queue, 1, self.submitInfo, self.fence)

            # The command will not have finished executing until the fence is signalled.
            # So we wait here.
            # We will directly after this read our buffer from the GPU,
            # and we will not be sure that the command has finished executing unless we wait for the fence.
            # Hence, we use a fence here.
            vkWaitForFences(self.device.vkDevice, 1, [self.fence], VK_TRUE, 100000000000)

            # we do CPU tings simultaneously
            newArray += self.fullAddArray
            #np.fmod(newArray, 2*np.pi, out=newArray)

            self.noteBasePhase.setBuffer(newArray)

            pa = np.frombuffer(self.pcmBufferOut.pmap, np.float32)[::4]
            #pa = np.reshape(pa, (self.SAMPLES_PER_DISPATCH, self.POLYPHONY))
            #pa = np.sum(pa, axis = 1)
            pa2 = np.ascontiguousarray(pa)
            # pa2 = pa #np.ascontiguousarray(pa)
            # pa3 = np.vstack((pa2, pa2))
            # pa4 = np.swapaxes(pa3, 0, 1)
            # pa5 = np.ascontiguousarray(pa4)
            # print(np.shape(pa5))
            if self.PYSOUND:
                self.stream.write(pa2)

            self.mm.eventLoop(self)

            if self.GRAPH:
                print(pa2)
                plt.plot(pa2)
                plt.ylabel("some numbers")
                plt.show()


            vkResetFences(device=self.device.vkDevice, fenceCount=1, pFences=[self.fence])

        vkDestroyFence(self.device.vkDevice, self.fence, None)
        # elegantly free all memory
        self.instance_inst.release()


if __name__ == "__main__":
    Graph = False
    args = sys.argv
    for i, arg in enumerate(args):
        if arg == "-g":
            Graph = eval(args[i+1])
    s = Synth(Graph)
    s.run()
