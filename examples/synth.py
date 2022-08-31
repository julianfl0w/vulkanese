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
            "POLYPHONY": 32,
            "PARTIALS_PER_VOICE": 256,
            "MINIMUM_FREQUENCY_HZ": 20,
            "MAXIMUM_FREQUENCY_HZ": 20000,
            # "SAMPLE_FREQUENCY"     : 48000,
            "SAMPLE_FREQUENCY": 44100,
            "UNDERVOLUME": 3,
            "CHANNELS": 1,
            "SAMPLES_PER_DISPATCH": 64,
            "LATENCY_SECONDS": 0.006,
            "FILTER_STEPS" : 2048,
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
            name="partialVolume",
            SIZEBYTES=4 * 4 * self.PARTIALS_PER_VOICE,
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
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.noteStrikeTime = Buffer(
            binding=6,
            device=device,
            type="float64_t",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteStrikeTime",
            readFromCPU=True,
            SIZEBYTES=4 * 8 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.noteReleaseTime = Buffer(
            binding=7,
            device=device,
            type="float64_t",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="noteReleaseTime",
            readFromCPU=True,
            SIZEBYTES=4 * 8 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )
        
        
        self.currTime = Buffer(
            binding=8,
            device=device,
            type="float64_t",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="currTime",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )
        
        self.envelopeParams = Buffer(
            binding=9,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="envelopeParams",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )
        
        
        self.freqFilter = Buffer(
            binding=10,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="freqFilter",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.FILTER_STEPS,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )
        
        
        self.pitchFactor = Buffer(
            binding=11,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="pitchFactor",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )
        
        smoothness = 1300
        resonance  = 4
        sharpness  = 200
        self.freqFilterTotal = np.ones((2 * 4 * self.FILTER_STEPS),dtype=np.float32)
        self.freqFilterTotal[0:4 * self.FILTER_STEPS] = 1
        self.freqFilterTotal[4*self.FILTER_STEPS:4*2*self.FILTER_STEPS] = 0
        self.freqFilterTotal[4 * (self.FILTER_STEPS) -sharpness: 4 * (self.FILTER_STEPS) +sharpness] = resonance
        self.freqFilterTotal[:] = np.convolve(self.freqFilterTotal, np.array([1]*smoothness))[smoothness-1:]
        self.freqFilterTotal /= np.max(self.freqFilterTotal)

        if self.GRAPH:
            # to run GUI event loop
            plt.ion()
             
            # here we are creating sub plots
            self.figure, ax = plt.subplots(figsize=(10, 8))
            self.plot, = ax.plot(self.freqFilterTotal[0:4 * self.FILTER_STEPS])
            plt.ylabel("some numbers")
            plt.show()
            plt.ylim(-2, 2)
            
        header = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
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
              float increment = noteBaseIncrement[noteNo]*pitchFactor[0];
              float phase = noteBasePhase[noteNo] + (timeSlice * increment);

              float innersum = 0;
              for (uint partialNo = 0; partialNo<PARTIALS_PER_VOICE; partialNo++)
              {
                float vol = partialVolume[partialNo];

                float harmonicRatio   = partialMultiplier[partialNo];
                float thisIncrement = increment * harmonicRatio;
                
                if(thisIncrement < 3.14){
                    int indexInFilter = int(thisIncrement*(FILTER_STEPS/(3.14)));
                    innersum += vol * sin(phase*harmonicRatio) * freqFilter[indexInFilter];
                }

              }
              sum+=innersum*noteVol;
          }
          
          pcmBufferOut[timeSlice] = sum/64;//(PARTIALS_PER_VOICE*POLYPHONY);
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
                self.noteStrikeTime,
                self.noteReleaseTime,
                self.currTime,
                self.envelopeParams,
                self.freqFilter,
                self.pitchFactor,
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
        self.sustain = False
        self.toRelease = []
        self.modWheelReal = 0.25
        self.pitchwheelReal = 1
        
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
            if self.sustain:
                self.toRelease += [msg.note]
                return

            note = self.getNoteFromMidi(msg.note)
            note.velocity = 0 
            note.velocityReal = 0 
            note.midiIndex = -1
            #if note.cluster is not None:
            #    note.cluster.silenceAllOps()
            #note.cluster = None
            note.held = False
            note.releaseTime = time.time()
            
            self.noteBaseIncrement.pmap[note.index*16:note.index*16+4]   = \
                np.array([0],dtype=np.float32)
            self.noteBasePhase.pmap[note.index*16:note.index*16+4]   = \
                np.array([0],dtype=np.float32)
            self.fullAddArray[note.index*4] = 0
            self.noteVolume.pmap[note.index*16:note.index*16+4]   = \
                np.array([0],dtype=np.float32)
            self.noteReleaseTime.pmap[note.index*16:note.index*16+8]   = \
                np.array(time.time(),dtype=np.float64)
            #print("NOTE OFF" + str(note.index))
            #print(str(msg))
        # if note on, spawn voices
        elif msg.type == "note_on":
            #print(msg)
            
            # if its already going, despawn it
            #if self.getNoteFromMidi(msg.note).held:
            #    self.midi2commands(mido.Message('note_off', note=note, velocity=0, time=6.2))
            
            note = self.spawnVoice()
            note.velocity     = msg.velocity
            note.velocityReal = (msg.velocity/127.0)**2
            note.held = True
            note.msg = msg
            note.midiIndex = msg.note
            
            incrementPerSample = 2*3.141592*noteToFreq(msg.note) / self.SAMPLE_FREQUENCY
            self.noteBaseIncrement.pmap[note.index*16:note.index*16+4]   = \
                np.array([incrementPerSample],dtype=np.float32)
            self.noteBasePhase.pmap[note.index*16:note.index*16+4]   = \
                np.array([0],dtype=np.float32)
            self.noteVolume.pmap[note.index*16:note.index*16+4]   = \
                np.array([1],dtype=np.float32)
            self.noteStrikeTime.pmap[note.index*16:note.index*16+8]   = \
                np.array(time.time(),dtype=np.float64)
            self.fullAddArray[note.index*4] = incrementPerSample*self.SAMPLES_PER_DISPATCH

            #print("NOTE ON" + str(note.index))
            #print(str(msg))
            #print(note.index)
            #print(incrementPerSample)

        elif msg.type == 'pitchwheel':
            #print("PW: " + str(msg.pitch))
            self.pitchwheel = msg.pitch
            ARTIPHON = 0
            if ARTIPHON:
                self.pitchwheel *= 2
            amountchange = self.pitchwheel / 8192.0
            octavecount = (2/12)
            self.pitchwheelReal = pow(2, amountchange*octavecount)
            #print("PWREAL " + str(self.pitchwheelReal))
            #self.setAllIncrements()

        elif msg.type == 'control_change':

            event = "control[" + str(msg.control) + "]"
                
            #print(event)
            # sustain pedal
            if msg.control == 64:
                print(msg.value)
                if msg.value:
                    self.sustain = True
                else:
                    self.sustain = False
                    for note in self.toRelease:
                        self.midi2commands(mido.Message('note_off', note=note, velocity=0, time=6.2))
                    self.toRelease = []
            
            # mod wheel
            elif msg.control == 1:
                valReal = msg.value / 127.0
                print(valReal)
                self.modWheelReal = valReal

        elif msg.type == 'polytouch':
            self.polytouch = msg.value
            self.polytouchReal = msg.value/127.0

        elif msg.type == 'aftertouch':
            self.aftertouch = msg.value
            self.aftertouchReal = msg.value/127.0


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
        pv = np.ones((4 * self.PARTIALS_PER_VOICE), dtype=np.float32)
        #for i in range(int(self.PARTIALS_PER_VOICE/2)):
        #    hm[4*i*2]= 1.5
        np.set_printoptions(threshold=sys.maxsize)
        PARTIALS_PER_HARMONIC = 7
        for harmonic in range(int(self.PARTIALS_PER_VOICE / PARTIALS_PER_HARMONIC)):
            for partial_in_harmonic in range(PARTIALS_PER_HARMONIC):
                if partial_in_harmonic == 0 or True:
                    pv[(harmonic*PARTIALS_PER_HARMONIC+partial_in_harmonic)*4]  = 0.7 / pow(harmonic+1, 1.1)
                else:
                    pv[(harmonic*PARTIALS_PER_HARMONIC+partial_in_harmonic)*4]  = 1 / pow(harmonic+1, 1.1)
                hm[(harmonic*PARTIALS_PER_HARMONIC+partial_in_harmonic)*4]  = 1 + harmonic# + partial_in_harmonic*0.0001

        self.partialMultiplier.setBuffer(hm)
        self.partialVolume.setBuffer(pv)

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
        
        
        sineFilt = 4 + np.sin(3.141592*16*np.arange(4*self.FILTER_STEPS)/(4*self.FILTER_STEPS))
        sineFilt /= np.max(sineFilt)
        
        # into the loop
        #for i in range(int(1024 * 128 / self.SAMPLES_PER_DISPATCH)):
        while(1):
            # We submit the command buffer on the queue, at the same time giving a fence.
            vkQueueSubmit(self.device.compute_queue, 1, self.submitInfo, self.fence)

            # do CPU things. 
            # NO MEMORY ACCESS
            # NO PMAP
            startPoint =int((1-self.modWheelReal)*4*self.FILTER_STEPS)
            currFilt = self.freqFilterTotal[startPoint:startPoint+self.FILTER_STEPS*4]
            if self.GRAPH:
                #print(pa2)    
                # updating data values
                self.plot.set_ydata(currFilt)

                # drawing updated values
                self.figure.canvas.draw()

                # This will run the GUI event
                # loop until all UI events
                # currently waiting have been processed
                self.figure.canvas.flush_events()

            #sineFilt = np.roll(sineFilt, 10)
            #currFilt += sineFilt
            #cfm = np.max(currFilt)
            #if cfm:
            #    currFilt /= cfm
            
            # The command will not have finished executing until the fence is signalled.
            # So we wait here.
            # We will directly after this read our buffer from the GPU,
            # and we will not be sure that the command has finished executing unless we wait for the fence.
            # Hence, we use a fence here.
            vkWaitForFences(self.device.vkDevice, 1, [self.fence], VK_TRUE, 100000000000)

            # we do CPU tings simultaneously
            newArray += self.fullAddArray*self.pitchwheelReal
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


            vkResetFences(device=self.device.vkDevice, fenceCount=1, pFences=[self.fence])

            self.freqFilter.pmap[:] = currFilt
            self.currTime.pmap[0:8]   = \
                np.array(time.time(),dtype=np.float64)
            
            self.pitchFactor.pmap[0:4]   = \
                np.array([self.pitchwheelReal],dtype=np.float32)
            
            
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
