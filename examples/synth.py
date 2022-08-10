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

here = os.path.dirname(os.path.abspath(__file__))
print(sys.path)

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
        self.msg  = None
        self.defaultIncrement = 2**32 * (noteToFreq(index) / 96000.0)
        self.releaseTime = 0

# WORKGROUP_SIZE = 1  # Workgroup size in compute shader.
# SAMPLES_PER_DISPATCH = 512
class Synth:
    def __init__(self):

        self.mm = midiManager.MidiManager()

        self.GRAPH = False
        self.SOUND = not self.GRAPH

        # device selection and instantiation
        self.instance_inst = Instance()
        print("available Devices:")
        # for i, d in enumerate(instance_inst.getDeviceList()):
        # 	print("    " + str(i) + ": " + d.deviceName)
        print("")

        # choose a device
        print("naively choosing device 0")
        device = self.instance_inst.getDevice(0)
        self.device = device

        replaceDict = {
            "POLYPHONY": 1,
            "SINES_PER_VOICE": 128,
            "MINIMUM_FREQUENCY_HZ": 20,
            "MAXIMUM_FREQUENCY_HZ": 20000,
            # "SAMPLE_FREQUENCY"     : 48000,
            "SAMPLE_FREQUENCY": 44100,
            "UNDERVOLUME": 3,
            "CHANNELS": 1,
            "SAMPLES_PER_DISPATCH": 32,
            "LATENCY_SECONDS": 0.006,
        }
        for k, v in replaceDict.items():
            exec("self." + k + " = " + str(v))

        if self.SOUND:
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
            * self.SAMPLES_PER_DISPATCH,  # Actually this is the number of sine oscillators
            initData=np.zeros((4 * self.SAMPLES_PER_DISPATCH), dtype=np.float32),
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.phaseBuffer = Buffer(
            binding=1,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="phaseBuffer",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY * self.SINES_PER_VOICE,
            initData=np.zeros(
                (4 * self.POLYPHONY * self.SINES_PER_VOICE), dtype=np.float32
            ),
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.baseIncrement = Buffer(
            binding=2,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="baseIncrement",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            initData=np.ones((4 * self.POLYPHONY), dtype=np.float32)
            * 2
            * 3.141592
            * 440
            / self.SAMPLE_FREQUENCY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.harmonicMultiplier = Buffer(
            binding=3,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            name="harmonicMultiplier",
            readFromCPU=True,
            SIZEBYTES=4 * 4 * self.SINES_PER_VOICE,
            initData=np.ones((4 * self.SINES_PER_VOICE), dtype=np.float32),
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )

        self.harmonicsVolume = Buffer(
            binding=4,
            device=device,
            type="float",
            descriptorSet=device.descriptorPool.descSetUniform,
            qualifier="",
            readFromCPU=True,
            initData=np.ones((4 * self.POLYPHONY), dtype=np.float32),
            name="harmonicsVolume",
            SIZEBYTES=4 * 4 * self.POLYPHONY,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
            format=VK_FORMAT_R32_SFLOAT,
        )
        
        self.allNotes = [Note(index = i) for i in range(128)] 

        # noteAge = Buffer(
        #    binding=5,
        #    device=device,
        #    type="float",
        #    descriptorSet=device.descriptorPool.descSetGlobal,
        #    qualifier="",
        #    name="noteAge",
        #    SIZEBYTES=4 * 4 * SINES_PER_VOICE,
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
        #    SIZEBYTES=4 * 4 * SINES_PER_VOICE,
        #    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        #    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
        #    location=0,
        #    format=VK_FORMAT_R32_SFLOAT,
        # )

        header = """#version 450
        #extension GL_ARB_separate_shader_objects : enable
        """
        for k, v in replaceDict.items():
            header += "#define " + k + " " + str(v) + "\n"
        header += "layout (local_size_x = 1, local_size_y = SINES_PER_VOICE, local_size_z = 1 ) in;"

        main = """

        void main() {

          /*
          In order to fit the work into workgroups, some unnecessary threads are launched.
          We terminate those threads here.
          if(gl_GlobalInvocationID.x >= 0 || gl_GlobalInvocationID.y >= 0)
            return;
          */

          uint noteNo = gl_GlobalInvocationID.x;
          uint timeSlice = gl_GlobalInvocationID.y;

          uint outindex = 0;
          float increment = baseIncrement[noteNo];
          float phase = phaseBuffer[outindex] + (timeSlice * increment);
          float sum = 0;

          for (int voiceNo = 0; voiceNo<POLYPHONY; voiceNo++)
          {
              float vol = harmonicsVolume[voiceNo];
              for (int sineNo = 0; sineNo<SINES_PER_VOICE; sineNo++)
              {

                float harmonicRatio   = harmonicMultiplier[sineNo];
                //sum += vol * sin(phase*harmonicRatio)/(SINES_PER_VOICE*POLYPHONY);
                sum += vol * sin(phase*harmonicRatio)/(SINES_PER_VOICE);

              }
          }
          
          pcmBufferOut[timeSlice] = sum;
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
                self.phaseBuffer,
                self.baseIncrement,
                self.harmonicMultiplier,
                self.harmonicsVolume,
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

    def midi2commands(self, msg):
        
        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            #if self.sustain:
            #    self.toRelease[msg.note] = True
            #    return

            note = self.allNotes[msg.note] 
            note.velocity = 0 
            note.velocityReal = 0 
            #if note.cluster is not None:
            #    note.cluster.silenceAllOps()
            #note.cluster = None
            note.held = False
            note.releaseTime = time.time()
            #self.harmonicsVolume.pmap[msg.note*4:(msg.note+1)*4] = np.array([0                                                      ],dtype=np.float32) 
            self.harmonicsVolume.pmap[0:4] = np.array([0                                                      ],dtype=np.float32) 

        # if note on, spawn voices
        elif msg.type == "note_on":
            #print(msg)
            note = self.allNotes[msg.note]
            note.velocity     = msg.velocity
            note.velocityReal = (msg.velocity/127.0)**2
            note.held = True
            note.msg = msg
            #self.harmonicsVolume.pmap[msg.note*4:(msg.note+1)*4] = np.array([1                                                      ],dtype=np.float32) 
            #self.baseIncrement.pmap[msg.note*4:(msg.note+1)*4]   = np.array([2*3.141592*noteToFreq(msg.note) / self.SAMPLE_FREQUENCY],dtype=np.float32)
            self.harmonicsVolume.pmap[0:4] = np.array([1                                                      ],dtype=np.float32) 
            self.baseIncrement.pmap[0:4]   = np.array([2*3.141592*noteToFreq(msg.note) / self.SAMPLE_FREQUENCY],dtype=np.float32)
            self.fullAddArray[0] = 2*3.141592*noteToFreq(msg.note)*self.SAMPLES_PER_DISPATCH / self.SAMPLE_FREQUENCY
            print(2*3.141592*noteToFreq(msg.note) / self.SAMPLE_FREQUENCY)

        elif msg.type == 'pitchwheel':
            logger.debug("PW: " + str(msg.pitch))
            self.pitchwheel = msg.pitch
            ARTIPHON = 1
            if ARTIPHON:
                self.pitchwheel *= 2
            amountchange = self.pitchwheel / 8192.0
            self.pitchwheelReal = pow(2, amountchange)
            logger.debug("PWREAL " + str(self.pitchwheelReal))
            self.setAllIncrements()

        elif msg.type == 'control_change':

            logger.debug("control : " + str(msg.control) + " (" + dtfm.controlNum2Name[msg.control] +  "): " + str(msg.value))

            event = "control[" + str(msg.control) + "]"

            # forward some controls

            # route control3 to control 7 because sometimes 3 is volume control
            if msg.control == 3:
                self.midi2commands(mido.Message('control_change', control= 7, value = msg.value ))

            if msg.control == dtfm.ctrl_vibrato_env:
                dtfm.formatAndSend(dtfm.cmd_env_rate , self.lowestVoiceIndex, 7, [0] * self.polyphony)
                dtfm.formatAndSend(dtfm.cmd_env , self.lowestVoiceIndex, 7, [(msg.value/127.0)*2**29] * self.polyphony)
                dtfm.formatAndSend(dtfm.cmd_env_rate , self.lowestVoiceIndex, 7, [(msg.value/127.0)*2**29] * self.polyphony)

            if msg.control == dtfm.ctrl_tremolo_env:
                dtfm.formatAndSend(dtfm.cmd_env_rate , self.lowestVoiceIndex, 6, [0] * self.polyphony)
                dtfm.formatAndSend(dtfm.cmd_env , self.lowestVoiceIndex, 6, [(msg.value/127.0)*2**29] * self.polyphony)
                dtfm.formatAndSend(dtfm.cmd_env_rate , self.lowestVoiceIndex, 6, [(msg.value/127.0)*2**29] * self.polyphony)

            if msg.control == dtfm.ctrl_silence:
                for op in range(6):
                    dtfm.formatAndSend(dtfm.cmd_env , self.lowestVoiceIndex,      op, [0] * self.polyphony)
                    dtfm.formatAndSend(dtfm.cmd_env_rate , self.lowestVoiceIndex, op, [0] * self.polyphony)


            # OPERATOR CONCERNS
            if msg.control == dtfm.ctrl_sustain: 
                self.sustain  = msg.value
                if not self.sustain:
                    for note, release in enumerate(self.toRelease):
                        if release:
                            self.midi2commands(mido.Message('note_off', note = note, velocity = 0))
                    self.toRelease = [False]*MIDINOTES



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
        fence = vkCreateFence(self.device.vkDevice, fenceCreateInfo, None)

        # precompute some arrays
        self.fullAddArray = (
            np.ones((int(self.phaseBuffer.size / 4)), dtype=np.float32)
            * 3.141592
            * 2
            * self.SAMPLES_PER_DISPATCH
            * 440
            / self.SAMPLE_FREQUENCY
        )
        print(np.shape(self.fullAddArray))

        if self.SOUND:
            self.stream.start()

        hm = np.ones((4 * self.SINES_PER_VOICE), dtype=np.float32)
        hm[4] *= 1.5
        hm[8] *= 1.01
        hm[12] *= 0.99
        hm[16] *= 1.02
        hm[20] *= 0.98
        hm[24] *= 0.97
        hm[28] *= 1.03
        hm[32] *= 0.96
        hm[36] *= 1.51
        hm[40] *= 1.49
        hm[44] *= 1.52
        hm[48] *= 1.48
        hm[52] *= 2
        hm[56] *= 2.01
        hm[60] *= 1.98
        #hm[64] *= 1.97
        #hm[68] *= 2.03
        #hm[72] *= 1.96
        #hm[76] *= 2.51
        #hm[80] *= 2.49
        #hm[84] *= 2.52
        #hm[88] *= 2.48
        #hm[92] *= 4
        self.harmonicMultiplier.setBuffer(hm)

        hm2 = np.zeros((4 * self.POLYPHONY), dtype=np.float32)
        self.harmonicsVolume.setBuffer(hm2)

        newArray = self.fullAddArray.copy()
        # into the loop
        #for i in range(int(1024 * 128 / self.SAMPLES_PER_DISPATCH)):
        while(1):

            # we do CPU tings simultaneously
            newArray += self.fullAddArray
            self.phaseBuffer.setBuffer(newArray)

            pa = np.frombuffer(self.pcmBufferOut.pmap, np.float32)[::4]
            pa2 = np.ascontiguousarray(pa)
            # pa2 = pa #np.ascontiguousarray(pa)
            # pa3 = np.vstack((pa2, pa2))
            # pa4 = np.swapaxes(pa3, 0, 1)
            # pa5 = np.ascontiguousarray(pa4)
            # print(np.shape(pa5))
            if self.SOUND:
                self.stream.write(pa2)

            # We submit the command buffer on the queue, at the same time giving a fence.
            vkQueueSubmit(self.device.compute_queue, 1, self.submitInfo, fence)

            # The command will not have finished executing until the fence is signalled.
            # So we wait here.
            # We will directly after this read our buffer from the GPU,
            # and we will not be sure that the command has finished executing unless we wait for the fence.
            # Hence, we use a fence here.
            vkWaitForFences(self.device.vkDevice, 1, [fence], VK_TRUE, 100000000000)

            self.mm.eventLoop(self)

            if self.GRAPH:
                print(pa2[:16])
                plt.plot(pa2)
                plt.ylabel("some numbers")
                plt.show()

            vkResetFences(device=self.device.vkDevice, fenceCount=1, pFences=[fence])

        vkDestroyFence(self.device.vkDevice, fence, None)
        # elegantly free all memory
        self.instance_inst.release()


if __name__ == "__main__":
    s = Synth()
    s.run()
