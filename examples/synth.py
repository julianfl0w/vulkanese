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
            "POLYPHONY": 64,
            "SINES_PER_VOICE": 64,
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
                sum += vol * sin(phase*harmonicRatio)/(SINES_PER_VOICE*POLYPHONY);

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
        print(msg)

    def run(self):
        timer = 0
        # We create a fence.
        fenceCreateInfo = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=0
        )
        fence = vkCreateFence(self.device.vkDevice, fenceCreateInfo, None)

        # precompute some arrays
        fullAddArray = (
            np.ones((int(self.phaseBuffer.size / 4)), dtype=np.float32)
            * 3.141592
            * 2
            * self.SAMPLES_PER_DISPATCH
            * 440
            / self.SAMPLE_FREQUENCY
        )
        print(np.shape(fullAddArray))

        if self.SOUND:
            self.stream.start()

        hm = np.ones((4 * self.SINES_PER_VOICE), dtype=np.float32)
        hm[: int(len(hm) / 2)] *= 1.5
        self.harmonicMultiplier.setBuffer(hm)

        hm = np.ones((4 * self.POLYPHONY), dtype=np.float32)
        self.harmonicsVolume.setBuffer(hm)

        newArray = fullAddArray.copy()
        # into the loop
        for i in range(int(1024 * 128 / self.SAMPLES_PER_DISPATCH)):

            # we do CPU tings simultaneously
            newArray += fullAddArray
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

            self.mm.eventLoop(self)

            # The command will not have finished executing until the fence is signalled.
            # So we wait here.
            # We will directly after this read our buffer from the GPU,
            # and we will not be sure that the command has finished executing unless we wait for the fence.
            # Hence, we use a fence here.
            vkWaitForFences(self.device.vkDevice, 1, [fence], VK_TRUE, 100000000000)

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
