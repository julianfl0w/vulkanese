#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import cv2 as cv
#import matplotlib
import matplotlib.pyplot as plt
import sounddevice as sd

GRAPH = True
SOUND = not GRAPH

here = os.path.dirname(os.path.abspath(__file__))
print(sys.path)

localtest = True
if localtest == True:
    vkpath = os.path.join(here, "..", "vulkanese")
    #sys.path.append(vkpath)
    sys.path = [vkpath] + sys.path
    print(vkpath)
    print(sys.path)
    from vulkanese import Instance
    from vulkanese import *
else:
    from vulkanese.vulkanese import *

# from vulkanese.vulkanese import *

# device selection and instantiation
instance_inst = Instance()
print("available Devices:")
# for i, d in enumerate(instance_inst.getDeviceList()):
# 	print("    " + str(i) + ": " + d.deviceName)
print("")

# choose a device
print("naively choosing device 0")
device = instance_inst.getDevice(0)


#######################################################

#WORKGROUP_SIZE = 1  # Workgroup size in compute shader.
#SAMPLES_PER_DISPATCH = 512

replaceDict = {
    "POLYPHONY": 1,
    "SINES_PER_VOICE": 1,
    "MINIMUM_FREQUENCY_HZ" : 20,
    "MAXIMUM_FREQUENCY_HZ" : 20000,
    #"SAMPLE_FREQUENCY"     : 48000,
    "SAMPLE_FREQUENCY"     : 44100,
    "UNDERVOLUME"     : 3,
    "CHANNELS"     : 1,
    "SAMPLES_PER_DISPATCH" : 32,
    "LATENCY_SECONDS" : 0.006
}

for k, v in replaceDict.items():
    exec(k + " = " + str(v))

if SOUND:
    stream = sd.Stream(
        samplerate=SAMPLE_FREQUENCY, 
        blocksize=SAMPLES_PER_DISPATCH, 
        device=None, 
        channels=CHANNELS, 
        dtype=np.float32, 
        latency=LATENCY_SECONDS, 
        extra_settings=None, 
        callback=None,
        finished_callback=None, 
        clip_off=None, 
        dither_off=None, 
        never_drop_input=None, 
        prime_output_buffers_using_stream_callback=None)



pcmBufferOut = Buffer(
    binding=0,
    device=device,
    type="float",
    descriptorSet=device.descriptorPool.descSetGlobal,
    qualifier="in",
    name="pcmBufferOut",
    readFromCPU = True,
    SIZEBYTES=4 * 4 * SAMPLES_PER_DISPATCH, # Actually this is the number of sine oscillators
    initData = np.zeros((4 * SAMPLES_PER_DISPATCH), dtype = np.float32),
    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
    location=0,
    format=VK_FORMAT_R32_SFLOAT,
)

phaseBuffer = Buffer(
    binding=1,
    device=device,
    type="float",
    descriptorSet=device.descriptorPool.descSetUniform,
    qualifier="",
    name="phaseBuffer",
    readFromCPU = True,
    SIZEBYTES=4 * 4 * POLYPHONY * SINES_PER_VOICE,
    initData = np.zeros((4 * POLYPHONY * SINES_PER_VOICE), dtype = np.float32),
    usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
    location=0,
    format=VK_FORMAT_R32_SFLOAT,
)

baseFrequency = Buffer(
    binding=2,
    device=device,
    type="float",
    descriptorSet=device.descriptorPool.descSetUniform,
    qualifier="",
    name="baseFrequency",
    SIZEBYTES=4 * 4 * POLYPHONY,
    initData = np.ones((4 * POLYPHONY), dtype = np.float32)*440/SAMPLE_FREQUENCY,
    usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
    location=0,
    format=VK_FORMAT_R32_SFLOAT,
)

harmonicMultiplier = Buffer(
    binding=3,
    device=device,
    type="float",
    descriptorSet=device.descriptorPool.descSetUniform,
    qualifier="",
    name="harmonicMultiplier",
    SIZEBYTES=4 * 4 * SINES_PER_VOICE,
    initData = np.ones((4 * SINES_PER_VOICE), dtype = np.float32),
    usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
    location=0,
    format=VK_FORMAT_R32_SFLOAT,
)


#harmonicsVolume = Buffer(
#    binding=4,
#    device=device,
#    type="float",
#    descriptorSet=device.descriptorPool.descSetGlobal,
#    qualifier="",
#    name="harmonicsVolume",
#    SIZEBYTES=4 * 4 * SINES_PER_VOICE,
#    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
#    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
#    location=0,
#    format=VK_FORMAT_R32_SFLOAT,
#)
#
#noteAge = Buffer(
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
#)
#
#ADSR = Buffer(
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
#)

header = """#version 450
#extension GL_ARB_separate_shader_objects : enable
"""
for k, v in replaceDict.items():
    header += "#define " + k + " " + str(v) + "\n"
header += "layout (local_size_x = POLYPHONY, local_size_y = SINES_PER_VOICE, local_size_z = 1 ) in;"
    
main = """

void main() {

  /*
  In order to fit the work into workgroups, some unnecessary threads are launched.
  We terminate those threads here.
  if(gl_GlobalInvocationID.x >= 0 || gl_GlobalInvocationID.y >= 0)
    return;
  */
  
  uint noteNo = gl_GlobalInvocationID.x;
  uint sineNo = gl_GlobalInvocationID.y;
  
  uint outindex = 0;
  float frequency_hz    = baseFrequency[noteNo];
  //float harmonicRatio   = harmonicMultiplier[sineNo];
  //float thisFreq = frequency_hz*harmonicRatio;
  //float frequency_hz    = 440;
  float increment = frequency_hz;// * (1.0 / SAMPLE_FREQUENCY);
  float phase = phaseBuffer[outindex];
  
  for (int i = 0; i<SAMPLES_PER_DISPATCH; i++)
  {
  
    //pcmBufferOut[outindex+i] = int((pow(2,(32-UNDERVOLUME))-1) * sin(3.141592*2*phase));
    pcmBufferOut[outindex+i] = sin(3.141592*2*phase);
    //pcmBufferOut[outindex+i] = frequency_hz;
    //pcmBufferOut[outindex+i] = int(phaseBuffer[outindex+i]);
    //pcmBufferOut[outindex+i] = int(outindex + i);
    
    phase += increment;
    
  }
  //float intPart = 0;
  //phaseBuffer[outindex] = modf(phase, intPart);
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
    buffers=[pcmBufferOut, phaseBuffer, baseFrequency, harmonicMultiplier],
)

#######################################################
# Pipeline
device.descriptorPool.finalize()

computePipeline = ComputePipeline(device=device, 
                                  workgroupShape = [POLYPHONY,SINES_PER_VOICE,1], 
                                  stages=[mandleStage])
device.children += [computePipeline]

# print the object hierarchy
print("Object tree:")

print(json.dumps(device.asDict(), indent=4))

# Now we shall finally submit the recorded command buffer to a queue.
submitInfo = VkSubmitInfo(
    sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
    commandBufferCount=1,  # submit a single command buffer
    pCommandBuffers=[
        computePipeline.commandBuffer.vkCommandBuffers[0]
    ],  # the command buffer to submit.
)

# We create a fence.
fenceCreateInfo = VkFenceCreateInfo(sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=0)
fence = vkCreateFence(device.vkDevice, fenceCreateInfo, None)

# precompute some arrays
fullAddArray = np.ones((int(phaseBuffer.size/4)), dtype = np.float32)*3.141592*2*SAMPLES_PER_DISPATCH * 440 / SAMPLE_FREQUENCY
print(np.shape(fullAddArray))

if SOUND:
    stream.start()

newArray = fullAddArray.copy()

# into the loop 
for i in range(int(1024*128/SAMPLES_PER_DISPATCH)):
    
    # we do CPU tings simultaneously
    newArray += fullAddArray
    phaseBuffer.setBuffer(newArray)
    print(newArray)
    
    pa = np.frombuffer(pcmBufferOut.pmap, np.float32)[::4]
    pa2 = np.ascontiguousarray(pa)
    #pa2 = pa #np.ascontiguousarray(pa)
    #pa3 = np.vstack((pa2, pa2))
    #pa4 = np.swapaxes(pa3, 0, 1)
    #pa5 = np.ascontiguousarray(pa4)
    #print(np.shape(pa5))
    if SOUND:
        stream.write(pa2)
        
    # We submit the command buffer on the queue, at the same time giving a fence.
    vkQueueSubmit(device.compute_queue, 1, submitInfo, fence)

    
    # The command will not have finished executing until the fence is signalled.
    # So we wait here.
    # We will directly after this read our buffer from the GPU,
    # and we will not be sure that the command has finished executing unless we wait for the fence.
    # Hence, we use a fence here.
    vkWaitForFences(device.vkDevice, 1, [fence], VK_TRUE, 100000000000)
    
    # CONSIDER DOUBLE BUFFER HERE

    if GRAPH:
        print(pa2[:16])
        plt.plot(pa2)
        plt.ylabel('some numbers')
        plt.show()
    
    vkResetFences(
        device = device.vkDevice,
        fenceCount = 1,
        pFences = [fence])

vkDestroyFence(device.vkDevice, fence, None)
# elegantly free all memory
instance_inst.release()
