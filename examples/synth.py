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

GRAPH = False
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
    "WORKGROUP_SIZE": 1,
    "MINIMUM_FREQUENCY_HZ" : 20,
    "MAXIMUM_FREQUENCY_HZ" : 20000,
    #"SAMPLE_FREQUENCY"     : 48000,
    "SAMPLE_FREQUENCY"     : 44100,
    "UNDERVOLUME"     : 3,
    "CHANNELS"     : 1,
    "SAMPLES_PER_DISPATCH" : 32,
    "LATENCY_SECONDS" : 0.010
}

for k, v in replaceDict.items():
    exec(k + " = " + str(v))

if SOUND:
    stream = sd.Stream(
        samplerate=SAMPLE_FREQUENCY, 
        blocksize=SAMPLES_PER_DISPATCH, 
        device=None, 
        channels=CHANNELS, 
        dtype=np.int32, 
        latency=LATENCY_SECONDS, 
        extra_settings=None, 
        callback=None,
        finished_callback=None, 
        clip_off=None, 
        dither_off=None, 
        never_drop_input=None, 
        prime_output_buffers_using_stream_callback=None)


phaseBuffer = Buffer(
    binding=0,
    device=device,
    type="float",
    descriptorSet=device.descriptorPool.descSetGlobal,
    qualifier="",
    name="phaseBuffer",
    SIZEBYTES=4 * 4 * SAMPLES_PER_DISPATCH,
    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
    location=0,
    format=VK_FORMAT_R32_SFLOAT,
)

#die
#phaseBuffer.pmap[:8*4] = fullAddArray[:8]

pcmBufferOut = Buffer(
    binding=1,
    device=device,
    type="int",
    descriptorSet=device.descriptorPool.descSetGlobal,
    qualifier="in",
    name="pcmBufferOut",
    SIZEBYTES=4 * 4 * SAMPLES_PER_DISPATCH, # Actually this is the number of sine oscillators
    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
    location=1,
    format=VK_FORMAT_R32_SINT,
)

header = """#version 450
#extension GL_ARB_separate_shader_objects : enable
"""
for k, v in replaceDict.items():
    header += "#define " + k + " " + str(v) + "\n"
header += "layout (local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1 ) in;"
    
main = """

void main() {

  /*
  In order to fit the work into workgroups, some unnecessary threads are launched.
  We terminate those threads here.
  if(gl_GlobalInvocationID.x >= 0 || gl_GlobalInvocationID.y >= 0)
    return;
  */
  
  //uint outindex = gl_GlobalInvocationID.x;
  uint outindex = 0;
  float frequency_hz = 440;
  float increment = frequency_hz / SAMPLE_FREQUENCY;
  float phase = phaseBuffer[outindex];
  
  for (int i = 0; i<SAMPLES_PER_DISPATCH; i++)
  {
  
    pcmBufferOut[outindex+i] = int((pow(2,(32-UNDERVOLUME))-1) * sin(3.141592*2*phase));
    //pcmBufferOut[outindex+i] = int(phaseBuffer[outindex+i]);
    //pcmBufferOut[outindex+i] = int(outindex + i);
    
    phase += increment;
    /*
    if(phase > 1){
        phase -= 1;
    }*/
  }
  //phaseBuffer[outindex] = phase;
  // Multiple shaders will pull from phase array, so it needs to be updated by host
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
    buffers=[phaseBuffer, pcmBufferOut],
)

#######################################################
# Pipeline
device.descriptorPool.finalize()

computePipeline = ComputePipeline(device=device, stages=[mandleStage])
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
a = SAMPLES_PER_DISPATCH * 440 / SAMPLE_FREQUENCY
addArray = np.array([a,a,a,a], dtype=np.float32)
fullAddArray = np.tile(addArray, int(phaseBuffer.size/16))
if SOUND:
    stream.start()

newArray = fullAddArray.copy()
# into the loop 
for i in range(int(1024*128/SAMPLES_PER_DISPATCH)):
    
    # we do CPU tings simultaneously
    #newArray = np.add(np.frombuffer(phaseBuffer.pmap, np.float32), fullAddArray)
    newArray += fullAddArray
    #newArray = np.modf(newArray)[0]
    phaseBuffer.setBuffer(newArray)

    pa = np.frombuffer(pcmBufferOut.pmap, np.int32)[::4]
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
