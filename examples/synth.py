#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import trimesh
import cv2 as cv
import open3d as o3d
import copy
from exutils import *
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
print(sys.path)

localtest = True
if localtest == True:
    vkpath = os.path.join(here, "..", "vulkanese")
    sys.path.append(vkpath)
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

WORKGROUP_SIZE = 1  # Workgroup size in compute shader.
SAMPLES_PER_DISPATCH = 512

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
)

pcmBufferOut = Buffer(
    binding=2,
    device=device,
    type="uint32_t",
    descriptorSet=device.descriptorPool.descSetGlobal,
    qualifier="in",
    name="pcmBufferOut",
    SIZEBYTES=4 * 4 * SAMPLES_PER_DISPATCH, # Actually this is the number of sine oscillators
    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
    location=0,
)

replaceDict = {
    "WORKGROUP_SIZE": WORKGROUP_SIZE,
    "MINIMUM_FREQUENCY_HZ" : 20,
    "MAXIMUM_FREQUENCY_HZ" : 20000,
    "SAMPLE_FREQUENCY"     : 48000,
    "UNDERVOLUME"     : 3,
    "SAMPLES_PER_DISPATCH" : SAMPLES_PER_DISPATCH
}

header = """
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1 ) in;
"""

main = """

void main() {

  /*
  In order to fit the work into workgroups, some unnecessary threads are launched.
  We terminate those threads here.
  if(gl_GlobalInvocationID.x >= 0 || gl_GlobalInvocationID.y >= 0)
    return;
  */
  
  uint32_t phaseindex = gl_GlobalInvocationID.x;
  uint32_t outindex = gl_GlobalInvocationID.x;
  float frequency_hz = 440;
  float increment = frequency_hz / SAMPLE_FREQUENCY;
  float phase = phaseBuffer[index];
  
  for (int i = 0; i<SAMPLES_PER_DISPATCH; i++)
  {
    pcmBufferOut[index] = (2**(32-UNDERVOLUME)-1) * sin(3.141592*2*phase);
    
    phase += increment;
    if(phase > 1){
        phase -= 1;
    }
  }
  //phaseBuffer[index] = phase;
  // Multiple shaders will pull from phase array, so it needs to be updated by host
}
"""

for k, v in replaceDict.items():
    header.replace(k, str(v))
    main.replace(k, str(v))

# Stage
existingBuffers = [phaseBuffer]
mandleStage = Stage(
    device=device,
    name="mandlebrot.comp",
    stage=VK_SHADER_STAGE_COMPUTE_BIT,
    existingBuffers=existingBuffers,
    outputWidthPixels=700,
    outputHeightPixels=700,
    header=header,
    main=main,
    buffers=[phaseBuffer],
)

#######################################################
# Pipeline
device.descriptorPool.finalize()

computePipeline = ComputePipeline(device=device, stages=[mandleStage])
device.children += [computePipeline]

# print the object hierarchy
print("Object tree:")
WIDTH = 3200  # Size of rendered mandelbrot set.
HEIGHT = 2400  # Size of renderered mandelbrot set.
WORKGROUP_SIZE = 32  # Workgroup size in compute shader.

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

# We submit the command buffer on the queue, at the same time giving a fence.
vkQueueSubmit(device.compute_queue, 1, submitInfo, fence)

# The command will not have finished executing until the fence is signalled.
# So we wait here.
# We will directly after this read our buffer from the GPU,
# and we will not be sure that the command has finished executing unless we wait for the fence.
# Hence, we use a fence here.
vkWaitForFences(device.vkDevice, 1, [fence], VK_TRUE, 100000000000)

vkDestroyFence(device.vkDevice, fence, None)

pa = np.frombuffer(phaseBuffer.pmap, np.float32)
pa = pa.reshape((HEIGHT, WIDTH, 4))
pa *= 255

# Now we save the acquired color data to a .png.
image = Image.fromarray(pa.astype(np.uint8))
image.save("mandelbrot.png")

# elegantly free all memory
instance_inst.release()
