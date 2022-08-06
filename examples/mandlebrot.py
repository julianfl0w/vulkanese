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

WIDTH = 3200  # Size of rendered mandelbrot set.
HEIGHT = 2400  # Size of renderered mandelbrot set.
WORKGROUP_SIZE = 32  # Workgroup size in compute shader.

imageBuffer = Buffer(
    device=device,
    type="vec3",
    descriptorSet = device.descriptorPool.descSetGlobal,
    qualifier="out",
    name="imageBuffer",
    SIZEBYTES=4 * 4 * WIDTH * HEIGHT,
    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
    location=0,
)

header = """
#version 450
#extension GL_ARB_separate_shader_objects : enable

#define WIDTH 3200
#define HEIGHT 2400
#define WORKGROUP_SIZE 32
layout (local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1 ) in;

"""

main = """

struct Pixel
{
    vec4 value;
};

layout(std140, binding = 0) buffer buf
{
   Pixel imageData[];
};

void main() {

  /*
  In order to fit the work into workgroups, some unnecessary threads are launched.
  We terminate those threads here.
  */
  if(gl_GlobalInvocationID.x >= WIDTH || gl_GlobalInvocationID.y >= HEIGHT)
    return;

  float x = float(gl_GlobalInvocationID.x) / float(WIDTH);
  float y = float(gl_GlobalInvocationID.y) / float(HEIGHT);

  /*
  What follows is code for rendering the mandelbrot set.
  */
  vec2 uv = vec2(x,y);
  float n = 0.0;
  vec2 c = vec2(-.445, 0.0) +  (uv - 0.5)*(2.0+ 1.7*0.2  ),
  z = vec2(0.0);
  const int M =128;
  for (int i = 0; i<M; i++)
  {
    z = vec2(z.x*z.x - z.y*z.y, 2.*z.x*z.y) + c;
    if (dot(z, z) > 2) break;
    n++;
  }

  // we use a simple cosine palette to determine color:
  // http://iquilezles.org/www/articles/palettes/palettes.htm
  float t = float(n) / float(M);
  vec3 d = vec3(0.3, 0.3 ,0.5);
  vec3 e = vec3(-0.2, -0.3 ,-0.5);
  vec3 f = vec3(2.1, 2.0, 3.0);
  vec3 g = vec3(0.0, 0.1, 0.0);
  vec4 color = vec4( d + e*cos( 6.28318*(f*t+g) ) ,1.0);

  // store the rendered mandelbrot set into a storage buffer:
  imageData[WIDTH * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x].value = color;
}
"""


# Stage
existingBuffers = [imageBuffer]
mandleStage = Stage(
    device=device,
    name="mandlebrot.comp",
    stage=VK_SHADER_STAGE_COMPUTE_BIT,
    existingBuffers=existingBuffers,
    outputWidthPixels=700,
    outputHeightPixels=700,
    header=header,
    main=main,
    buffers=[imageBuffer],
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

pa = np.frombuffer(imageBuffer.pmap, np.float32)
pa = pa.reshape((HEIGHT, WIDTH, 4))
pa *= 255

# Now we save the acquired color data to a .png.
image = Image.fromarray(pa.astype(np.uint8))
image.save("mandelbrot.png")

# elegantly free all memory
instance_inst.release()
