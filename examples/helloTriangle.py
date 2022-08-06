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
# vertex stage
# buffers
location = 0
position = VertexBuffer(device=device,
    name="position",
   location=location)
location += position.getSize()
normal = VertexBuffer(device=device,
    name="normal",
   location=location)
location += normal.getSize()
color = VertexBuffer(device=device,
    name="color",
   location=location)
location += color.getSize()
index = VertexBuffer(
    name="index",
   location=location,
    device=device,
    type="uint",
    usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT
    | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    format=VK_FORMAT_R32_UINT,
    stride=4,
)
location += index.getSize()

fragColor = Buffer(device=device, type="vec3", qualifier="out",
    name="fragColor",
   location=location)

main = """
void main() {                         
    gl_Position = vec4(position, 1.0);
    fragColor = color;                
}                                     
"""

header = """
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
"""
# Stage
existingBuffers = [position, normal, color, index, fragColor]
vertex = Stage(
    device=device,
    name="passthrough.vert",
    stage=VK_SHADER_STAGE_VERTEX_BIT,
    existingBuffers = existingBuffers,
    outputWidthPixels=700,
    outputHeightPixels=700,
    header=header,
    main=main,
    buffers=[position, normal, color, index, fragColor],
)

#######################################################
# fragment stage
# buffers,
location=0
outColor = Buffer(
    device=device,
    name="outColor",
    qualifier="out",
    type="vec4",
    descriptorSet="global",
    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    SIZEBYTES=65536,
    format=VK_FORMAT_R32G32B32_SFLOAT,
    stride=12,
    rate=VK_VERTEX_INPUT_RATE_VERTEX,
    memProperties=VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
   location=location
)
location += fragColor.getSize()
outColor.location = 0
#fragColor.location= 1
fragColor.qualifier = "in"
# Stage
existingBuffers += [fragColor, outColor]
header = """
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
"""
main = """
void main() {
    outColor = vec4(fragColor, 1.0);
}
"""
fragment = Stage(
    device=device,
    name="passthrough.frag",
    existingBuffers = existingBuffers,
    stage=VK_SHADER_STAGE_FRAGMENT_BIT,
    outputWidthPixels=700,
    outputHeightPixels=700,
    header=header,
    main=main,
    buffers=[outColor, fragColor],
)

#######################################################
# Pipeline

rasterPipeline = RasterPipeline(
    device=device,
    indexBuffer = index,
    stages=[vertex, fragment],
    culling=VK_CULL_MODE_BACK_BIT,
    oversample=VK_SAMPLE_COUNT_1_BIT,
    outputClass="surface",
    outputWidthPixels=700,
    outputHeightPixels=700,
)
device.descriptorPool.finalize()
device.children += [rasterPipeline]

# print the object hierarchy
print("Object tree:")
print(json.dumps(device.asDict(), indent=4))

pyramidMesh = getPyramid()
TRANSLATION = (0.0, 0.5, 0.5)
pyramidMesh.translate(TRANSLATION)
pyramidVerticesColor = np.array(
    [
        [[1.0, 0.0, 0.0]] * 3
        + [[1.0, 1.0, 0.0]] * 3
        + [[0.0, 0.0, 1.0]] * 3
        + [[0.0, 1.0, 1.0]] * 3
    ],
    dtype=np.dtype("f4"),
)
pyramidVerticesColorHSV = cv.cvtColor(pyramidVerticesColor, cv.COLOR_BGR2HSV)
print(np.asarray(pyramidMesh.vertices))

# Main loop
clock = time.perf_counter
last_time = clock() * 1000
fps = 60
fps_last = 60
running = True
while running:
    # timing
    fps += 1
    if clock() * 1000 - last_time >= 1000:
        last_time = clock() * 1000
        print("FPS: %s" % fps)
        fps_last = fps
        fps = 0

    # get quit, mouse, keypress etc
    for event in rasterPipeline.surface.getEvents():
        if event.type == sdl2.SDL_QUIT:
            running = False
            vkDeviceWaitIdle(device.vkDevice)
            break

    R = pyramidMesh.get_rotation_matrix_from_xyz((0, -np.pi / max(6 * fps_last, 1), 0))
    pyramidMesh.rotate(R, center=(0, 0, TRANSLATION[2]))
    meshVert = np.asarray(pyramidMesh.vertices, dtype="f4")
    # print(np.asarray(pyramidMesh.vertices).flatten())
    index.setBuffer(
        np.asarray(pyramidMesh.triangles, dtype="u4").flatten()
    )

    position.setBuffer(meshVert)
    pyramidVerticesColorHSV[:, :, 0] = np.fmod(
        pyramidVerticesColorHSV[:, :, 0] + 0.01, 360
    )
    # pyramidVerticesColor =  cv.cvtColor(pyramidVerticesColorHSV, cv.COLOR_HSV2RGB)
    vp = pyramidVerticesColor.flatten()
    print(vp)
    color.setBuffer(vp)
    # draw the frame!
    rasterPipeline.draw_frame()

# elegantly free all memory
instance_inst.release()
