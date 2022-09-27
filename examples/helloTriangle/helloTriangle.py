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
    vkpath = os.path.join(here, "..", "..", "vulkanese")
    sys.path.append(vkpath)
    from vulkanese import *
else:
    from vulkanese.vulkanese import *

# from vulkanese.vulkanese import *
class HelloTriangle:
    def __init__(self):
        # device selection and instantiation
        instance_inst = Instance()
        print("available Devices:")
        # for i, d in enumerate(instance_inst.getDeviceList()):
        # 	print("    " + str(i) + ": " + d.deviceName)
        print("")

        # choose a device
        print("naively choosing device 0")
        self.device = instance_inst.getDevice(0)

        # create the pyramid
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
        
        self.constantsDict = {
            "VERTEX_COUNT": len(pyramidMesh.vertices)
        }
        for k, v in self.constantsDict.items():
            exec("self." + k + " = " + str(v))

        # Input buffers to the shader
        # These are Uniform Buffers normally,
        # Storage Buffers in DEBUG Mode
        # PIPELINE WILL CREATE ITS OWN INDEX BUFFER
        vertexShaderInputBuffers = [
            {"name": "position", "type": "vec3", "dims": ["VERTEX_COUNT"]},
            {"name": "normal", "type": "vec3", "dims": ["VERTEX_COUNT"]},
            {"name": "color", "type": "vec3", "dims": ["VERTEX_COUNT"]},
        ]

        # any input buffers you want to exclude from debug
        # for example, a sine lookup table
        vertexShaderInputBuffersNoDebug = []

        # variables that are usually intermediate variables in the shader
        # but in DEBUG mode they are made visible to the CPU (as Storage Buffers)
        # so that you can view shader intermediate values :)
        vertexShaderDebuggableVars = []

        # the output of the compute shader,
        # which in our case is always a Storage Buffer
        vertexShaderOutputBuffers = [
            {"name": "fragColor", "type": "vec3", "dims": ["VERTEX_COUNT"]},
        ]
        
        vertexHeader = """
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
"""
        vertexMain = """
void main() {                         
    gl_Position = vec4(position, 1.0);
    fragColor = color;                
}                                     
"""

        fragHeader = """
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
"""
        fragMain = """
void main() {
    outColor = vec4(fragColor, 1.0);
}
"""
        
        #location = 0
        #outColor = Buffer(
        #    device=device,
        #    name="outColor",
        #    qualifier="out",
        #    binding=5,
        #    type="vec4",
        #    descriptorSet=device.descriptorPool.descSetGlobal,
        #    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        #    sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        #    SIZEBYTES=65536,
        #    format=VK_FORMAT_R32G32B32_SFLOAT,
        #    memProperties=VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        #    | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        #    location=location,
        #)
        
        fragShaderOutputBuffers = [
            {"name": "outColor", "type": "vec4", "dims": ["VERTEX_COUNT"]},
        ]
        
        #######################################################
        # Pipeline

        rasterPipeline = RasterPipeline(
            device=self.device,
            constantsDict=self.constantsDict,
            vertexShaderInputBuffers=vertexShaderInputBuffers,
            vertexShaderInputBuffersNoDebug=vertexShaderInputBuffersNoDebug,
            vertexShaderDebuggableVars=vertexShaderDebuggableVars,
            vertexShaderOutputBuffers=vertexShaderOutputBuffers,
            vertexHeader=vertexHeader,
            vertexMain=vertexMain,
            fragmentShaderInputBuffers=vertexShaderOutputBuffers, # In this case, they are the same
            fragmentShaderDebuggableVars=[],
            fragmentShaderOutputBuffers=fragShaderOutputBuffers,
            fragHeader=fragHeader,
            fragMain=fragMain,
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

    def run(self):
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

            # rotate the pyrimid
            R = pyramidMesh.get_rotation_matrix_from_xyz((0, -np.pi / max(6 * fps_last, 1), 0))
            pyramidMesh.rotate(R, center=(0, 0, TRANSLATION[2]))
            meshVert = np.asarray(pyramidMesh.vertices, dtype="f4")
            index.setBuffer(np.asarray(pyramidMesh.triangles, dtype="u4").flatten())

            position.setBuffer(meshVert)
            pyramidVerticesColorHSV[:, :, 0] = np.fmod(
                pyramidVerticesColorHSV[:, :, 0] + 0.01, 360
            )
            # pyramidVerticesColor =  cv.cvtColor(pyramidVerticesColorHSV, cv.COLOR_HSV2RGB)
            vp = pyramidVerticesColor.flatten()
            color.setBuffer(vp)
            # draw the frame!
            rasterPipeline.draw_frame()

        # elegantly free all memory
        instance_inst.release()

if __name__ == "__main__":
    ht = HelloTriangle()
    ht.run()