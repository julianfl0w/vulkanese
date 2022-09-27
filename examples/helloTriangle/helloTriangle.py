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
        
        # constants declared here will be visible in 
        # this class as an attribute,
        # and in the shader as a #define
        self.constantsDict = {
            "VERTEX_COUNT": len(pyramidMesh.vertices)
        }
        for k, v in self.constantsDict.items():
            exec("self." + k + " = " + str(v))

        #######################################################
        # Pipeline
        # create raster pipeline
        rasterPipeline = RasterPipeline(
            device=self.device,
            constantsDict=self.constantsDict,
            outputClass="surface",
            outputWidthPixels=700,
            outputHeightPixels=700,
        )
        # -- ADD BUFFERS HERE --
        # we will be using the standard set
        rasterPipeline.createStandardBuffers()
        # option here to change the code in each shader
        # (vertex -> tesselate -> fragment)
        rasterPipeline.createStages()
        # create the standard set
        rasterPipeline.createGraphicPipeline()
        
        
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