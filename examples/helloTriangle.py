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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import vulkanese as ve
import vulkan    as vk


class HelloTriangle:
    def __init__(self, device, constantsDict):
        self.device = device
        self.constantsDict = constantsDict
        for k, v in self.constantsDict.items():
            exec("self." + k + " = " + str(v))

        FragBuffer = (
            ve.buffer.StorageBuffer(
                device=self.device,
                name="fragColor",
                memtype="vec3",
                qualifier="readonly",
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
            ),
        )
        # Input buffers to the shader
        # These are Uniform Buffers normally,
        # Storage Buffers in DEBUG Mode
        # PIPELINE WILL CREATE ITS OWN INDEX BUFFER
        self.vertexBuffers = [
            ve.buffer.VertexBuffer(
                device=self.device,
                name="position",
                memtype="vec3",
                qualifier="readonly",
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
            ),
            ve.buffer.VertexBuffer(
                device=self.device,
                name="normal",
                memtype="vec3",
                qualifier="readonly",
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
            ),
            ve.buffer.VertexBuffer(
                device=self.device,
                name="color",
                memtype="vec3",
                qualifier="readonly",
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
            ),
            ve.buffer.VertexBuffer(
                device=self.device,
                name="index",
                dimensionVals=[self.TRIANGLE_COUNT, self.VERTS_PER_TRIANGLE],
                memtype="uint",
                usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                format=vk.VK_FORMAT_R32_UINT,
                stride=4,
            ),
            FragBuffer,
        ]

        self.fragmentBuffers = [
            ve.buffer.StorageBuffer(
                device=self.device,
                name="outColor",
                memtype="vec4",
                qualifier="readonly",
                dimensionVals=[self.VERTEX_COUNT],
                stride=16,
            ),
            FragBuffer,
        ]

        # (vertex -> tesselate -> fragment)

        # Vertex Stage
        self.vertexStage = VertexStage(
            device=self.device,
            parent=self,
            constantsDict=self.constantsDict,
            buffers=self.vertexBuffers,
            name="passthrough.vert",
        )

        # fragment stage
        self.fragmentStage = FragmentStage(
            device=self.device,
            parent=self,
            buffers=self.fragmentBuffers,
            constantsDict=self.constantsDict,
            name="passthrough.frag",
        )

        self.stages = [self.vertexStage, self.fragmentStage]

        # create the standard set
        self.rasterPipeline = RasterPipeline(
            device=self.device,
            constantsDict=self.constantsDict,
            outputClass="surface",
            outputWidthPixels=700,
            outputHeightPixels=700,
        )

        self.device.descriptorPool.finalize()
        self.device.children += [self.rasterPipeline]

        # print the object hierarchy
        print("Object tree:")
        print(json.dumps(self.device.asDict(), indent=4))

    def run(self):

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
            dtype=np.float32,
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
            for event in self.rasterPipeline.surface.getEvents():
                if event.type == sdl2.SDL_QUIT:
                    running = False
                    vkDeviceWaitIdle(self.device.vkDevice)
                    break

            # rotate the pyrimid
            R = pyramidMesh.get_rotation_matrix_from_xyz(
                (0, -np.pi / max(6 * fps_last, 1), 0)
            )
            pyramidMesh.rotate(R, center=(0, 0, TRANSLATION[2]))
            meshVert = np.asarray(pyramidMesh.vertices, dtype="f4")
            print(np.asarray(pyramidMesh.triangles, dtype="u4"))
            # the index buffer is the 3 vert indices of each triangle
            self.rasterPipeline.indexBuffer.set(
                np.asarray(pyramidMesh.triangles, dtype="u4").flatten()
            )

            self.rasterPipeline.vertexStage.gpuBuffers.position.set(meshVert)
            pyramidVerticesColorHSV[:, :, 0] = np.fmod(
                pyramidVerticesColorHSV[:, :, 0] + 0.01, 360
            )
            # pyramidVerticesColor =  cv.cvtColor(pyramidVerticesColorHSV, cv.COLOR_HSV2RGB)
            vp = pyramidVerticesColor.flatten()
            self.rasterPipeline.vertexStage.gpuBuffers.color.set(vp)
            # draw the frame!
            self.rasterPipeline.draw_frame()


if __name__ == "__main__":

    # constants declared here will be visible in
    # this class as an attribute,
    # and in the shader as a #define
    # we need 12 verts for a pyramid
    constantsDict = {
        "VERTEX_COUNT": 12,  # there are redundant verts
        "TRIANGLE_COUNT": 4,
        "VERTS_PER_TRIANGLE": 3,
        "SPATIAL_DIMENSIONS": 3,
        "COLOR_DIMENSIONS": 3,
    }

    # device selection and instantiation
    instance = ve.instance.Instance()
    print("naively choosing device 0")
    device = instance.getDevice(0)

    ht = HelloTriangle(device, constantsDict)
    ht.run()

    # elegantly free all memory
    instance.release()
