#!/bin/env python
import os
import time
import sys
import numpy as np
import json
import open3d as o3d
import copy
import sdl2
import shapes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import vulkanese as ve
import vulkan as vk
from sinode import Sinode

htHere = os.path.dirname(os.path.abspath(__file__))


class HelloTriangle(Sinode):
    def __init__(self, device, constantsDict):
        Sinode.__init__(self)
        self.device = device
        self.constantsDict = constantsDict
        for k, v in self.constantsDict.items():
            exec("self." + k + " = " + str(v))

        FragBuffer = ve.buffer.VertexBuffer(
            device=self.device,
            name="fragColor",
            memtype="vec3",
            qualifier="out",
            location=3,
            dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
            stride=12,
            compress=False,
        )

        self.indexBuffer = ve.buffer.IndexBuffer(
            device=self.device,
            name="index",
            dimensionVals=[self.TRIANGLE_COUNT, self.VERTS_PER_TRIANGLE],
            stride=4,
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
                qualifier="in",
                location=0,
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
                compress=False,
            ),
            ve.buffer.VertexBuffer(
                device=self.device,
                name="normal",
                memtype="vec3",
                qualifier="in",
                location=1,
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
                compress=False,
            ),
            ve.buffer.VertexBuffer(
                device=self.device,
                name="color",
                memtype="vec3",
                qualifier="in",
                location=2,
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
                compress=False,
            ),
            FragBuffer,
        ]

        self.fragmentBuffers = [
            ve.buffer.VertexBuffer(
                device=self.device,
                name="outColor",
                memtype="vec4",
                location=0,
                qualifier="out",
                dimensionVals=[self.VERTEX_COUNT],
                stride=16,
                compress=False,
            ),
            FragBuffer,
        ]

        # (vertex -> tesselate -> fragment)
        # Vertex Stage
        self.vertexStage = ve.shader.VertexStage(
            device=self.device,
            parent=self,
            constantsDict=self.constantsDict,
            buffers=self.vertexBuffers,
            name="vertexStage",
            sourceFilename=os.path.join(htHere, "shaders", "passthrough_vert.c"),
        )

        FragBuffer.qualifier = "in"
        # fragment stage
        self.fragmentStage = ve.shader.FragmentStage(
            device=self.device,
            parent=self,
            buffers=self.fragmentBuffers,
            constantsDict=self.constantsDict,
            name="fragmentStage",
            sourceFilename=os.path.join(htHere, "shaders", "passthrough_frag.c"),
        )

        # create the standard set
        self.rasterPipeline = ve.raster_pipeline.GraphicsPipeline(
            device=self.device,
            buffers=self.vertexBuffers + self.fragmentBuffers,
            shaders=[self.vertexStage, self.fragmentStage],
            indexBuffer=self.indexBuffer,
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
        self.pyramid = shapes.Pyramid()
        trianglesShape = np.shape(np.array(self.pyramid.mesh.triangles))
        trianglesLen = np.prod(trianglesShape)
        ii = (np.arange(trianglesLen * 4) / 4).astype(np.uint32)
        ptf = np.array(self.pyramid.mesh.triangles).flatten()
        print(ptf)
        print(self.rasterPipeline.indexBuffer.skipval)
        print(self.rasterPipeline.indexBuffer.itemSize)
        print(len(self.rasterPipeline.indexBuffer.pmap))
        self.rasterPipeline.indexBuffer.set(ptf.astype(np.uint32))
        self.rasterPipeline.vertexStage.gpuBuffers.position.set(
            np.array(self.pyramid.mesh.vertices).flatten()
        )
        self.rasterPipeline.vertexStage.gpuBuffers.color.set(
            self.pyramid.verticesColorBGR
        )
        # self.rasterPipeline.vertexStage.gpuBuffers.normal.set(self.pyramid.verticesColorBGR)

        print(self.rasterPipeline.indexBuffer.getAsNumpyArray())
        print(self.rasterPipeline.vertexStage.gpuBuffers.position.getAsNumpyArray())
        print(self.rasterPipeline.vertexStage.gpuBuffers.color.getAsNumpyArray())

        # Main loop
        last_time = 0
        fps = 60
        fps_last = 60
        running = True
        while running:

            # timing
            fps += 1
            if time.time() - last_time >= 1:
                last_time = time.time()
                print("FPS: %s" % fps)
                fps_last = fps
                fps = 0

            # self.pyramid.rotate(fps_last)

            # get quit, mouse, keypress etc
            for event in self.rasterPipeline.surface.getEvents():
                if event.type == sdl2.SDL_QUIT:
                    running = False
                    vk.vkDeviceWaitIdle(self.device.vkDevice)
                    break

            # the index buffer is the 3 vert indices of each triangle
            # print(self.pyramid.mesh.triangles)
            # print(ii)
            # print(np.array(self.pyramid.mesh.triangles))
            # print(self.rasterPipeline.indexBuffer.getAsNumpyArray())
            # die
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
    instance = ve.instance.Instance(verbose=True)
    print("naively choosing device 0")
    device = instance.getDevice(0)

    ht = HelloTriangle(device, constantsDict)
    ht.run()

    # elegantly free all memory
    instance.release()
