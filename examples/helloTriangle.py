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


class HelloTriangle:
    def __init__(self, device, constantsDict):

        self.width  = 700
        self.height = 700
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
            dimensionVals=[self.TRIANGLE_COUNT, self.VERTS_PER_TRIANGLE],
            stride=4,
        )
        
        #self.debugBuff = ve.buffer.DebugBuffer(
        #        device=self.device,
        #        name="debugBuff",
        #        memtype="float",
        #        qualifier="writeonly",
        #        dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
        #        stride=4,
        #        compress=True,
        #    )
        
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
                dimensionVals=[self.width, self.height],
                stride=16,
                compress=False,
            ),
            FragBuffer,
        ]

        # finalize the buffers
        # device.descriptorPool.finalize()

        # (vertex -> tesselate -> fragment)
        # Vertex Stage
        self.vertexStage = ve.shader.VertexStage(
            device=self.device,
            constantsDict=self.constantsDict,
            buffers=self.vertexBuffers,
            name="vertexStage",
            sourceFilename=os.path.join(htHere, "shaders", "passthrough_vert.c"),
        )

        FragBuffer.qualifier = "in"
        # fragment stage
        self.fragmentStage = ve.shader.FragmentStage(
            device=self.device,
            buffers=self.fragmentBuffers,
            constantsDict=self.constantsDict,
            name="fragmentStage",
            sourceFilename=os.path.join(htHere, "shaders", "passthrough_frag.c"),
        )

        self.surface = ve.surface.Surface(
            instance=self.device.instance,
            device=self.device,
            width=self.width,
            height=self.height,
        )
        #self.children += [self.surface]
            
        # create the standard set
        self.graphicsPipeline = ve.graphics_pipeline.GraphicsPipeline(
            device=self.device,
            buffers=self.vertexBuffers + self.fragmentBuffers,
            shaders=[self.vertexStage, self.fragmentStage],
            indexBuffer=self.indexBuffer,
            constantsDict=self.constantsDict,
            surface=self.surface,
            outputWidthPixels=700,
            outputHeightPixels=700,
        )

        # print the object hierarchy
        print("Object tree:")
        print(json.dumps(self.device.asDict(), indent=4))

    def run(self):

        # create the pyramid
        self.pyramid = shapes.Pyramid()
        self.graphicsPipeline.indexBuffer.set(np.array(self.pyramid.mesh.triangles, dtype = np.uint32).flatten())
        self.graphicsPipeline.vertexStage.gpuBuffers.position.set(
            np.array(self.pyramid.mesh.vertices).flatten()
        )
        self.graphicsPipeline.vertexStage.gpuBuffers.color.set(
            self.pyramid.verticesColorBGR
        )
        # self.graphicsPipeline.vertexStage.gpuBuffers.normal.set(self.pyramid.verticesColorBGR)

        print(self.graphicsPipeline.indexBuffer.getAsNumpyArray())
        print(self.graphicsPipeline.vertexStage.gpuBuffers.position.getAsNumpyArray())
        print(self.graphicsPipeline.vertexStage.gpuBuffers.color.getAsNumpyArray())

        self.graphicsPipeline.indexBuffer.flush()
        self.graphicsPipeline.vertexStage.gpuBuffers.position.flush()
        self.graphicsPipeline.vertexStage.gpuBuffers.color.flush()
        
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

            self.pyramid.rotate(fps_last)
            self.graphicsPipeline.vertexStage.gpuBuffers.position.set(
                np.array(self.pyramid.mesh.vertices).flatten()
            )
            self.graphicsPipeline.vertexStage.gpuBuffers.position.flush()
            #self.graphicsPipeline.vertexStage.gpuBuffers.color.set(
            #    self.pyramid.verticesColorBGR
            # )

            # get quit, mouse, keypress etc
            for event in self.graphicsPipeline.surface.getEvents():
                if event.type == sdl2.SDL_QUIT:
                    running = False
                    vk.vkDeviceWaitIdle(self.device.vkDevice)
                    break

            # the index buffer is the 3 vert indices of each triangle
            # print(self.pyramid.mesh.triangles)
            # print(ii)
            # print(np.array(self.pyramid.mesh.triangles))
            # print(self.graphicsPipeline.indexBuffer.getAsNumpyArray())
            # die
            # draw the frame!
            self.graphicsPipeline.draw_frame()


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
    instance = ve.instance.Instance(verbose=False)
    print("naively choosing device 0")
    device = instance.getDevice(0)

    ht = HelloTriangle(device, constantsDict)
    ht.run()

    # elegantly free all memory
    instance.release()
