import ctypes
import os
import time
import json
from sinode import *
import vulkan as vk

from . import synchronization

# all pipelines contain:
# at least 1 stage (shader)
class Pipeline(Sinode):

    def getAllBuffers(self):
        allBuffers = []
        for shader in self.shaders:
            allBuffers += shader.buffers

        return allBuffers

    def release(self):
        self.device.instance.debug("generic pipeline release")

        for shader in self.shaders:
            shader.release()

        for semaphore in self.signalSemaphores:
            semaphore.release()

        vkDestroyPipeline(self.vkDevice, self.vkPipeline, None)
        vkDestroyPipelineLayout(self.vkDevice, self.vkPipelineLayout, None)

        if hasattr(self, "surface"):
            self.device.instance.debug("releasing surface")
            self.surface.release()

        if hasattr(self, "renderPass"):
            self.renderPass.release()

        self.commandBuffer.release()
