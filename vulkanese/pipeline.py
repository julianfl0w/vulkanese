import ctypes
import os
import time
import json
from sinode import *
from vulkan import *

from . import vulkanese
from . import synchronization

from PIL import Image as pilImage

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


# all pipelines contain:
# references to instance, device, etc
# at least 1 stage
# an output size
class Pipeline(Sinode):
    def __init__(
        self,
        device,
        indexBuffer=None,
        stages=[],
        outputClass="surface",
        outputWidthPixels=500,
        outputHeightPixels=500,
        waitSemaphores=[],
    ):

        # We create a fence.
        # So the CPU can know when processing is done
        self.waitSemaphores = waitSemaphores
        self.waitStages = []
        self.fence = synchronization.Fence(device=self.device)
        self.semaphore = synchronization.Semaphore(device=self.device)
        self.fences = [self.fence]
        self.signalSemaphores = [self.semaphore]

        self.indexBuffer = indexBuffer
        Sinode.__init__(self, device)
        self.location = 0
        self.outputClass = outputClass
        self.vkDevice = device.vkDevice
        self.device = device
        self.instance = device.instance
        self.outputWidthPixels = outputWidthPixels
        self.outputHeightPixels = outputHeightPixels
        self.shaders = []
        # Add Stages
        # if not compute
        # self.stages = stages

        # if not type(self) == "ComputePipeline":
        #    self.children += self.stages

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
