import ctypes
import os
import time
import json
from vulkan import *
from .surface import *
from .stage import *
from .renderpass import *
from .commandbuffer import *
from .vutil import *
from .vulkanese import *
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
    ):
        self.indexBuffer = indexBuffer
        Sinode.__init__(self, device)
        self.location = 0
        self.outputClass = outputClass
        self.vkDevice = device.vkDevice
        self.device = device
        self.instance = device.instance
        self.outputWidthPixels = outputWidthPixels
        self.outputHeightPixels = outputHeightPixels
        # Add Stages
        self.stages = stages

        # Create semaphores
        semaphore_create = VkSemaphoreCreateInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, flags=0
        )
        self.semaphore_image_available = vkCreateSemaphore(
            self.vkDevice, semaphore_create, None
        )
        self.semaphore_render_finished = vkCreateSemaphore(
            self.vkDevice, semaphore_create, None
        )

        self.wait_stages = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        self.wait_semaphores = [self.semaphore_image_available]
        self.signal_semaphores = [self.semaphore_render_finished]

        # Create a surface, if indicated
        if outputClass == "surface":
            newSurface = Surface(self.device.instance, self.device, self)
            self.surface = newSurface
            self.children += [self.surface]

        self.vkAcquireNextImageKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkAcquireNextImageKHR"
        )
        self.vkQueuePresentKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkQueuePresentKHR"
        )

        self.children += self.stages

    def draw_frame(self):
        image_index = self.vkAcquireNextImageKHR(
            self.vkDevice,
            self.surface.swapchain,
            UINT64_MAX,
            self.semaphore_image_available,
            None,
        )
        self.commandBuffer.draw_frame(image_index)

    def getAllBuffers(self):
        allBuffers = []
        for stage in self.stages:
            allBuffers += stage.buffers

        return allBuffers

    def release(self):
        print("generic pipeline release")
        vkDestroySemaphore(self.vkDevice, self.semaphore_image_available, None)
        vkDestroySemaphore(self.vkDevice, self.semaphore_render_finished, None)

        for stage in self.stages:
            stage.release()

        vkDestroyPipeline(self.vkDevice, self.vkPipeline, None)
        vkDestroyPipelineLayout(self.vkDevice, self.vkPipelineLayout, None)

        if hasattr(self, "surface"):
            print("releasing surface")
            self.surface.release()

        if hasattr(self, "renderPass"):
            self.renderPass.release()

        self.commandBuffer.release()
