import ctypes
import os
import time
import json
from sinode import *
from vulkan import *

try:
    from vulkanese import *
    from semaphore import *
except:
    from .vulkanese import *
    from .semaphore import *
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
        self.shaders = []
        # Add Stages
        # if not compute
        #self.stages = stages

        # Create semaphores
        self.semaphore_image_available = Semaphore(device = self.device)
        self.semaphore_render_finished = Semaphore(device = self.device)
        self.semaphores = [self.semaphore_image_available, self.semaphore_render_finished]

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

        #if not type(self) == "ComputePipeline":
        #    self.children += self.stages

    def draw_frame(self):
        image_index = self.vkAcquireNextImageKHR(
            self.vkDevice,
            self.surface.swapchain,
            UINT64_MAX,
            self.semaphore_image_available.vkSemaphore,
            None,
        )
        self.commandBuffer.draw_frame(image_index)

    def getAllBuffers(self):
        allBuffers = []
        for shader in self.shaders:
            allBuffers += shader.buffers

        return allBuffers

    def release(self):
        self.device.instance.debug("generic pipeline release")
        
        for shader in self.shaders:
            shader.release()
            
        for semaphore in self.semaphores:
            semaphore.release()
        
        vkDestroyPipeline(self.vkDevice, self.vkPipeline, None)
        vkDestroyPipelineLayout(self.vkDevice, self.vkPipelineLayout, None)

        if hasattr(self, "surface"):
            self.device.instance.debug("releasing surface")
            self.surface.release()

        if hasattr(self, "renderPass"):
            self.renderPass.release()

        self.commandBuffer.release()
