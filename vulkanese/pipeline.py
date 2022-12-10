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
    def __init__(
        self, device, waitSemaphores, shaders,
    ):

        Sinode.__init__(self, device)

        self.shaders = shaders
        self.device = device

        push_constant_ranges = vk.VkPushConstantRange(stageFlags=0, offset=0, size=0)

        self.pipelineCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            flags=0,
            setLayoutCount=0,
            pSetLayouts=None,
            pushConstantRangeCount=0,
            pPushConstantRanges=[push_constant_ranges],
        )

        self.vkPipelineLayout = vk.vkCreatePipelineLayout(
            device.vkDevice, self.pipelineCreateInfo, None
        )

        # Information describing the queue submission
        # Now we shall finally submit the recorded command buffer to a queue.
        if waitSemaphores == []:
            self.submitInfo = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=self.commandBuffer.commandBufferCount,
                pCommandBuffers=self.vkCommandBuffers,
                signalSemaphoreCount=len(self.signalSemaphores),
                pSignalSemaphores=[s.vkSemaphore for s in self.signalSemaphores],
                pWaitDstStageMask=waitStages,
            )
        else:
            self.submitInfo = VkSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=self.commandBufferCount,
                pCommandBuffers=self.vkCommandBuffers,
                waitSemaphoreCount=int(len(waitSemaphores)),
                pWaitSemaphores=[s.vkSemaphore for s in waitSemaphores],
                signalSemaphoreCount=len(self.signalSemaphores),
                pSignalSemaphores=[s.vkSemaphore for s in self.signalSemaphores],
                pWaitDstStageMask=waitStages,
            )

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
