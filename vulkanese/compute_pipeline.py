import ctypes
import os
import time
import json
import vulkan as vk
import sinode
import re
from . import buffer
from . import synchronization

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


# THIS CONTAINS EVERYTHING YOU NEED!
# The Vulkanese Compute Pipeline includes the following componenets
# command buffer
# pipeline,
# shader
# All in one. it is self-contained
class ComputePipeline(sinode.Sinode):
    def __init__(
        self,
        computeShader,
        device,
        constantsDict,
        workgroupCount=[1, 1, 1],
        signalSemaphoreCount=0,
        useFence=False,
        waitSemaphores=[],
        waitStages=[],
    ):
        sinode.Sinode.__init__(self)

        self.device = device
        device.children += [self]

        # synchronization is owned by the pipeline (command buffer?)
        self.waitSemaphores = waitSemaphores
        self.waitStages = waitStages
        self.fence = None
        if useFence:
            self.fence = synchronization.Fence(device=self.device)
        self.signalSemaphores = []
        for semaphore in range(signalSemaphoreCount):
            self.signalSemaphores += [synchronization.Semaphore(device=self.device)]

        push_constant_ranges = vk.VkPushConstantRange(stageFlags=0, offset=0, size=0)

        # The pipeline layout allows the pipeline to access descriptor sets.
        # So we just specify the established descriptor set
        self.vkPipelineLayoutCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            flags=0,
            setLayoutCount=len(device.descriptorPool.descSets),
            pSetLayouts=[
                d.vkDescriptorSetLayout for d in device.descriptorPool.descSets
            ],
            pushConstantRangeCount=0,
            pPushConstantRanges=[push_constant_ranges],
        )

        self.vkPipelineLayout = vk.vkCreatePipelineLayout(
            device=device.vkDevice,
            pCreateInfo=self.vkPipelineLayoutCreateInfo,
            pAllocator=None,
        )

        self.vkComputePipelineCreateInfo = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=computeShader.vkPipelineShaderStageCreateInfo,
            layout=self.vkPipelineLayout,
        )

        # Now, we finally create the compute pipeline.
        self.vkPipeline = vk.vkCreateComputePipelines(
            device=device.vkDevice,
            pipelineCache=vk.VK_NULL_HANDLE,
            createInfoCount=1,
            pCreateInfos=[self.vkComputePipelineCreateInfo],
            pAllocator=None,
        )[0]

        # Now we shall start recording commands into the newly allocated command buffer.
        self.beginInfo = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            # the buffer is only submitted and used once in this application.
            # flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            flags=0,
        )

        # wrap it all up into a command buffer
        self.vkCommandBufferAllocateInfo = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=device.vkCommandPool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )

        self.vkCommandBuffer = vk.vkAllocateCommandBuffers(
            device=device.vkDevice, pAllocateInfo=self.vkCommandBufferAllocateInfo
        )[0]

        vk.vkBeginCommandBuffer(self.vkCommandBuffer, self.beginInfo)

        # We need to bind a pipeline, AND a descriptor set before we dispatch.
        # The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
        vk.vkCmdBindPipeline(
            self.vkCommandBuffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.vkPipeline,
        )

        vk.vkCmdBindDescriptorSets(
            commandBuffer=self.vkCommandBuffer,
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            layout=self.vkPipelineLayout,
            firstSet=0,
            descriptorSetCount=len(device.descriptorPool.activevkDescriptorSets),
            pDescriptorSets=device.descriptorPool.activevkDescriptorSets,
            dynamicOffsetCount=0,
            pDynamicOffsets=None,
        )

        # Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
        # The number of workgroups is specified in the arguments.
        # If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
        vk.vkCmdDispatch(
            self.vkCommandBuffer,
            workgroupCount[0],
            workgroupCount[1],
            workgroupCount[2],
        )

        vk.vkEndCommandBuffer(self.vkCommandBuffer)

        if len(waitSemaphores):
            pWaitSemaphores = [s.vkSemaphore for s in waitSemaphores]
        else:
            pWaitSemaphores = None

        if len(waitSemaphores):
            pSignalSemaphores = [s.vkSemaphore for s in self.signalSemaphores]
        else:
            pSignalSemaphores = None

        # Information describing the queue submission
        # Now we shall finally submit the recorded command buffer to a queue.
        self.submitInfo = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.vkCommandBuffer],
            waitSemaphoreCount=len(waitSemaphores),
            pWaitSemaphores=pWaitSemaphores,
            signalSemaphoreCount=len(self.signalSemaphores),
            pSignalSemaphores=pSignalSemaphores,
            pWaitDstStageMask=waitStages,
        )

    # this help if you run the main loop in C/C++
    # just use the Vulkan addresses!
    def getVulkanAddresses(self):
        addrDict = {}
        addrDict["FENCEADDR"] = hex(eval(str(self.fence).split(" ")[-1][:-1]))
        addrDict["DEVADDR"] = str(self.device.vkDevice).split(" ")[-1][:-1]
        addrDict["SUBMITINFOADDR"] = str(ffi.addressof(self.submitInfo)).split(" ")[-1][
            :-1
        ]
        return addrDict

    # the main loop
    def run(self, blocking=True):

        # We submit the command buffer on the queue, at the same time giving a fence.
        vk.vkQueueSubmit(
            queue=self.device.compute_queue,
            submitCount=1,
            pSubmits=self.submitInfo,
            fence=self.fence.vkFence,
        )

        if blocking:
            self.wait()

    def wait(self):
        self.fence.wait()

    def release(self):
        self.fence.release()

        for semaphore in self.signalSemaphores:
            semaphore.release()

        self.device.instance.debug("destroying children")
        for child in self.children:
            child.release()

        for semaphore in self.signalSemaphores:
            semaphore.release()

        vk.vkDestroyPipeline(self.device.vkDevice, self.vkPipeline, None)
        vk.vkDestroyPipelineLayout(self.device.vkDevice, self.vkPipelineLayout, None)
