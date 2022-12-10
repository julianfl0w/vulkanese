import ctypes
import os
import time
import json
from vulkan import *
import sinode
import re
from . import buffer
from . import command_buffer
from . import pipeline
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
class ComputePipeline(pipeline.Pipeline):
    def __init__(
        self,
        computeShader,
        device,
        constantsDict,
        workgroupCount=[1, 1, 1],
        waitSemaphores=[],
        waitStages=[],
    ):
        sinode.Sinode.__init__(self)

        self.device = device
        self.computeShader = computeShader
        device.children += [self]

        #######################################################
        # Pipeline
        pipeline.Pipeline.__init__(
            self, device, stages=[computeShader], waitSemaphores=waitSemaphores,
        )

        device.descriptorPool.finalize()

        # The pipeline layout allows the pipeline to access descriptor sets.
        # So we just specify the descriptor set layout we created earlier.
        pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=len(device.descriptorPool.descSets),
            pSetLayouts=[
                d.vkDescriptorSetLayout for d in device.descriptorPool.descSets
            ],
        )

        self.vkPipelineLayout = vkCreatePipelineLayout(
            self.vkDevice, pipelineLayoutCreateInfo, None
        )

        # Now let us actually create the compute pipeline.
        # A compute pipeline is very simple compared to a graphics pipeline.
        # It only consists of a single stage with a compute shader.
        # So first we specify the compute shader stage, and it's entry point(main).
        shaderStageCreateInfo = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=computeShader.vkShaderModule,
            pName="main",
        )

        self.pipelineCreateInfo = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=shaderStageCreateInfo,
            layout=self.vkPipelineLayout,
        )

        # Now, we finally create the compute pipeline.
        vkpipelines = vkCreateComputePipelines(
            self.vkDevice, VK_NULL_HANDLE, 1, self.pipelineCreateInfo, None
        )
        if len(vkpipelines) == 1:
            self.vkPipeline = vkpipelines[0]

        # self.children += [pipelines]
        # wrap it all up into a command buffer
        self.commandBuffer = command_buffer.ComputeCommandBuffer(
            device=device,
            pipeline=self,
            workgroupCount=workgroupCount,
            waitSemaphores=waitSemaphores,
            waitStages=waitStages,
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
        vkQueueSubmit(
            queue=self.device.compute_queue,
            submitCount=1,
            pSubmits=self.commandBuffer.submitInfo,
            fence=self.commandBuffer.fence.vkFence,
        )

        if blocking:
            self.wait()

    def wait(self):
        for fence in self.commandBuffer.fences:
            fence.wait()

    def release(self):
        for fence in self.fences:
            fence.release()

        for semaphore in self.signalSemaphores:
            semaphore.release()

        self.device.instance.debug("destroying pipeline")
        pipeline.Pipeline.release(self)

        self.device.instance.debug("destroying children")
        for child in self.children:
            child.release()
