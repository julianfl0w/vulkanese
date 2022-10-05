import ctypes
import os
import time
import json
from vulkan import *

try:
    from vulkanese import *
    from buffer import *
    from stage import *
    from commandbuffer import *
except:
    from .vulkanese import *
    from .buffer import *
    from .stage import *
    from .commandbuffer import *
from PIL import Image as pilImage
import re

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


# THIS CONTAINS EVERYTHING YOU NEED!
# The Vulkanese Compute Pipeline includes the following componenets
# command buffer
# pipeline,
# shader
# All in one. it is self-contained
class ComputePipeline(Pipeline):
    def __init__(self, computeShader, device, constantsDict, workgroupShape=[1, 1, 1]):
        Sinode.__init__(self)
        self.device = device
        device.instance.children += [self]

        #######################################################
        # Pipeline
        device.descriptorPool.finalize()
        Pipeline.__init__(self, device, stages=[computeShader], outputClass="image")

        self.descriptorSet = device.descriptorPool.descSetGlobal

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
        pipelines = vkCreateComputePipelines(
            self.vkDevice, VK_NULL_HANDLE, 1, self.pipelineCreateInfo, None
        )
        if len(pipelines) == 1:
            self.vkPipeline = pipelines[0]

        self.children += [pipelines]
        # wrap it all up into a command buffer
        self.commandBuffer = ComputeCommandBuffer(self, workgroupShape=workgroupShape)

        device.children += [self]

        # Now we shall finally submit the recorded command buffer to a queue.
        self.submitInfo = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,  # submit a single command buffer
            pCommandBuffers=[
                self.commandBuffer.vkCommandBuffers[0]
            ],  # the command buffer to submit.
        )

        # We create a fence.
        # So the CPU can know when processing is done
        fenceCreateInfo = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=0
        )
        self.fence = vkCreateFence(self.device.vkDevice, fenceCreateInfo, None)

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
    def run(self):

        # We submit the command buffer on the queue, at the same time giving a fence.
        vkQueueSubmit(
            queue=self.device.compute_queue,
            submitCount=1,
            pSubmits=self.submitInfo,
            fence=self.fence,
        )

        # The command will not have finished executing until the fence is signalled.
        # So we wait here.
        # We will directly after this read our buffer from the GPU,
        # and we will not be sure that the command has finished executing unless we wait for the fence.
        # Hence, we use a fence here.
        vkWaitForFences(
            device=self.device.vkDevice,
            fenceCount=1,
            pFences=[self.fence],
            waitAll=VK_TRUE,
            timeout=1000000000,
        )
        vkResetFences(device=self.device.vkDevice, fenceCount=1, pFences=[self.fence])

    def release(self):
        vkDestroyFence(self.device.vkDevice, self.fence, None)
