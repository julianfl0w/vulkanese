import json
from sinode import *
import os

here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *
from . import synchronization
import math
import numpy as np


class CommandBuffer(Sinode):
    def __init__(self, device):
        Sinode.__init__(self, device)

        self.device = device

        # Create command buffers, one for each image in the triple-buffer (swapchain + framebuffer)
        # OR one for each non-surface pass
        self.vkCommandBuffers_create = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=device.vkCommandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )

        self.vkCommandBuffer = vkAllocateCommandBuffers(
            device.vkDevice, self.vkCommandBuffers_create
        )[0]


class ComputeCommandBuffer(CommandBuffer):
    def __init__(
        self,
        device,
        pipeline,
        workgroupCount,
        waitSemaphores,
        waitStages,
        flags=VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
    ):
        CommandBuffer.__init__(
            self,
            device=device,
            pipeline=pipeline,
            waitSemaphores=waitSemaphores,
            waitStages=waitStages,
        )
        self.vkCommandBuffer = self.vkCommandBuffers[0]
        # Now we shall start recording commands into the newly allocated command buffer.
        beginInfo = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            # the buffer is only submitted and used once in this application.
            # flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            flags=VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
        )
        vkBeginCommandBuffer(self.vkCommandBuffer, beginInfo)

        # We need to bind a pipeline, AND a descriptor set before we dispatch.
        # The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
        vkCmdBindPipeline(
            self.vkCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline.vkPipeline,
        )
        vkCmdBindDescriptorSets(
            commandBuffer=self.vkCommandBuffer,
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_COMPUTE,
            layout=self.pipeline.vkPipelineLayout,
            firstSet=0,
            descriptorSetCount=len(
                self.pipeline.device.descriptorPool.activevkDescriptorSets
            ),
            pDescriptorSets=self.pipeline.device.descriptorPool.activevkDescriptorSets,
            dynamicOffsetCount=0,
            pDynamicOffsets=None,
        )

        # Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
        # The number of workgroups is specified in the arguments.
        # If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
        vkCmdDispatch(
            self.vkCommandBuffer,
            # int(math.ceil(WIDTH / float(WORKGROUP_SIZE))),  # int for py2 compatible
            # int(math.ceil(HEIGHT / float(WORKGROUP_SIZE))),  # int for py2 compatible
            workgroupCount[0],  # int for py2 compatible
            workgroupCount[1],  # int for py2 compatible
            workgroupCount[2],
        )

        vkEndCommandBuffer(self.vkCommandBuffer)


class RaytraceCommandBuffer(CommandBuffer):
    def __init__(self, pipeline):
        CommandBuffer.__init__(self, pipeline)

        vkCmdBindPipeline(
            cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, self.rtPipeline
        )
        vkCmdBindDescriptorSets(
            cmdBuf,
            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            self.rtPipelineLayout,
            0,
            size(descSets),
            descSets,
            0,
            nullptr,
        )

        vkCmdTraceRaysKHR(
            cmdBuf,
            self.shaderDict["rgen"].vkStridedDeviceAddressRegion,
            self.shaderDict["miss"].vkStridedDeviceAddressRegion,
            self.shaderDict["hit"].vkStridedDeviceAddressRegion,
            self.shaderDict["call"],
            self.outputWidthPixels,
            self.outputHeightPixels,
            1,
        )

        vkEndCommandBuffer(vkCommandBuffer)
        self.debug.endLabel(cmdBuf)

    # def drawPost(VkCommandBuffer cmdBuf):
    # 	m_debug.beginLabel(cmdBuf, "Post");
    #
    # 	setViewport(cmdBuf);
    #
    # 	auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
    # 	vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &aspectRatio);
    # 	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
    # 	vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
    # 	vkCmdDraw(cmdBuf, 3, 1, 0, 0);
