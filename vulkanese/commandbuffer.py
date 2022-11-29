import json
from sinode import *
import os

here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *
import math


class CommandBuffer(Sinode):
    def __init__(self, pipeline):
        Sinode.__init__(self, pipeline)
        self.pipeline = pipeline
        self.vkCommandPool = pipeline.device.vkCommandPool
        self.device = pipeline.device
        self.vkDevice = pipeline.device.vkDevice
        self.outputWidthPixels = pipeline.outputWidthPixels
        self.outputHeightPixels = pipeline.outputHeightPixels
        self.commandBufferCount = 0

        # assume triple-buffering for surfaces
        if pipeline.outputClass == "surface":
            self.device.instance.debug(
                "allocating 3 command buffers, one for each image"
            )
            self.commandBufferCount += 3
        # single-buffering for images
        else:
            self.commandBufferCount += 1

        self.device.instance.debug(
            "Creating buffers of size "
            + str(self.outputWidthPixels)
            + ", "
            + str(self.outputHeightPixels)
        )

        # Create command buffers, one for each image in the triple-buffer (swapchain + framebuffer)
        # OR one for each non-surface pass
        self.vkCommandBuffers_create = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.vkCommandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=self.commandBufferCount,
        )

        self.vkCommandBuffers = vkAllocateCommandBuffers(
            self.vkDevice, self.vkCommandBuffers_create
        )

        # Information describing the queue submission
        self.submit_create = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=len(self.pipeline.wait_semaphores),
            pWaitSemaphores=[s.vkSemaphore for s in self.pipeline.wait_semaphores],
            pWaitDstStageMask=self.pipeline.wait_stages,
            commandBufferCount=1,
            pCommandBuffers=[self.vkCommandBuffers[0]],
            signalSemaphoreCount=len(self.pipeline.signal_semaphores),
            pSignalSemaphores=[s.vkSemaphore for s in self.pipeline.signal_semaphores],
        )


class RasterCommandBuffer(CommandBuffer):
    def __init__(self, pipeline):
        CommandBuffer.__init__(self, pipeline)
        # optimization to avoid creating a new array each time
        self.submit_list = ffi.new("VkSubmitInfo[1]", [self.submit_create])

        # Record command buffer
        for i, vkCommandBuffer in enumerate(self.vkCommandBuffers):

            vkCommandBuffer_begin_create = VkCommandBufferBeginInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
                pInheritanceInfo=None,
            )

            vkBeginCommandBuffer(vkCommandBuffer, vkCommandBuffer_begin_create)
            vkCmdBeginRenderPass(
                vkCommandBuffer,
                self.pipeline.renderPass.render_pass_begin_create[i],
                VK_SUBPASS_CONTENTS_INLINE,
            )
            # Bind graphicsPipeline
            vkCmdBindPipeline(
                vkCommandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.pipeline.vkPipeline,
            )

            # Provided by VK_VERSION_1_0
            allBuffers = self.pipeline.getAllBuffers()

            self.device.instance.debug("--- ALL BUFFERS ---")
            for i, buffer in enumerate(allBuffers):
                self.device.instance.debug("-------------------------")
                self.device.instance.debug(i)
            allVertexBuffers = [
                b
                for b in allBuffers
                if (b.usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT or b.name == "index")
            ]
            self.device.instance.debug("--- ALL VERTEX BUFFERS ---")
            for i, buffer in enumerate(allVertexBuffers):
                self.device.instance.debug("-------------------------")
                self.device.instance.debug(i)

            allVertexBuffersVk = [b.vkBuffer for b in allVertexBuffers]

            pOffsets = [0] * len(allVertexBuffersVk)
            self.device.instance.debug("pOffsets")
            self.device.instance.debug(pOffsets)
            vkCmdBindVertexBuffers(
                commandBuffer=vkCommandBuffer,
                firstBinding=0,
                bindingCount=len(allVertexBuffersVk),
                pBuffers=allVertexBuffersVk,
                pOffsets=pOffsets,
            )

            vkCmdBindIndexBuffer(
                commandBuffer=vkCommandBuffer,
                buffer=pipeline.indexBuffer.vkBuffer,
                offset=0,
                indexType=VK_INDEX_TYPE_UINT32,
            )

            # Draw
            # void vkCmdDraw(
            # 	VkCommandBuffer commandBuffer,
            # 	uint32_t        vertexCount,
            # 	uint32_t        instanceCount,
            # 	uint32_t        firstVertex,
            # 	uint32_t        firstInstance);
            # vkCmdDraw(vkCommandBuffer, 6400, 1, 0, 1)

            # void vkCmdDrawIndexed(
            # 	VkCommandBuffer                             commandBuffer,
            # 	uint32_t                                    indexCount,
            # 	uint32_t                                    instanceCount,
            # 	uint32_t                                    firstIndex,
            # 	int32_t                                     vertexOffset,
            # 	uint32_t                                    firstInstance);
            vkCmdDrawIndexed(vkCommandBuffer, 6400, 1, 0, 0, 0)

            # End
            vkCmdEndRenderPass(vkCommandBuffer)
            vkEndCommandBuffer(vkCommandBuffer)

    def draw_frame(self, image_index):
        self.submit_create.pCommandBuffers[0] = self.vkCommandBuffers[image_index]
        vkQueueSubmit(self.device.graphic_queue, 1, self.submit_list, None)

        self.pipeline.surface.present_create.pImageIndices[0] = image_index
        self.pipeline.vkQueuePresentKHR(
            self.device.presentation_queue, self.pipeline.surface.present_create
        )

        # Fix #55 but downgrade performance -1000FPS)
        vkQueueWaitIdle(self.device.presentation_queue)


class ComputeCommandBuffer(CommandBuffer):
    def __init__(
        self,
        pipeline,
        workgroupCount,
        flags=VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
    ):
        CommandBuffer.__init__(self, pipeline)
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
