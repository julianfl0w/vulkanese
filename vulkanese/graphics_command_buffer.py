import ctypes
import os
import time
import json
import vulkan as vk
import sinode
from . import synchronization
import numpy as np


class GraphicsCommandBuffer(sinode.Sinode):
    def __init__(self, device, pipeline, renderpass, vkCommandBuffer, surface, index,
        waitStages=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,):
        sinode.Sinode.__init__(self, pipeline)
        self.device = device
        self.renderpass = renderpass
        self.pipeline = pipeline
        self.surface = surface
        self.waitStages = waitStages
        self.vkCommandBuffer = vkCommandBuffer

        # we're also gonna store corresponding render pass info here

        self.subresourceRange = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )

        self.components = vk.VkComponentMapping(
            r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
        )

        self.vkImageViewCreateInfo = vk.VkImageViewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=self.surface.vkSwapchainImages[index],
            flags=0,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=self.surface.surface_format.format,
            components=self.components,
            subresourceRange=self.subresourceRange,
        )

        self.imageView = vk.vkCreateImageView(self.device.vkDevice, self.vkImageViewCreateInfo, None)

            
        # Create Graphics render pass
        self.render_area = vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=self.surface.extent)
        # Framebuffers creation
        self.framebuffer_create = vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            flags=0,
            renderPass=renderpass.vkRenderPass,
            attachmentCount=1,
            pAttachments=[self.imageView],
            width=self.surface.WIDTH,
            height=self.surface.HEIGHT,
            layers=1,
        )

        self.vkFramebuffer = vk.vkCreateFramebuffer(
            device.vkDevice, self.framebuffer_create, None
        )

        self.color = vk.VkClearColorValue(float32=[0, 1, 0, 1])
        self.clear_value = vk.VkClearValue(color=self.color)

        self.vkRenderPassBeginInfo = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=renderpass.vkRenderPass,
            framebuffer=self.vkFramebuffer,
            renderArea=self.render_area,
            clearValueCount=1,
            pClearValues=[self.clear_value],
        )

        # each command buffer gets a couple semaphores
        self.renderSemaphore  = synchronization.Semaphore(device=self.device)
        self.presentSemaphore = synchronization.Semaphore(device=self.device)
        self.recordBuffer()

        
        # Information describing the queue submission
        self.vkSubmitInfo = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.presentSemaphore.vkSemaphore],
            #waitSemaphoreCount=0,
            #pWaitSemaphores=None,
            pWaitDstStageMask=self.waitStages,
            commandBufferCount=1,
            pCommandBuffers=[self.vkCommandBuffer],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.renderSemaphore.vkSemaphore],
        )

        print(self.surface.swapchain)
        # presentation creator
        self.vkPresentInfoKHR = vk.VkPresentInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.renderSemaphore.vkSemaphore], # wait on the render before presenting
            swapchainCount=1,
            pSwapchains=[self.surface.swapchain],
            pImageIndices=[0],
            pResults=None,
        )
    def recordBuffer(self):
        
        # start recording commands into it
        vkCommandBufferBeginInfo = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            # flags=vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
            flags=vk.VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
            pInheritanceInfo=None,
        )

        vk.vkBeginCommandBuffer(self.vkCommandBuffer, vkCommandBufferBeginInfo)
        
        # bind the buffers  
        print([b.name for b in self.pipeline.allVertexBuffers])
        #die
        vk.vkCmdBindVertexBuffers(
            commandBuffer=self.vkCommandBuffer,
            firstBinding=0,
            bindingCount=len(self.pipeline.allVertexBuffers),
            pBuffers=[b.vkBuffer for b in self.pipeline.allVertexBuffers],
            pOffsets=[0] * len(self.pipeline.allVertexBuffers),
        )

        vk.vkCmdBindIndexBuffer(
            commandBuffer=self.vkCommandBuffer,
            buffer=self.pipeline.indexBuffer.vkBuffer,
            offset=0,
            indexType=vk.VK_INDEX_TYPE_UINT32,
        )
        
        vk.vkCmdBeginRenderPass(
            self.vkCommandBuffer,
            self.vkRenderPassBeginInfo,
            vk.VK_SUBPASS_CONTENTS_INLINE,
        )
        # Bind graphicsPipeline
        vk.vkCmdBindPipeline(
            self.vkCommandBuffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.vkPipeline,
        )
        
        # Draw
        # void vkCmdDraw(
        # 	self.vkCommandBuffer commandBuffer,
        # 	uint32_t        vertexCount,
        # 	uint32_t        instanceCount,
        # 	uint32_t        firstVertex,
        # 	uint32_t        firstInstance);
        # vkCmdDraw(self.vkCommandBuffer, 6400, 1, 0, 1)

        # void vkCmdDrawIndexed(
        # 	self.vkCommandBuffer                             commandBuffer,
        # 	uint32_t                                    indexCount,
        # 	uint32_t                                    instanceCount,
        # 	uint32_t                                    firstIndex,
        # 	int32_t                                     vertexOffset,
        # 	uint32_t                                    firstInstance);
        vk.vkCmdDrawIndexed(
            self.vkCommandBuffer,
            np.prod(self.pipeline.indexBuffer.dimensionVals),
            1,
            0,
            0,
            0,
        )

        # End
        vk.vkCmdEndRenderPass(self.vkCommandBuffer)
        vk.vkEndCommandBuffer(self.vkCommandBuffer)
        