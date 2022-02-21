import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *

class CommandBuffer(PrintClass):
	def __init__(self, pipeline):
		PrintClass.__init__(self)
		self.pipeline = pipeline
		self.pipelineDict = pipeline.setupDict
		self.vkCommandPool  = pipeline.device.vkCommandPool
		self.device       = pipeline.device
		self.vkDevice     = pipeline.device.vkDevice
		self.outputWidthPixels  = self.pipelineDict["outputWidthPixels"]
		self.outputHeightPixels = self.pipelineDict["outputHeightPixels"]
		self.commandBufferCount = 0
		
		# assume triple-buffering for surfaces
		if self.pipelineDict["outputClass"] == "surface":
			print("allocating 3 command buffers, one for each image")
			self.commandBufferCount += 3 
		# single-buffering for images
		else:
			self.commandBufferCount += 1 
				
				
		print("Creating buffers of size " + str(self.outputWidthPixels) + 
			", " + str(self.outputHeightPixels))
			
		# Create command buffers, one for each image in the triple-buffer (swapchain + framebuffer)
		# OR one for each non-surface pass
		self.vkCommandBuffers_create = VkCommandBufferAllocateInfo(
			sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			commandPool=self.vkCommandPool,
			level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			commandBufferCount=self.commandBufferCount)

		self.vkCommandBuffers = vkAllocateCommandBuffers(self.vkDevice, self.vkCommandBuffers_create)
		
		# Information describing the queue submission
		self.submit_create = VkSubmitInfo(
			sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
			waitSemaphoreCount=len(self.pipeline.wait_semaphores),
			pWaitSemaphores=self.pipeline.wait_semaphores,
			pWaitDstStageMask=self.pipeline.wait_stages,
			commandBufferCount=1,
			pCommandBuffers=[self.vkCommandBuffers[0]],
			signalSemaphoreCount=len(self.pipeline.signal_semaphores),
			pSignalSemaphores=self.pipeline.signal_semaphores)

		# optimization to avoid creating a new array each time
		self.submit_list = ffi.new('VkSubmitInfo[1]', [self.submit_create])

		
		# Record command buffer
		for i, vkCommandBuffer in enumerate(self.vkCommandBuffers):

			vkCommandBuffer_begin_create = VkCommandBufferBeginInfo(
				sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
				pInheritanceInfo=None)

			vkBeginCommandBuffer(vkCommandBuffer, vkCommandBuffer_begin_create)
			vkCmdBeginRenderPass(vkCommandBuffer, self.pipeline.renderPass.render_pass_begin_create[i], VK_SUBPASS_CONTENTS_INLINE)
			# Bind graphicsPipeline
			vkCmdBindPipeline(vkCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.vkPipeline)

			
			# Provided by VK_VERSION_1_0
			allBuffers = self.pipeline.getAllBuffers()
			print("allBuffers " + str(allBuffers))
			allVertexBuffers = [b.vkBuffer for b in allBuffers if b.setupDict["usage"] == "VK_BUFFER_USAGE_VERTEX_BUFFER_BIT"]
			print("allVertexBuffers " + str(allVertexBuffers))
			
			vkCmdBindVertexBuffers(
				commandBuffer       = vkCommandBuffer,
				firstBinding        = 0,
				bindingCount        = len(allVertexBuffers),
				pBuffers            = allVertexBuffers,
				pOffsets            = [0]*len(allVertexBuffers));
				
			# Draw
			#void vkCmdDraw(
			#	VkCommandBuffer commandBuffer,
			#	uint32_t        vertexCount,
			#	uint32_t        instanceCount,
			#	uint32_t        firstVertex,
			#	uint32_t        firstInstance);
			vkCmdDraw(vkCommandBuffer, 3, 1, 0, 1)
			# End
			vkCmdEndRenderPass(vkCommandBuffer)
			vkEndCommandBuffer(vkCommandBuffer)
			
			

	def draw_frame(self, image_index):
		self.submit_create.pCommandBuffers[0] = self.vkCommandBuffers[image_index]
		vkQueueSubmit(self.device.graphic_queue, 1, self.submit_list, None)

		self.pipeline.surface.present_create.pImageIndices[0] = image_index
		self.pipeline.vkQueuePresentKHR(self.device.presentation_queue, self.pipeline.surface.present_create)

		# Fix #55 but downgrade performance -1000FPS)
		vkQueueWaitIdle(self.device.presentation_queue)
		