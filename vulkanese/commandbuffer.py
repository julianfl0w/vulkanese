import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *

class CommandBuffer(Sinode):
	def __init__(self, pipeline):
		Sinode.__init__(self, pipeline)
		self.pipeline = pipeline
		self.pipelineDict  = pipeline.setupDict
		self.vkCommandPool = pipeline.device.vkCommandPool
		self.device        = pipeline.device
		self.vkDevice      = pipeline.device.vkDevice
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

		
class RasterCommandBuffer(CommandBuffer):
	def __init__(self, pipeline):
		CommandBuffer.__init__(self, pipeline)
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
			
			print("--- ALL BUFFERS ---")
			for i, buffer in enumerate(allBuffers):
				print("-------------------------")
				print(i)
				print(json.dumps(buffer.setupDict, indent=4))
			allVertexBuffers = [b for b in allBuffers if (\
			eval(b.setupDict["usage"]) & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT \
			 or b.setupDict["name"] == "index")]
			print("--- ALL VERTEX BUFFERS ---")
			for i, buffer in enumerate(allVertexBuffers):
				print("-------------------------")
				print(i)
				print(json.dumps(buffer.setupDict, indent=4))
				
			allVertexBuffersVk = [b.vkBuffer for b in allVertexBuffers]
			
			pOffsets = [0]*len(allVertexBuffersVk)
			print("pOffsets")
			print(pOffsets)
			vkCmdBindVertexBuffers(
				commandBuffer       = vkCommandBuffer,
				firstBinding        = 0,
				bindingCount        = len(allVertexBuffersVk),
				pBuffers            = allVertexBuffersVk,
				pOffsets            = pOffsets
				)
			
			vkCmdBindIndexBuffer(
				commandBuffer  = vkCommandBuffer,
				buffer         = pipeline.getBuffDict()["index"].vkBuffer, 
				offset         = 0 ,  
				indexType      = VK_INDEX_TYPE_UINT32 )
				
			# Draw
			#void vkCmdDraw(
			#	VkCommandBuffer commandBuffer,
			#	uint32_t        vertexCount,
			#	uint32_t        instanceCount,
			#	uint32_t        firstVertex,
			#	uint32_t        firstInstance);
			#vkCmdDraw(vkCommandBuffer, 6400, 1, 0, 1)
			
			#void vkCmdDrawIndexed(
			#	VkCommandBuffer                             commandBuffer,
			#	uint32_t                                    indexCount,
			#	uint32_t                                    instanceCount,
			#	uint32_t                                    firstIndex,
			#	int32_t                                     vertexOffset,
			#	uint32_t                                    firstInstance);
			vkCmdDrawIndexed(vkCommandBuffer, 6400, 1, 0, 0, 0)
			
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
		
class RaytraceCommandBuffer(CommandBuffer):
	def __init__(self, pipeline):
		CommandBuffer.__init__(self, pipeline)
		
		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, self.rtPipeline);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, self.rtPipelineLayout, 0, size(descSets), descSets, 0, nullptr);

		vkCmdTraceRaysKHR(
			cmdBuf, 
			self.shaderDict["rgen"].vkStridedDeviceAddressRegion, 
			self.shaderDict["miss"].vkStridedDeviceAddressRegion,
			self.shaderDict["hit"].vkStridedDeviceAddressRegion,
			self.shaderDict["call"], 
			self.outputWidthPixels, 
			self.outputHeightPixels, 
			1);

 
		vkEndCommandBuffer(vkCommandBuffer)
		self.debug.endLabel(cmdBuf);
		
	#def drawPost(VkCommandBuffer cmdBuf):
	#	m_debug.beginLabel(cmdBuf, "Post");
	#
	#	setViewport(cmdBuf);
	#
	#	auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
	#	vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &aspectRatio);
	#	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
	#	vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
	#	vkCmdDraw(cmdBuf, 3, 1, 0, 0);