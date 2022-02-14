import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *

class RenderPass(PrintClass):
	def __init__(self, pipeline, setupDict, surface):
		PrintClass.__init__(self)
		self.pipeline = pipeline
		self.surface  = surface
		self.vkDevice = self.surface.device.vkDevice
		self.instance = self.surface.device.instance
		
		# Create render pass
		color_attachement = VkAttachmentDescription(
			flags=0,
			format=self.surface.surface_format.format,
			samples=VK_SAMPLE_COUNT_1_BIT,
			loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
			storeOp=VK_ATTACHMENT_STORE_OP_STORE,
			stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
			initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
			#initialLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)

		color_attachement_reference = VkAttachmentReference(
			attachment=0,
			layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)

		sub_pass = VkSubpassDescription(
			flags=0,
			pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
			inputAttachmentCount=0,
			pInputAttachments=None,
			pResolveAttachments=None,
			pDepthStencilAttachment=None,
			preserveAttachmentCount=0,
			pPreserveAttachments=None,
			colorAttachmentCount=1,
			pColorAttachments=[color_attachement_reference])

		dependency = VkSubpassDependency(
			dependencyFlags=0,
			srcSubpass=VK_SUBPASS_EXTERNAL,
			dstSubpass=0,
			srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			srcAccessMask=0,
			dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)

		render_pass_create = VkRenderPassCreateInfo(
			flags=0,
			sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			attachmentCount=1,
			pAttachments=[color_attachement],
			subpassCount=1,
			pSubpasses=[sub_pass],
			dependencyCount=1,
			pDependencies=[dependency])

		self.vkRenderPass = vkCreateRenderPass(self.vkDevice, render_pass_create, None)
		self.children += [self.vkRenderPass]
		
		# Create image view for each image in swapchain
		self.image_views = []
		self.framebuffers = []
		self.render_pass_begin_create = []
		
		for i, image in enumerate(self.surface.vkSwapchainImages):
			subresourceRange = VkImageSubresourceRange(
				aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
				baseMipLevel=0,
				levelCount=1,
				baseArrayLayer=0,
				layerCount=1)

			components = VkComponentMapping(
				r=VK_COMPONENT_SWIZZLE_IDENTITY,
				g=VK_COMPONENT_SWIZZLE_IDENTITY,
				b=VK_COMPONENT_SWIZZLE_IDENTITY,
				a=VK_COMPONENT_SWIZZLE_IDENTITY)

			imageview_create = VkImageViewCreateInfo(
				sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				image=image,
				flags=0,
				viewType=VK_IMAGE_VIEW_TYPE_2D,
				format=self.surface.surface_format.format,
				components=components,
				subresourceRange=subresourceRange)

			newImageView = vkCreateImageView(self.vkDevice, imageview_create, None)
			self.image_views += [newImageView]
			
			# Create Graphics render pass
			render_area = VkRect2D(offset=VkOffset2D(x=0, y=0),
								   extent=self.surface.extent)
			color = VkClearColorValue(float32=[0, 1, 0, 1])
			clear_value = VkClearValue(color=color)

			# Framebuffers creation
			attachments = [newImageView]
			framebuffer_create = VkFramebufferCreateInfo(
				sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				flags=0,
				renderPass=self.vkRenderPass,
				attachmentCount=len(attachments),
				pAttachments=attachments,
				width=self.surface.WIDTH,
				height=self.surface.HEIGHT,
				layers=1)
				
			thisFramebuffer = vkCreateFramebuffer(self.vkDevice, framebuffer_create, None)
			self.framebuffers += [thisFramebuffer]
			
			thisRenderPass = VkRenderPassBeginInfo(
				sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				renderPass=self.vkRenderPass,
				framebuffer=thisFramebuffer,
				renderArea=render_area,
				clearValueCount=1,
				pClearValues=[clear_value])
				
			self.render_pass_begin_create += [thisRenderPass]
			
		self.children += self.framebuffers
		self.children += self.image_views
		
		print("%s images view created" % len(self.image_views))
	def release(self):
	
		print("destroying framebuffers")
		for i, f in enumerate(self.framebuffers):
			print("destroying framebuffer " + str(i))
			vkDestroyFramebuffer(self.vkDevice, f, None)
			
		for i in self.image_views:
			vkDestroyImageView(self.vkDevice, i, None)
		vkDestroyRenderPass(self.vkDevice, self.vkRenderPass, None)
		
		