import json
from sinode import *
import os
import vulkan as vk


class RenderPass(Sinode):
    def __init__(self, device, pipeline, oversample, surface):
        Sinode.__init__(self, pipeline)
        self.pipeline = pipeline
        self.surface = surface
        self.vkDevice = device.vkDevice

        # Create render pass
        color_attachement = vk.VkAttachmentDescription(
            flags=0,
            format=self.surface.surface_format.format,
            samples=oversample,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            # initialLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )

        color_attachement_reference = vk.VkAttachmentReference(
            attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        sub_pass = vk.VkSubpassDescription(
            flags=0,
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            inputAttachmentCount=0,
            pInputAttachments=None,
            pResolveAttachments=None,
            pDepthStencilAttachment=None,
            preserveAttachmentCount=0,
            pPreserveAttachments=None,
            colorAttachmentCount=1,
            pColorAttachments=[color_attachement_reference],
        )

        dependency = vk.VkSubpassDependency(
            dependencyFlags=0,
            srcSubpass=vk.VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
            | vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        )

        render_pass_create = vk.VkRenderPassCreateInfo(
            flags=0,
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=1,
            pAttachments=[color_attachement],
            subpassCount=1,
            pSubpasses=[sub_pass],
            dependencyCount=1,
            pDependencies=[dependency],
        )

        self.vkRenderPass = vk.vkCreateRenderPass(
            device.vkDevice, render_pass_create, None
        )
        self.children += [self.vkRenderPass]

        # print("%s images view created" % len(self.image_views))

    def release(self):

        print("destroying framebuffers")
        for i, f in enumerate(self.framebuffers):
            print("destroying framebuffer " + str(i))
            vkDestroyFramebuffer(self.vkDevice, f, None)

        for i in self.image_views:
            vkDestroyImageView(self.vkDevice, i, None)
        vkDestroyRenderPass(self.vkDevice, self.vkRenderPass, None)
