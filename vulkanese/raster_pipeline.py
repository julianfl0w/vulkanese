import ctypes
import os
import time
import json
import vulkan as vk

from . import vulkanese
from . import pipeline
from . import buffer
from . import shader
from . import synchronization
from . import renderpass
from . import surface
from . import command_buffer

from sinode import *


class RasterPipeline(pipeline.Pipeline):
    def __init__(
        self,
        device,
        constantsDict,
        indexBuffer,
        stages,
        buffers,
        outputClass="surface",
        outputWidthPixels=700,
        outputHeightPixels=700,
        culling=vk.VK_CULL_MODE_BACK_BIT,
        oversample=vk.VK_SAMPLE_COUNT_1_BIT,
    ):
        self.indexBuffer = indexBuffer
        self.outputClass = outputClass
        self.DEBUG = False
        self.constantsDict = constantsDict
        self.stages = stages
        self.outputWidthPixels = outputWidthPixels
        self.outputHeightPixels = outputHeightPixels
        for stage in stages:
            # make the buffer accessable as a local attribute
            exec("self." + stage.name + "= stage")
        pipeline.Pipeline.__init__(
            self,
            device=device,
            stages=stages,
            outputClass=self.outputClass,
            outputWidthPixels=self.outputWidthPixels,
            outputHeightPixels=self.outputHeightPixels,
        )

        push_constant_ranges = vk.VkPushConstantRange(stageFlags=0, offset=0, size=0)

        self.pipelineCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            flags=0,
            setLayoutCount=0,
            pSetLayouts=None,
            pushConstantRangeCount=0,
            pPushConstantRanges=[push_constant_ranges],
        )

        self.vkPipelineLayout = vk.vkCreatePipelineLayout(
            self.vkDevice, self.pipelineCreateInfo, None
        )
        self.children += [self.vkPipelineLayout]


        # get global lists
        allVertexBuffers = []
        for b in set(buffers):
            if b.usage==vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT:
                allVertexBuffers += [b]

        allBindingDescriptors = [b.bindingDescription for b in allVertexBuffers]
        allAttributeDescriptors = [b.attributeDescription for b in allVertexBuffers]
        print("allAttributeDescriptors " + str(allAttributeDescriptors))

        # Create graphic Pipeline
        vertex_input_create = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            flags=0,
            vertexBindingDescriptionCount=len(allBindingDescriptors),
            pVertexBindingDescriptions=allBindingDescriptors,
            vertexAttributeDescriptionCount=len(allAttributeDescriptors),
            pVertexAttributeDescriptions=allAttributeDescriptors,
        )

        input_assembly_create = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            flags=0,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=vk.VK_FALSE,
        )
        viewport = vk.VkViewport(
            x=0.0,
            y=0.0,
            width=float(self.outputWidthPixels),
            height=float(self.outputHeightPixels),
            minDepth=0.0,
            maxDepth=1.0,
        )

        scissor_offset = vk.VkOffset2D(x=0, y=0)
        self.extent = vk.VkExtent2D(
            width=self.outputWidthPixels, height=self.outputHeightPixels
        )
        scissor = vk.VkRect2D(offset=scissor_offset, extent=self.extent)
        viewport_state_create = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            flags=0,
            viewportCount=1,
            pViewports=[viewport],
            scissorCount=1,
            pScissors=[scissor],
        )

        rasterizer_create = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            flags=0,
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1,
            cullMode=culling,
            frontFace=vk.VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=vk.VK_FALSE,
            depthBiasConstantFactor=0.0,
            depthBiasClamp=0.0,
            depthBiasSlopeFactor=0.0,
        )

        multisample_create = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            flags=0,
            sampleShadingEnable=vk.VK_FALSE,
            rasterizationSamples=oversample,
            minSampleShading=1,
            pSampleMask=None,
            alphaToCoverageEnable=vk.VK_FALSE,
            alphaToOneEnable=vk.VK_FALSE,
        )

        color_blend_attachement = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
            | vk.VK_COLOR_COMPONENT_G_BIT
            | vk.VK_COLOR_COMPONENT_B_BIT
            | vk.VK_COLOR_COMPONENT_A_BIT,
            blendEnable=vk.VK_FALSE,
            srcColorBlendFactor=vk.VK_BLEND_FACTOR_ONE,
            dstColorBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
            colorBlendOp=vk.VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
            alphaBlendOp=vk.VK_BLEND_OP_ADD,
        )

        color_blend_create = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            flags=0,
            logicOpEnable=vk.VK_FALSE,
            logicOp=vk.VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=[color_blend_attachement],
            blendConstants=[0, 0, 0, 0],
        )

        # Create a surface, if indicated
        if outputClass == "surface":
            newSurface = surface.Surface(self.device.instance, self.device, self)
            self.surface = newSurface
            self.children += [self.surface]

        self.vkAcquireNextImageKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkAcquireNextImageKHR"
        )
        self.vkQueuePresentKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkQueuePresentKHR"
        )
        
        # Create a generic render pass
        self.renderPass = renderpass.RenderPass(self, oversample=oversample, surface=self.surface)
        self.children += [self.renderPass]
        
        # Finally create graphicsPipeline
        self.pipelinecreate = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            flags=0,
            stageCount=len(self.stages),
            pStages=[s.shader_stage_create for s in self.stages],
            pVertexInputState=vertex_input_create,
            pInputAssemblyState=input_assembly_create,
            pTessellationState=None,
            pViewportState=viewport_state_create,
            pRasterizationState=rasterizer_create,
            pMultisampleState=multisample_create,
            pDepthStencilState=None,
            pColorBlendState=color_blend_create,
            pDynamicState=None,
            layout=self.vkPipelineLayout,
            renderPass=self.renderPass.vkRenderPass,
            subpass=0,
            basePipelineHandle=None,
            basePipelineIndex=-1,
        )

        pipelines = vk.vkCreateGraphicsPipelines(
            self.vkDevice, None, 1, [self.pipelinecreate], None
        )
        self.children += [pipelines]

        self.vkPipeline = pipelines[0]

        # wrap it all up into a command buffer
        self.commandBuffer = command_buffer.RasterCommandBuffer(self)

    def draw_frame(self):
        image_index = self.vkAcquireNextImageKHR(
            self.vkDevice,
            self.surface.swapchain,
            vk.UINT64_MAX,
            self.signalSemaphores[0].vkSemaphore,
            None,
        )
        self.commandBuffer.draw_frame(image_index)
