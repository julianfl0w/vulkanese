import ctypes
import os
import time
import json
from vulkan import *
from pipelines import *
from surface import *
from stage import *
from renderpass import *
from commandbuffer import *
from vutil import *
from vulkanese import *
from PIL import Image as pilImage

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


class RasterPipeline(Pipeline):
    def __init__(
        self,
        device,
        indexBuffer,
        stages,
        culling=VK_CULL_MODE_BACK_BIT,
        oversample=VK_SAMPLE_COUNT_1_BIT,
        outputClass="surface",
        outputWidthPixels=700,
        outputHeightPixels=700,
    ):

        Pipeline.__init__(
            self,
            device=device,
            stages=stages,
            indexBuffer=indexBuffer,
            outputClass=outputClass,
            outputWidthPixels=outputWidthPixels,
            outputHeightPixels=outputHeightPixels,
        )

        push_constant_ranges = VkPushConstantRange(stageFlags=0, offset=0, size=0)

        self.pipelineCreateInfo = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            flags=0,
            setLayoutCount=0,
            pSetLayouts=None,
            pushConstantRangeCount=0,
            pPushConstantRanges=[push_constant_ranges],
        )

        self.vkPipelineLayout = vkCreatePipelineLayout(
            self.vkDevice, self.pipelineCreateInfo, None
        )
        self.children += [self.vkPipelineLayout]

        # Create a generic render pass
        self.renderPass = RenderPass(self, oversample=oversample, surface=self.surface)
        self.children += [self.renderPass]

        # get global lists
        allVertexBuffers = []
        for s in self.stages:
            allVertexBuffers += s.getVertexBuffers()

        allBindingDescriptors = [b.bindingDescription for b in allVertexBuffers]
        allAttributeDescriptors = [b.attributeDescription for b in allVertexBuffers]
        print("allAttributeDescriptors " + str(allAttributeDescriptors))

        # Create graphic Pipeline
        vertex_input_create = VkPipelineVertexInputStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            flags=0,
            vertexBindingDescriptionCount=len(allBindingDescriptors),
            pVertexBindingDescriptions=allBindingDescriptors,
            vertexAttributeDescriptionCount=len(allAttributeDescriptors),
            pVertexAttributeDescriptions=allAttributeDescriptors,
        )

        input_assembly_create = VkPipelineInputAssemblyStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            flags=0,
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=VK_FALSE,
        )
        viewport = VkViewport(
            x=0.0,
            y=0.0,
            width=float(self.outputWidthPixels),
            height=float(self.outputHeightPixels),
            minDepth=0.0,
            maxDepth=1.0,
        )

        scissor_offset = VkOffset2D(x=0, y=0)
        self.extent = VkExtent2D(
            width=self.outputWidthPixels, height=self.outputHeightPixels
        )
        scissor = VkRect2D(offset=scissor_offset, extent=self.extent)
        viewport_state_create = VkPipelineViewportStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            flags=0,
            viewportCount=1,
            pViewports=[viewport],
            scissorCount=1,
            pScissors=[scissor],
        )

        rasterizer_create = VkPipelineRasterizationStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            flags=0,
            depthClampEnable=VK_FALSE,
            rasterizerDiscardEnable=VK_FALSE,
            polygonMode=VK_POLYGON_MODE_FILL,
            lineWidth=1,
            cullMode=culling,
            frontFace=VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=VK_FALSE,
            depthBiasConstantFactor=0.0,
            depthBiasClamp=0.0,
            depthBiasSlopeFactor=0.0,
        )

        multisample_create = VkPipelineMultisampleStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            flags=0,
            sampleShadingEnable=VK_FALSE,
            rasterizationSamples=oversample,
            minSampleShading=1,
            pSampleMask=None,
            alphaToCoverageEnable=VK_FALSE,
            alphaToOneEnable=VK_FALSE,
        )

        color_blend_attachement = VkPipelineColorBlendAttachmentState(
            colorWriteMask=VK_COLOR_COMPONENT_R_BIT
            | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT
            | VK_COLOR_COMPONENT_A_BIT,
            blendEnable=VK_FALSE,
            srcColorBlendFactor=VK_BLEND_FACTOR_ONE,
            dstColorBlendFactor=VK_BLEND_FACTOR_ZERO,
            colorBlendOp=VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO,
            alphaBlendOp=VK_BLEND_OP_ADD,
        )

        color_blend_create = VkPipelineColorBlendStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            flags=0,
            logicOpEnable=VK_FALSE,
            logicOp=VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=[color_blend_attachement],
            blendConstants=[0, 0, 0, 0],
        )

        # Finally create graphicsPipeline
        self.pipelinecreate = VkGraphicsPipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
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

        pipelines = vkCreateGraphicsPipelines(
            self.vkDevice, None, 1, [self.pipelinecreate], None
        )
        self.children += [pipelines]

        self.vkPipeline = pipelines[0]

        # wrap it all up into a command buffer
        self.commandBuffer = RasterCommandBuffer(self)
