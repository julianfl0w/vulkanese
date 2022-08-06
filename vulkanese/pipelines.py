import ctypes
import os
import time
import json
from vulkan import *
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


# all pipelines contain:
# references to instance, device, etc
# at least 1 stage
# an output size
class Pipeline(Sinode):
    def __init__(self, device, setupDict):

        Sinode.__init__(self, device)
        self.location = 0

        self.setupDict = setupDict
        self.vkDevice = device.vkDevice
        self.device = device
        self.instance = device.instance

        # Create semaphores
        semaphore_create = VkSemaphoreCreateInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, flags=0
        )
        self.semaphore_image_available = vkCreateSemaphore(
            self.vkDevice, semaphore_create, None
        )
        self.semaphore_render_finished = vkCreateSemaphore(
            self.vkDevice, semaphore_create, None
        )

        self.wait_stages = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        self.wait_semaphores = [self.semaphore_image_available]
        self.signal_semaphores = [self.semaphore_render_finished]

        # Create a surface, if indicated
        if setupDict["outputClass"] == "surface":
            newSurface = Surface(self.device.instance, self.device, self)
            self.surface = newSurface
            self.children += [self.surface]

        self.pipelineClass = setupDict["class"]
        self.outputWidthPixels = setupDict["outputWidthPixels"]
        self.outputHeightPixels = setupDict["outputHeightPixels"]

        self.vkAcquireNextImageKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkAcquireNextImageKHR"
        )
        self.vkQueuePresentKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkQueuePresentKHR"
        )

        # Add Stages
        self.stages = []
        for key, value in setupDict["stage"].items():
            if key == "name":
                continue
            self.stages += Stage(self, value)

        self.children += self.stages

        push_constant_ranges = VkPushConstantRange(stageFlags=0, offset=0, size=0)

        self.pipelineCreateInfo = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            flags=0,
            setLayoutCount=0,
            pSetLayouts=None,
            pushConstantRangeCount=0,
            pPushConstantRanges=[push_constant_ranges],
        )

        self.pipelineLayout = vkCreatePipelineLayout(
            self.vkDevice, self.pipelineCreateInfo, None
        )
        self.children += [self.pipelineLayout]

    def draw_frame(self):
        image_index = self.vkAcquireNextImageKHR(
            self.vkDevice,
            self.surface.swapchain,
            UINT64_MAX,
            self.semaphore_image_available,
            None,
        )
        self.commandBuffer.draw_frame(image_index)

    def getAllBuffers(self):
        allBuffers = []
        for stage in self.stages:
            allBuffers += stage.buffers

        return allBuffers

    def release(self):
        print("generic pipeline release")
        vkDestroySemaphore(self.vkDevice, self.semaphore_image_available, None)
        vkDestroySemaphore(self.vkDevice, self.semaphore_render_finished, None)

        for stage in self.stages:
            stage.release()

        vkDestroyPipeline(self.vkDevice, self.vkPipeline, None)
        vkDestroyPipelineLayout(self.vkDevice, self.pipelineLayout, None)

        print("releasing surface")
        if self.surface is not None:
            print("releasing surface")
            self.surface.release()

        if self.renderPass is not None:
            self.renderPass.release()

        self.commandBuffer.release()


# the compute pipeline is so much simpler than the old-school
# graphics pipeline. it should be considered separately
class ComputePipeline(Pipeline):
    def __init__(self, device, setupDict):
        Pipeline.__init__(self, device, setupDict)

        self.descriptorSet = DescriptorSet(device.descriptorPool)

        # The pipeline layout allows the pipeline to access descriptor sets.
        # So we just specify the descriptor set layout we created earlier.
        pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.__descriptorSetLayout],
        )
        self.pipelineLayout = vkCreatePipelineLayout(
            self.vkDevice, pipelineLayoutCreateInfo, None
        )

        self.pipelineCreateInfo = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=shaderStageCreateInfo,
            layout=self.pipelineLayout,
        )

        # Now, we finally create the compute pipeline.
        pipelines = vkCreateComputePipelines(
            self.vkDevice, VK_NULL_HANDLE, 1, self.pipelineCreateInfo, None
        )
        if len(pipelines) == 1:
            self.__pipeline = pipelines[0]


class RasterPipeline(Pipeline):
    def __init__(self, device, setupDict):
        Pipeline.__init__(self, device, setupDict)

        # Create a generic render pass
        self.renderPass = RenderPass(self, setupDict, self.surface)
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
            cullMode=eval(setupDict["culling"]),
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
            rasterizationSamples=eval(setupDict["oversample"]),
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
            layout=self.pipelineLayout,
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

