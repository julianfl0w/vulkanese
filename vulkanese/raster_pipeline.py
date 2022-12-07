import ctypes
import os
import time
import json
from vulkan import *

from . import vulkanese
from . import pipeline
from . import buffer
from . import shader
    
from sinode import *
from PIL import Image as pilImage

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


class RasterPipeline(pipeline.Pipeline):
    def __init__(
        self,
        device,
        constantsDict,
        outputClass="surface",
        outputWidthPixels=700,
        outputHeightPixels=700,
    ):
        self.outputClass = outputClass
        self.DEBUG = False
        self.device = device
        self.constantsDict = constantsDict

        self.outputWidthPixels = outputWidthPixels
        self.outputHeightPixels = outputHeightPixels

    def createVertexBuffers(self):

        # if we're debugging, all intermediate variables become output buffers
        allBufferDescriptions = (
            self.vertexShaderInputBuffers
            + self.vertexShaderInputBuffersNoDebug
            + self.vertexShaderOutputBuffers
        )

        # if debugging, we make the internal variables into CPU-readable buffers
        if self.DEBUG:
            allBufferDescriptions += self.vertexShaderDebuggableVars

        self.vertexBuffers = []
        location = 0
        for bd in allBufferDescriptions:
            if bd in self.vertexShaderOutputBuffers:
                qualifier = "out"
            else:
                qualifier = "in"
            newBuff = Buffer(
                device=self.device,
                dimensionNames=bd["dims"],
                dimensionVals=[self.constantsDict[d] for d in bd["dims"]],
                name=bd["name"],
                descriptorSet=self.device.descriptorPool.descSetGlobal,
                location=location,
                qualifier=qualifier,
            )
            location += newBuff.getSize()
            self.vertexBuffers += [newBuff]

        # in the vertex stage, we also need to create an index buffer

        # add index buffer at end
        # (was in middle before)
        # actually this should probably belong to the pipeline
        self.indexBuffer = Buffer(
            name="index",
            readFromCPU=True,
            dimensionNames=["TRIANGLE_COUNT", "VERTS_PER_TRIANGLE"],
            dimensionVals=[
                self.constantsDict["TRIANGLE_COUNT"],
                self.constantsDict["VERTS_PER_TRIANGLE"],
            ],
            location=location,
            descriptorSet=self.device.descriptorPool.descSetGlobal,
            device=self.device,
            memtype="uint",
            usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
            | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            format=VK_FORMAT_R32_UINT,
            stride=4,
        )

    def createFragmentBuffers(self):

        # if we're debugging, all intermediate variables become output buffers
        allBufferDescriptions = (
            self.fragmentShaderInputBuffers + self.fragmentShaderOutputBuffers
        )

        # if debugging, we make the internal variables into CPU-readable buffers
        if self.DEBUG:
            allBufferDescriptions += self.fragmentShaderDebuggableVars

        self.fragmentBuffers = []
        location = 0
        for bd in allBufferDescriptions:
            if bd in self.fragmentShaderOutputBuffers:
                qualifier = "out"
            else:
                qualifier = "in"

            newBuff = Buffer(
                device=self.device,
                dimensionNames=bd["dims"],
                dimensionVals=[self.constantsDict[d] for d in bd["dims"]],
                name=bd["name"],
                descriptorSet=self.device.descriptorPool.descSetGlobal,
                location=location,
                qualifier=qualifier,
                memtype=bd["type"],
            )
            location += newBuff.getSize()
            self.fragmentBuffers += [newBuff]

    def createStandardBuffers(self):

        # Input buffers to the shader
        # These are Uniform Buffers normally,
        # Storage Buffers in DEBUG Mode
        # PIPELINE WILL CREATE ITS OWN INDEX BUFFER
        self.vertexShaderInputBuffers = [
            {
                "name": "position",
                "type": "vec3",
                "dims": ["VERTEX_COUNT", "SPATIAL_DIMENSIONS"],
            },
            {
                "name": "normal",
                "type": "vec3",
                "dims": ["VERTEX_COUNT", "SPATIAL_DIMENSIONS"],
            },
            {
                "name": "color",
                "type": "vec3",
                "dims": ["VERTEX_COUNT", "COLOR_DIMENSIONS"],
            },
        ]

        # any input buffers you want to exclude from debug
        # for example, a sine lookup table
        self.vertexShaderInputBuffersNoDebug = []

        # variables that are usually intermediate variables in the shader
        # but in DEBUG mode they are made visible to the CPU (as Storage Buffers)
        # so that you can view shader intermediate values :)
        self.vertexShaderDebuggableVars = []

        # the output of the compute shader,
        # which in our case is always a Storage Buffer
        self.vertexShaderOutputBuffers = [
            {
                "name": "fragColor",
                "type": "vec3",
                "dims": ["VERTEX_COUNT", "COLOR_DIMENSIONS"],
            }
        ]

        # location = 0
        # outColor = Buffer(
        #    device=device,
        #    name="outColor",
        #    qualifier="out",
        #    binding=5,
        #    type="vec4",
        #    descriptorSet=device.descriptorPool.descSetGlobal,
        #    usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        #    sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        #    SIZEBYTES=65536,
        #    format=VK_FORMAT_R32G32B32_SFLOAT,
        #    memProperties=VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        #    | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        #    location=location,
        # )

        self.fragmentShaderInputBuffers = []
        self.fragmentShaderDebuggableVars = []
        self.fragmentShaderOutputBuffers = [
            {"name": "outColor", "type": "vec4", "dims": ["VERTEX_COUNT"]}
        ]

        self.createVertexBuffers()
        self.createFragmentBuffers()

    def createStages(self):

        # Stage
        self.vertexStage = VertexStage(
            device=self.device,
            parent=self,
            constantsDict=self.constantsDict,
            buffers=self.vertexBuffers,
            name="passthrough.vert",
        )
        #######################################################
        # fragment stage
        # buffers,
        # location += fragColor.getSize()
        # outColor.location = 0
        # fragColor.location= 1
        # fragColor.qualifier = "in"
        # Stage
        # idk, pipe one to the next
        sharedBuffer = self.vertexStage.getBufferByName("fragColor")
        self.fragmentBuffers += [sharedBuffer]
        self.fragmentStage = FragmentStage(
            device=self.device,
            parent=self,
            buffers=self.fragmentBuffers,
            constantsDict=self.constantsDict,
            name="passthrough.frag",
        )

        self.stages = [self.vertexStage, self.fragmentStage]

    def createGraphicPipeline(
        self, culling=VK_CULL_MODE_BACK_BIT, oversample=VK_SAMPLE_COUNT_1_BIT
    ):

        # Create semaphores
        self.semaphore_image_available = Semaphore(device=self.device)
        self.semaphore_render_finished = Semaphore(device=self.device)
        self.semaphores = [
            self.semaphore_image_available,
            self.semaphore_render_finished,
        ]

        self.wait_stages = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        self.wait_semaphores = [self.semaphore_image_available]
        self.signal_semaphores = [self.semaphore_render_finished]

        # Create a surface, if indicated
        if outputClass == "surface":
            newSurface = Surface(self.device.instance, self.device, self)
            self.surface = newSurface
            self.children += [self.surface]

        self.vkAcquireNextImageKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkAcquireNextImageKHR"
        )
        self.vkQueuePresentKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkQueuePresentKHR"
        )

        
        # compile all the stages
        [s.compile() for s in self.stages]

        Pipeline.__init__(
            self,
            device=self.device,
            stages=self.stages,
            indexBuffer=self.indexBuffer,
            outputClass=self.outputClass,
            outputWidthPixels=self.outputWidthPixels,
            outputHeightPixels=self.outputHeightPixels,
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
