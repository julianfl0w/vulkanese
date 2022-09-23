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
import re

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


# the compute pipeline is so much simpler than the old-school
# graphics pipeline. it should be considered separately
class ComputePipeline(Pipeline):
    def __init__(self, device, stages, workgroupShape=[1, 1, 1]):
        Pipeline.__init__(self, device, stages=stages, outputClass="image")

        self.descriptorSet = device.descriptorPool.descSetGlobal

        # The pipeline layout allows the pipeline to access descriptor sets.
        # So we just specify the descriptor set layout we created earlier.
        pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=len(device.descriptorPool.descSets),
            pSetLayouts=[
                d.vkDescriptorSetLayout for d in device.descriptorPool.descSets
            ],
        )

        self.vkPipelineLayout = vkCreatePipelineLayout(
            self.vkDevice, pipelineLayoutCreateInfo, None
        )

        # Now let us actually create the compute pipeline.
        # A compute pipeline is very simple compared to a graphics pipeline.
        # It only consists of a single stage with a compute shader.
        # So first we specify the compute shader stage, and it's entry point(main).
        shaderStageCreateInfo = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=stages[0].vkShaderModule,
            pName="main",
        )

        self.pipelineCreateInfo = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=shaderStageCreateInfo,
            layout=self.vkPipelineLayout,
        )

        # Now, we finally create the compute pipeline.
        pipelines = vkCreateComputePipelines(
            self.vkDevice, VK_NULL_HANDLE, 1, self.pipelineCreateInfo, None
        )
        if len(pipelines) == 1:
            self.vkPipeline = pipelines[0]

        self.children += [pipelines]
        # wrap it all up into a command buffer
        self.commandBuffer = ComputeCommandBuffer(self, workgroupShape=workgroupShape)


# THIS CONTAINS EVERYTHING YOU NEED!
# kind of a misnomer, because it includes 
# command buffer
# pipeline,
# shader
# All in one
class ComputeShader:
    def __init__(
        self,
        main,
        header,
        device,
        shaderOutputBuffers,
        debuggableVars,
        shaderInputBuffers,
        shaderInputBuffersNoDebug,
        DEBUG,
        workgroupShape=[1, 1, 1],
    ):

        bindingUniform = 0
        bindingStorage = 0
        allBuffers = []
        self.DEBUG = DEBUG
        self.device = device

        # if we're debugging, all intermediate variables become output buffers
        if self.DEBUG:
            allBufferDescriptions = (
                shaderOutputBuffers
                + debuggableVars
                + shaderInputBuffers
                + shaderInputBuffersNoDebug
            )
        else:
            allBufferDescriptions = (
                shaderOutputBuffers + shaderInputBuffers + shaderInputBuffersNoDebug
            )

        # create all buffers according to their description
        for s in allBufferDescriptions:
            format = VK_FORMAT_R32_SFLOAT

            if s in shaderOutputBuffers or s in debuggableVars:
                descriptorSet = device.descriptorPool.descSetGlobal
                usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                binding = bindingStorage
                bindingStorage += 1
            else:
                descriptorSet = device.descriptorPool.descSetUniform
                usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                binding = bindingUniform
                bindingUniform += 1

            newBuff = Buffer(
                binding=binding,
                device=device,
                type=s["type"],
                descriptorSet=descriptorSet,
                qualifier="out",
                name=s["name"],
                readFromCPU=True,
                SIZEBYTES=s["SIZEBYTES"],
                usage=usage,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                location=0,
                format=format,
            )
            allBuffers += [newBuff]
            exec("self." + s["name"] + " = newBuff")

        VARSDECL = ""
        if self.DEBUG:
            main = self.addIndicesToOutputs(debuggableVars, main)
        else:
            # otherwise, just declare the variable type
            # INITIALIZE TO 0 !
            for var in debuggableVars:
                VARSDECL += var["type"] + " " + var["name"] + " = 0;\n"

        # always index output variables
        main = self.addIndicesToOutputs(shaderOutputBuffers, main)

        main = main.replace("VARIABLEDECLARATIONS", VARSDECL)

        # Stage
        existingBuffers = []
        mandleStage = Stage(
            device=device,
            name="mandlebrot.comp",
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            existingBuffers=existingBuffers,
            outputWidthPixels=700,
            outputHeightPixels=700,
            header=header,
            main=main,
            buffers=allBuffers,
        )

        #######################################################
        # Pipeline
        device.descriptorPool.finalize()

        self.computePipeline = ComputePipeline(
            device=device,
            # workgroupShape=[self.SAMPLES_PER_DISPATCH, self.SHADERS_PER_SAMPLE, 1],
            workgroupShape=workgroupShape,
            stages=[mandleStage],
        )
        device.children += [self.computePipeline]

        # Now we shall finally submit the recorded command buffer to a queue.
        self.submitInfo = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,  # submit a single command buffer
            pCommandBuffers=[
                self.computePipeline.commandBuffer.vkCommandBuffers[0]
            ],  # the command buffer to submit.
        )

        # We create a fence.
        fenceCreateInfo = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=0
        )
        self.fence = vkCreateFence(self.device.vkDevice, fenceCreateInfo, None)

    # take the template GLSL file
    # and add output indices
    # mostly for debugging
    def addIndicesToOutputs(self, outvars, shaderGLSL):
        # given dimension A, return index
        self.dim2index = {
            "SHADERS_PER_SAMPLE": "shaderIndexInSample",
            "SAMPLES_PER_DISPATCH": "sampleNo",
            "POLYPHONY": "noteNo",
            "PARTIALS_PER_VOICE": "partialNo",
        }
        indexedVarStrings = []
        # if this is debug mode, all vars have already been declared as output buffers
        # intermediate variables must be indexed
        for var in outvars:
            indexedVarString = var["name"] + "["
            for i, d in enumerate(var["dims"]):
                indexedVarString += self.dim2index[d]
                # since GLSL doesnt allow multidim arrays (kinda, see gl_arb_arrays_of_arrays)
                # ok i dont understand Vulkan multidim
                for j, rd in enumerate(var["dims"][i + 1 :]):
                    indexedVarString += "*" + rd
                if i < (len(var["dims"]) - 1):
                    indexedVarString += "+"
            indexedVarString += "]"
            indexedVarStrings += [(var["name"], indexedVarString)]

        outShaderGLSL = ""
        for line in shaderGLSL.split("\n"):
            # replace all non-comments
            if not line.strip().startswith("//"):
                # whole-word replacement
                for iv in indexedVarStrings:
                    line = re.sub(r"\b{}\b".format(iv[0]), iv[1], line)
            # keep the comments
            outShaderGLSL += line + "\n"

        return outShaderGLSL

    # might help when you run the main loop in C/C++
    def getVulkanAddresses(self):
        addrDict = {}
        addrDict["FENCEADDR"] = hex(eval(str(self.fence).split(" ")[-1][:-1]))
        addrDict["DEVADDR"] = str(self.device.vkDevice).split(" ")[-1][:-1]
        addrDict["SUBMITINFOADDR"] = str(ffi.addressof(self.submitInfo)).split(" ")[-1][
            :-1
        ]
        return addrDict

    def run(self):

        # We submit the command buffer on the queue, at the same time giving a fence.
        vkQueueSubmit(
            queue=self.device.compute_queue,
            submitCount=1,
            pSubmits=self.submitInfo,
            fence=self.fence,
        )

        # The command will not have finished executing until the fence is signalled.
        # So we wait here.
        # We will directly after this read our buffer from the GPU,
        # and we will not be sure that the command has finished executing unless we wait for the fence.
        # Hence, we use a fence here.
        vkWaitForFences(
            device=self.device.vkDevice,
            fenceCount=1,
            pFences=[self.fence],
            waitAll=VK_TRUE,
            timeout=1000000000,
        )
        vkResetFences(device=self.device.vkDevice, fenceCount=1, pFences=[self.fence])
