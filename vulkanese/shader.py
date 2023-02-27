import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode")))
import sinode.sinode as sinode
import re
import vulkan as vk
from . import buffer
from . import compute_pipeline

from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))


class Empty:
    def __init__(self):
        pass


class Shader(sinode.Sinode):
    def __init__(
        self,
        device,
        constantsDict,
        buffers,
        sourceFilename="",
        stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
        name="mandlebrot",
        DEBUG=False,
        workgroupCount=[1, 1, 1],
        compressBuffers=True,
        waitSemaphores=[],
        waitStages=None,
        signalSemaphoreCount=0,  # these only used for compute shaders
        fenceCount=0,  # these only used for compute shaders
        useFence=False,
    ):
        self.waitStages = waitStages
        self.constantsDict = constantsDict
        self.DEBUG = DEBUG
        self.device = device
        self.name = name
        self.basename = sourceFilename[:-2]  # take out ".c"
        self.stage = stage
        self.buffers = buffers
        self.gpuBuffers = Empty()
        sinode.Sinode.__init__(self, parent=device)

        self.debugBuffers = []
        for b in buffers:
            # make the buffer accessable as a local attribute
            exec("self.gpuBuffers." + b.name + "= b")

            # keep the debug buffers separately
            if b.DEBUG:
                self.debugBuffers += [b]

        outfilename = self.basename + ".spv"
        self.sourceFilename = sourceFilename
        # if its spv (compiled), just run it
        if sourceFilename.endswith(".spv"):
            with open(sourceFilename, "rb") as f:
                spirv = f.read()
        # if its not an spv, compile it
        elif sourceFilename.endswith(".c"):
            spirv = self.compile()
            with open(outfilename, "wb+") as f:
                f.write(spirv)
        else:
            raise Exception("source template filename must end with .c")

        # Create Stage
        self.vkShaderModuleCreateInfo = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            # flags=0,
            codeSize=len(spirv),
            pCode=spirv,
        )

        self.vkShaderModule = vk.vkCreateShaderModule(
            self.device.vkDevice, self.vkShaderModuleCreateInfo, None
        )

        # Create Shader stage
        self.vkPipelineShaderStageCreateInfo = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=self.stage,
            module=self.vkShaderModule,
            flags=0,
            pSpecializationInfo=None,
            pName="main",
        )
        self.device.instance.debug("creating Stage " + str(stage))

        # if this is a compute shader, it corresponds with a single pipeline. we create that here
        if stage == vk.VK_SHADER_STAGE_COMPUTE_BIT:
            # generate a compute cmd buffer
            self.computePipeline = compute_pipeline.ComputePipeline(
                parent=self,
                computeShader=self,
                device=self.device,
                constantsDict=self.constantsDict,
                workgroupCount=workgroupCount,
                waitSemaphores=waitSemaphores,
                signalSemaphoreCount=signalSemaphoreCount,
                useFence=useFence,
            )
            #self.computePipeline.children += [self]

        # first run is always slow
        # run once in init so people dont judge the first run
        # self.run()

    def release(self):
        self.device.instance.debug("destroying Stage")
        vk.vkDestroyShaderModule(self.device.vkDevice, self.vkShaderModule, None)

    def compile(self):

        with open(self.sourceFilename, "r") as f:
            glslCode = f.read()

        # PREPROCESS THE SHADER CODE
        # RELATIVE TO DEFINED BUFFERS

        BUFFERS_STRING = ""
        # novel INPUT buffers belong to THIS Stage (others are linked)
        for b in self.buffers:
            if self.stage == vk.VK_SHADER_STAGE_FRAGMENT_BIT and b.name == "fragColor":
                b.qualifier = "in"
            if self.stage != vk.VK_SHADER_STAGE_COMPUTE_BIT:
                BUFFERS_STRING += b.getDeclaration()
            else:
                BUFFERS_STRING += b.getComputeDeclaration()

        if self.DEBUG:
            glslCode = self.addIndicesToOutputs(glslCode)

        # put structs and buffers into the code
        glslCode = glslCode.replace("BUFFERS_STRING", BUFFERS_STRING)

        # add definitions from constants dict
        DEFINE_STRING = ""
        for k, v in self.constantsDict.items():
            DEFINE_STRING += "#define " + k + " " + str(v) + "\n"
        glslCode = glslCode.replace("DEFINE_STRING", DEFINE_STRING)

        # COMPILE GLSL TO SPIR-V
        self.device.instance.debug("compiling Stage")
        if self.stage == vk.VK_SHADER_STAGE_VERTEX_BIT:
            glslFilename = os.path.join(self.basename + ".vert")
        elif self.stage == vk.VK_SHADER_STAGE_COMPUTE_BIT:
            glslFilename = os.path.join(self.basename + ".comp")
        elif self.stage == vk.VK_SHADER_STAGE_FRAGMENT_BIT:
            glslFilename = os.path.join(self.basename + ".frag")
        with open(glslFilename, "w+") as f:
            f.write(glslCode)

        # delete the old one
        # POS always outputs to "a.spv"
        compiledFilename = "a.spv"
        if os.path.exists(compiledFilename):
            os.remove(compiledFilename)
        self.device.instance.debug("running " + glslFilename)
        # os.system("glslc --scalar-block-layout " + glslFilename)
        glslcbin = os.path.join(here, "glslc")
        os.system(glslcbin + " --target-env=vulkan1.1 " + glslFilename)
        with open(compiledFilename, "rb") as f:
            spirv = f.read()
        return spirv

    def run(self, blocking=True):
        self.computePipeline.run(blocking=blocking)

    def wait(self):
        self.computePipeline.wait()

    def getVertexBuffers(self):
        allVertexBuffers = []
        for b in self.buffers:
            if b.usage == VK_BUFFER_USAGE_VERTEX_BUFFER_BIT:
                allVertexBuffers += [b]
        return allVertexBuffers

    def dumpMemory(self, filename="debug.json"):
        outdict = {}
        for b in self.buffers:

            if not b.DEBUG:
                continue
            rcvdArray = b.getAsNumpyArray()[0]
            # convert to list to make it JSON serializable
            outdict[b.name] = rcvdArray.tolist()  # nested lists with same data, indices
        with open(filename, "w+") as f:
            self.device.instance.debug("dumping to " + filename)
            json.dump(outdict, f, indent=4)

    # take the template GLSL file
    # and add output indices
    # mostly for debugging
    def addIndicesToOutputs(self, shaderGLSL):
        indexedVarStrings = []
        # if this is debug mode, all vars have already been declared as output buffers
        # intermediate variables must be indexed
        for b in self.debugBuffers:
            indexedVarString = b.name + "["
            for i, d in enumerate(b.dimIndexNames):
                indexedVarString += d
                # since GLSL doesnt allow multidim arrays (kinda, see gl_arb_arrays_of_arrays)
                # ok i dont understand Vulkan multidim
                for j, rd in enumerate(b.dimensionVals[i + 1 :]):
                    indexedVarString += "*" + str(rd)
                if i < (len(b.dimIndexNames) - 1):
                    indexedVarString += "+"
            indexedVarString += "]"
            indexedVarStrings += [(b.name, indexedVarString)]

        outShaderGLSL = ""
        for line in shaderGLSL.split("\n"):
            # replace all non-comments
            if not line.strip().startswith("//") and not line.strip().endswith("debug"):
                # whole-word replacement
                for iv in indexedVarStrings:
                    line = re.sub(r"\b{}\b".format(iv[0]), iv[1], line)
            # keep the comments
            outShaderGLSL += line + "\n"

        return outShaderGLSL


class VertexStage(Shader):
    def __init__(
        self,
        device,
        buffers,
        constantsDict,
        sourceFilename,
        name="mandlebrot",
        DEBUG=False,
    ):
        Shader.__init__(
            self,
            device=device,
            buffers=buffers,
            constantsDict=constantsDict,
            sourceFilename=sourceFilename,
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
            name=name,
            DEBUG=DEBUG,
        )


class FragmentStage(Shader):
    def __init__(
        self,
        device,
        buffers,
        constantsDict,
        sourceFilename,
        name="mandlebrot",
        DEBUG=False,
    ):
        Shader.__init__(
            self,
            device=device,
            buffers=buffers,
            constantsDict=constantsDict,
            sourceFilename=sourceFilename,
            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            name=name,
            DEBUG=False,
        )
