import json
from sinode import *
import os
import re
import vulkan as vk

import pkg_resources

# if "vulkanese" not in [pkg.key for pkg in pkg_resources.working_set]:
#    from buffer import *
#    from computepipeline import *
#    DEV = True
# else:
from . import buffer
from . import compute_pipeline

DEV = False

from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))


class Empty:
    def __init__(self):
        pass


class Shader(Sinode):
    def __init__(
        self,
        parent,
        device,
        constantsDict,
        buffers,
        shaderCode="",
        sourceFilename="",
        stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
        name="mandlebrot",
        DEBUG=False,
        workgroupCount=[1, 1, 1],
        compressBuffers=True,
        waitSemaphores=[],
    ):
        self.constantsDict = constantsDict
        self.DEBUG = DEBUG
        Sinode.__init__(self, parent)
        self.vkDevice = device.vkDevice
        self.device = device
        self.name = name
        self.basename = sourceFilename[:-2]  # take out ".c"
        self.stage = stage
        self.buffers = buffers
        self.gpuBuffers = Empty()

        self.debugBuffers = []
        for b in buffers:
            # make the buffer accessable as a local attribute
            exec("self.gpuBuffers." + b.name + "= b")

            # keep the debug buffers separately
            if b.DEBUG:
                self.debugBuffers += [b]

        outfilename = self.basename + ".spv"
        # if its spv (compiled), just run it
        if sourceFilename.endswith(".spv"):
            with open(sourceFilename, "rb") as f:
                spirv = f.read()
        # if its not an spv, compile it
        elif sourceFilename.endswith(".c"):
            with open(sourceFilename, "r") as f:
                glslSource = f.read()
            spirv = self.compile(glslSource)
            with open(outfilename, "wb+") as f:
                f.write(spirv)
        # allow the simple passing of code
        elif shaderCode != "":
            spirv = self.compile(shaderCode)
            with open(self.name + ".spv", "wb+") as f:
                f.write(spirv)
        else:
            raise ("either source filename or shader text must be provided")

        # Create Stage
        self.shader_create = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            # flags=0,
            codeSize=len(spirv),
            pCode=spirv,
        )

        self.vkShaderModule = vk.vkCreateShaderModule(
            self.vkDevice, self.shader_create, None
        )

        # Create Shader stage
        self.shader_stage_create = vk.VkPipelineShaderStageCreateInfo(
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
                computeShader=self,
                device=self.device,
                constantsDict=self.constantsDict,
                workgroupCount=workgroupCount,
                waitSemaphores=waitSemaphores,
            )
            self.computePipeline.children += [self]

        # first run is always slow
        # run once in init so people dont judge the first run
        # self.run()

    def release(self):
        self.device.instance.debug("destroying Stage")
        vk.vkDestroyShaderModule(self.vkDevice, self.vkShaderModule, None)
        Sinode.release(self)


class ComputeShader(Shader):
    def compile(self, glslCode):

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
        glslFilename = os.path.join(self.basename + ".comp")
        with open(glslFilename, "w+") as f:
            f.write(glslCode)

        # delete the old one
        # POS always outputs to "a.spv"
        compiledFilename = "a.spv"
        if os.path.exists(compiledFilename):
            os.remove(compiledFilename)
        self.device.instance.debug("running " + glslFilename)
        # os.system("glslc --scalar-block-layout " + glslFilename)
        os.system("glslc --target-env=vulkan1.1 " + glslFilename)
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
        self, parent, device, buffers, constantsDict, name="mandlebrot", DEBUG=False
    ):
        glslCode = """
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
BUFFERS_STRING
void main() {                         
    gl_Position = vec4(position, 1.0);
    fragColor = color;                
}                                     
"""

        Shader.__init__(
            self,
            parent,
            device,
            buffers,
            constantsDict,
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
            name="vertex.vert",
            DEBUG=False,
        )

        # shader code belongs to the stage
        self.compile(glslCode)


class FragmentStage(Shader):
    def __init__(
        self,
        parent,
        device,
        buffers,
        constantsDict,
        stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
        name="mandlebrot",
        DEBUG=False,
    ):
        glslCode = """
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
BUFFERS_STRING
void main() {
    outColor = vec4(fragColor, 1.0);
}
"""
        Shader.__init__(
            self,
            parent,
            device,
            buffers,
            constantsDict,
            stage=VK_SHADER_STAGE_FRAGMENT_BIT,
            name="fragment.frag",
            DEBUG=False,
        )

        self.compile(glslCode)

        # shader code belongs to the stage
