import json
from sinode import *
import os
from vulkan import *

try:
    from buffer import *
except:
    from .buffer import *
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))


class Stage(Sinode):
    def __init__(
        self,
        parent,
        device,
        dim2index,
        constantsDict,
        shaderOutputBuffers,
        shaderInputBuffers,
        shaderCode="",
        sourceFilename="",
        debuggableVars=[],
        shaderInputBuffersNoDebug=[],
        stage=VK_SHADER_STAGE_VERTEX_BIT,
        name="mandlebrot",
        DEBUG=False,
    ):
        self.constantsDict = constantsDict
        self.DEBUG = DEBUG
        Sinode.__init__(self, parent)
        self.vkDevice = device.vkDevice
        self.device = device
        self.name = name
        self.stage = stage

        self.shaderOutputBuffers = shaderOutputBuffers
        self.debuggableVars = debuggableVars
        self.shaderInputBuffers = shaderInputBuffers
        self.shaderInputBuffersNoDebug = shaderInputBuffersNoDebug

        self.debugBuffers = []

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

        self.buffers = []
        # create all buffers according to their description
        for s in allBufferDescriptions:
            format = VK_FORMAT_R32_SFLOAT

            if s in shaderOutputBuffers or s in debuggableVars:
                descriptorSet = device.descriptorPool.descSetGlobal
                usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            else:
                descriptorSet = device.descriptorPool.descSetUniform
                usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT

            newBuff = Buffer(
                device=device,
                memtype=s["type"],
                descriptorSet=descriptorSet,
                qualifier="out",
                name=s["name"],
                readFromCPU=True,
                dimensionNames=s["dims"],
                dimensionVals=[constantsDict[d] for d in s["dims"]],
                usage=usage,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                location=0,
                format=format,
            )
            self.buffers += [newBuff]

            # make the buffer accessable as a local attribute
            exec("self." + s["name"] + " = newBuff")

            # save these buffers for reading later
            if s not in self.shaderInputBuffersNoDebug:
                self.debugBuffers += [newBuff]

        outfilename = self.name + ".spv"
        # if its spv (compiled), just run it
        if sourceFilename != "":

            if sourceFilename.endswith(".spv"):
                with open(sourceFilename, "rb") as f:
                    spirv = f.read()
            # if its not an spv, compile it
            else:
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
        self.shader_create = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            # flags=0,
            codeSize=len(spirv),
            pCode=spirv,
        )

        self.vkShaderModule = vkCreateShaderModule(
            self.vkDevice, self.shader_create, None
        )

        # Create Shader stage
        self.shader_stage_create = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=self.stage,
            module=self.vkShaderModule,
            flags=0,
            pSpecializationInfo=None,
            pName="main",
        )
        print("creating Stage " + str(stage))

    def getBufferByName(self, name):
        for b in self.buffers:
            if name == b.name:
                return b

    def compile(self, glslCode):

        # PREPROCESS THE SHADER CODE
        # RELATIVE TO DEFINED BUFFERS
        STRUCTS_STRING = "\n"
        # Create STRUCTS for each structured datatype
        reqdTypes = [b.type for b in self.buffers]
        with open(os.path.join(here, "derivedtypes.json"), "r") as f:
            derivedDict = json.loads(f.read())
            for structName, composeDict in derivedDict.items():
                if structName in reqdTypes:
                    STRUCTS_STRING += "struct " + structName + "{\n"
                    for name, ctype in composeDict.items():
                        STRUCTS_STRING += "  " + ctype + " " + name + ";\n"

                    STRUCTS_STRING += "};\n\n"

        BUFFERS_STRING = ""
        # novel INPUT buffers belong to THIS Stage (others are linked)
        for buffer in self.buffers:
            # THIS IS STUPID AND WRONG
            # FUCK
            if (
                self.stage == VK_SHADER_STAGE_FRAGMENT_BIT
                and buffer.name == "fragColor"
            ):
                buffer.qualifier = "in"
            if self.stage != VK_SHADER_STAGE_COMPUTE_BIT:
                BUFFERS_STRING += buffer.getDeclaration()
            else:
                BUFFERS_STRING += buffer.getComputeDeclaration()

        # put structs and buffers into the code
        glslCode = glslCode.replace("BUFFERS_STRING", BUFFERS_STRING).replace(
            "STRUCTS_STRING", STRUCTS_STRING
        )
        VARSDECL = ""
        if self.DEBUG:
            glslCode = self.addIndicesToOutputs(debuggableVars, glslCode)
        else:
            # otherwise, just declare the variable type
            # INITIALIZE TO 0 !
            for var in self.debuggableVars:
                VARSDECL += var["type"] + " " + var["name"] + " = 0;\n"

        # add definitions from constants dict
        DEFINE_STRING = ""
        for k, v in self.constantsDict.items():
            DEFINE_STRING += "#define " + k + " " + str(v) + "\n"
        glslCode = glslCode.replace("DEFINE_STRING", DEFINE_STRING)
        glslCode = glslCode.replace("VARIABLEDECLARATIONS", VARSDECL)

        # COMPILE GLSL TO SPIR-V
        print("compiling Stage")
        glslFilename = os.path.join(self.name + ".comp")
        with open(glslFilename, "w+") as f:
            f.write(glslCode)

        # delete the old one
        # POS always outputs to "a.spv"
        compiledFilename = "a.spv"
        if os.path.exists(compiledFilename):
            os.remove(compiledFilename)
        print("running " + glslFilename)
        os.system("glslc " + glslFilename)
        with open(compiledFilename, "rb") as f:
            spirv = f.read()
        return spirv

    def getVertexBuffers(self):
        allVertexBuffers = []
        for b in self.buffers:
            if b.usage == VK_BUFFER_USAGE_VERTEX_BUFFER_BIT:
                allVertexBuffers += [b]
        return allVertexBuffers

    def dumpMemory(self):

        outdict = {}
        for b in self.debugBuffers:
            if b.sizeBytes > 2 ** 14:
                continue
            rcvdArray = b.getAsNumpyArray()
            # convert to list to make it JSON serializable
            outdict[b.name] = rcvdArray.tolist()  # nested lists with same data, indices
        with open("debug.json", "w+") as f:
            json.dump(outdict, f, indent=4)

    def release(self):
        print("destroying Stage")
        Sinode.release(self)
        vkDestroyShaderModule(self.vkDevice, self.vkShaderModule, None)

    # take the template GLSL file
    # and add output indices
    # mostly for debugging
    def addIndicesToOutputs(self, outvars, shaderGLSL):
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


class VertexStage(Stage):
    def __init__(
        self, parent, device, buffers, constantsDict, name="mandlebrot", DEBUG=False
    ):
        glslCode = """
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
STRUCTS_STRING
BUFFERS_STRING
void main() {                         
    gl_Position = vec4(position, 1.0);
    fragColor = color;                
}                                     
"""

        Stage.__init__(
            self,
            parent,
            device,
            buffers,
            constantsDict,
            stage=VK_SHADER_STAGE_VERTEX_BIT,
            name="vertex.vert",
            DEBUG=False,
        )

        # shader code belongs to the stage
        self.compile(glslCode)


class FragmentStage(Stage):
    def __init__(
        self,
        parent,
        device,
        buffers,
        constantsDict,
        stage=VK_SHADER_STAGE_VERTEX_BIT,
        name="mandlebrot",
        DEBUG=False,
    ):
        glslCode = """
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
STRUCTS_STRING
BUFFERS_STRING
void main() {
    outColor = vec4(fragColor, 1.0);
}
"""
        Stage.__init__(
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
