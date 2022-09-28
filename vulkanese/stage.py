import json
from vutil import *
import os
from vulkan import *
from buffer import *
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))


class Stage(Sinode):
    def __init__(
        self,
        parent,
        device,
        buffers,
        constantsDict,
        stage=VK_SHADER_STAGE_VERTEX_BIT,
        name="mandlebrot",
        DEBUG=False,
        glslCode=None,
    ):
        if glslCode is not None:
            self.glslCode = glslCode
        self.constantsDict = constantsDict
        self.buffers = buffers
        self.DEBUG = DEBUG
        Sinode.__init__(self, parent)
        self.vkDevice = device.vkDevice
        self.device = device
        self.name = name
        self.stage = stage
        print("creating Stage " + str(stage))
    
    def getBufferByName(self, name):
        for b in self.buffers:
            if name == b.name:
                return b
    
    def compile(self):

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
            if self.stage == VK_SHADER_STAGE_FRAGMENT_BIT and buffer.name == "fragColor":
                buffer.qualifier = "in"
            if self.stage != VK_SHADER_STAGE_COMPUTE_BIT:
                BUFFERS_STRING += buffer.getDeclaration()
            else:
                BUFFERS_STRING += buffer.getComputeDeclaration()
                

        # put structs and buffers into the code
        self.glslCode = self.glslCode.replace("BUFFERS_STRING", BUFFERS_STRING).replace("STRUCTS_STRING", STRUCTS_STRING)

        # print("---final Stage code---")
        # print(shader_spirv)
        # print("--- end Stage code ---")

        
        print("compiling Stage")
        compStagesPath = os.path.join(here, "compiledStages")
        compStagesPath = "compiledStages"
        Path(compStagesPath).mkdir(parents=True, exist_ok=True)

        glslFilename = os.path.join(compStagesPath, self.name)
        with open(glslFilename, "w+") as f:
            f.write(self.glslCode)

        # delete the old one
        # POS always outputs to "a.spv"
        compiledFilename = "a.spv"
        if os.path.exists(compiledFilename):
            os.remove(compiledFilename)
        os.system("glslc " + self.name)
        with open(compiledFilename, "rb") as f:
            shader_spirv = f.read()

        # Create Stage
        self.shader_create = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            # flags=0,
            codeSize=len(shader_spirv),
            pCode=shader_spirv,
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

    def getVertexBuffers(self):
        allVertexBuffers = []
        for b in self.buffers:
            if b.usage==VK_BUFFER_USAGE_VERTEX_BUFFER_BIT:
                allVertexBuffers += [b]
        return allVertexBuffers

    
    def release(self):
        print("destroying Stage")
        Sinode.release(self)
        vkDestroyShaderModule(self.vkDevice, self.vkShaderModule, None)

class VertexStage(Stage):
    def __init__(
        self,
        parent,
        device,
        buffers,
        constantsDict,
        name="mandlebrot",
        DEBUG=False,
    ):
        self.glslCode = """
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
        self.glslCode = """
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
        
        # shader code belongs to the stage
    
    