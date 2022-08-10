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
        device,
        header,
        main,
        existingBuffers,
        outputWidthPixels=500,
        outputHeightPixels=500,
        stage=VK_SHADER_STAGE_VERTEX_BIT,
        name="mandlebrot",
        buffers=[],
    ):
        baseFilename = name
        Sinode.__init__(self, None)
        self.vkDevice = device.vkDevice
        self.device = device
        self.name = name
        self.outputWidthPixels = outputWidthPixels
        self.outputHeightPixels = outputHeightPixels
        self.stage = stage
        print("creating Stage " + str(stage))

        # attributes are ex. location, normal, color
        self.buffers = buffers
        # POS always outputs to "a.spv"
        compiledFilename = "a.spv"
        shader_spirv = header

        shader_spirv += "\n"
        # if stage != VK_SHADER_STAGE_COMPUTE_BIT:
        reqdTypes = [b.type for b in buffers]
        with open(os.path.join(here, "derivedtypes.json"), "r") as f:
            derivedDict = json.loads(f.read())
            for structName, composeDict in derivedDict.items():
                if structName in reqdTypes:
                    shader_spirv += "struct " + structName + "{\n"
                    for name, ctype in composeDict.items():
                        shader_spirv += "  " + ctype + " " + name + ";\n"

                    shader_spirv += "};\n\n"

        self.children += buffers

        # novel INPUT buffers belong to THIS Stage (others are linked)
        for buffer in buffers:
            if stage != VK_SHADER_STAGE_COMPUTE_BIT:
                shader_spirv += buffer.getDeclaration()
            else:
                shader_spirv += buffer.getComputeDeclaration()
            if buffer.name == "INDEX":
                self.pipeline.indexBuffer = buffer

        shader_spirv += main

        # print("---final Stage code---")
        # print(shader_spirv)
        # print("--- end Stage code ---")

        print("compiling Stage")
        compStagesPath = os.path.join(here, "compiledStages")
        compStagesPath = "compiledStages"
        Path(compStagesPath).mkdir(parents=True, exist_ok=True)

        with open(baseFilename, "w+") as f:
            f.write(shader_spirv)

        # delete the old one
        if os.path.exists(compiledFilename):
            os.remove(compiledFilename)
        os.system("glslc " + baseFilename)
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
            stage=stage,
            module=self.vkShaderModule,
            flags=0,
            pSpecializationInfo=None,
            pName="main",
        )

    def getVertexBuffers(self):
        allVertexBuffers = []
        for b in self.buffers:
            if type(b) == VertexBuffer:
                allVertexBuffers += [b]
        return allVertexBuffers

    def release(self):
        print("destroying Stage")
        Sinode.release(self)
        vkDestroyShaderModule(self.vkDevice, self.vkShaderModule, None)
