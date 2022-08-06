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
        pipeline,
        setupDict,
        outputWidthPixels=500,
        outputHeightPixels=500,
        stage="vertex",
        name="mandlebrot",
        buffers = []
    ):
        Sinode.__init__(self, pipeline)
        self.vkInstance = pipeline.instance.vkInstance
        self.vkDevice = pipeline.device.vkDevice
        self.setupDict = setupDict
        self.pipeline = pipeline
        self.name = name
        self.outputWidthPixels = outputWidthPixels
        self.outputHeightPixels = outputHeightPixels
        self.stage = stage
        print("creating Stage with description")
        print(json.dumps(setupDict, indent=4))

        # attributes are ex. location, normal, color
        self.buffers = {}
        if stage="vertex":
            baseFilename = name + ".vert"
            
        headerFilename = baseFilename + ".header"
        mainFilename   = baseFilename + ".main"
        
        with open(headerFilename) as f:
            shader_spirv = f.read()

        shader_spirv += "\n"
        with open(os.path.join(here, "derivedtypes.json"), "r") as f:
            derivedDict = json.loads(f.read())
            for structName, composeDict in derivedDict.items():
                shader_spirv += "struct " + structName + "\n"
                shader_spirv += "{\n"
                for name, ctype in composeDict.items():
                    shader_spirv += "    " + ctype + " " + name + ";\n"

                shader_spirv += "};\n\n"

        location = 0

        # novel INPUT buffers belong to THIS Stage (others are linked)
        for buffer in buffers:

            bufferMatch = False
            for existingBuffer in pipeline.getAllBuffers():
                print(buffer.name + " : " + existingBuffer.name)
                if buffer.name == existingBuffer.name:
                    print(buffer.name + " exists already. linking")
                    bufferMatch = existingBuffer

            if bufferMatch:
                shader_spirv += bufferMatch.getDeclaration()

            else:
                buffer.location = location
                location += self.getSize(buffer.type)

                if stage == "vertex":
                    newBuffer = VertexBuffer(pipeline.device, bufferDict)
                else:
                    newBuffer = Buffer(pipeline.device, bufferDict)

                shader_spirv += newBuffer.getDeclaration()

                self.children += [buffer]

                if buffer.name == "INDEX":
                    self.pipeline.indexBuffer = buffer

        with open(setupDict["main"]) as f:
            shader_spirv += f.read()

        # print("---final Stage code---")
        # print(shader_spirv)
        # print("--- end Stage code ---")

        print("compiling Stage")
        compStagesPath = os.path.join(here, "compiledStages")
        compStagesPath = "compiledStages"
        Path(compStagesPath).mkdir(parents=True, exist_ok=True)
		
        with open(baseFilename, "w+") as f:
            f.write(shader_spirv)

        os.system("glslc " + baseFilename)
        # POS always outputs to "a.spv"
        with open("a.spv", "rb") as f:
            shader_spirv = f.read()

        # Create Stage
        self.shader_create = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            flags=0,
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

    def getSize(self, bufftype):
        with open(os.path.join(here, "derivedtypes.json"), "r") as f:
            derivedDict = json.loads(f.read())
        with open(os.path.join(here, "ctypes.json"), "r") as f:
            cDict = json.loads(f.read())
        size = 0
        if bufftype in derivedDict.keys():
            for subtype in derivedDict[bufftype]:
                size += self.getSize(subtype)
        else:
            size += 1
        return size

    def getVertexBuffers(self):
        allVertexBuffers = []
        for b in self.buffers.values():
            if type(b) == VertexBuffer:
                allVertexBuffers += [b]
        return allVertexBuffers

    def release(self):
        print("destroying Stage")
        Sinode.release(self)
        vkDestroyShaderModule(self.vkDevice, self.vkShaderModule, None)
