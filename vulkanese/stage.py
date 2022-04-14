import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *
from buffer import *
from pathlib import Path

class Stage(Sinode):
	def __init__(self, pipeline, setupDict):
		Sinode.__init__(self, pipeline)
		self.vkDevice = pipeline.device.vkDevice
		self.setupDict = setupDict
		self.pipeline  = pipeline
		self.outputWidthPixels  = setupDict["outputWidthPixels"]
		self.outputHeightPixels = setupDict["outputHeightPixels"]
		
		print("creating Stage with description")
		print(json.dumps(setupDict, indent=4))

		# attributes are ex. location, normal, color
		self.buffers    = {}
		
		with open(setupDict["header"]) as f:
			shader_spirv = f.read()
		
		shader_spirv += "\n"
		with open(os.path.join(here, "derivedtypes.json"), 'r') as f:
			derivedDict = json.loads(f.read())
			for structName, composeDict in derivedDict.items():
				shader_spirv += "struct " + structName + "\n"
				shader_spirv += "{\n"
				for name, ctype in composeDict.items():
					shader_spirv += "    " + ctype + " " + name + ";\n"
					
				shader_spirv += "};\n\n" 
				
		location = 0
		
		# novel INPUT buffers belong to THIS Stage (others are linked)
		for bufferName, bufferDict in setupDict["buffers"].items():
			if type(bufferDict) is not dict:
				continue
			for k, v in setupDict["defaultbuffer"].items():
				if k not in bufferDict.keys():
					bufferDict[k] = v
				
			bufferMatch = False
			for existingBuffer in pipeline.getAllBuffers():
				print(bufferDict["name"] + " : " + existingBuffer.setupDict["name"])
				if bufferDict["name"] == existingBuffer.setupDict["name"]:
					print(bufferDict["name"] + " exists already. linking")
					bufferMatch = existingBuffer
			
			if bufferMatch:
				for k, v in bufferDict.items():
					bufferMatch.setupDict[k] = v
				shader_spirv += bufferMatch.getDeclaration()
				
			else:
				bufferDict["location"] = location
				location += self.getSize(bufferDict["type"])
					
				if "vertex" in setupDict["name"]:
					newBuffer     = VertexBuffer(pipeline.device, bufferDict)
				else:
					newBuffer     = Buffer(pipeline.device, bufferDict)

				shader_spirv  += newBuffer.getDeclaration()
					
				self.buffers [bufferName] = newBuffer
				self.children += [newBuffer]
			
				if bufferDict["name"] == "INDEX":
					self.pipeline.indexBuffer = newBuffer
			
				
		with open(setupDict["main"]) as f:
			shader_spirv += f.read()
			
		print("---final Stage code---")
		print(shader_spirv)
		print("--- end Stage code ---")
		
		print("compiling Stage")
		compStagesPath = os.path.join(here, "compiledStages")
		compStagesPath = "compiledStages"
		Path(compStagesPath).mkdir(parents=True, exist_ok=True)
		basename = os.path.basename(setupDict["header"])
		outfilename = os.path.join(compStagesPath, basename.replace(".header", ""))
		with open(outfilename, 'w+') as f:
			f.write(shader_spirv)
		
		os.system("glslc " + outfilename)
		# POS always outputs to "a.spv"
		with open("a.spv", 'rb') as f:
			shader_spirv = f.read()
			
		# Create Stage
		self.shader_create = VkShaderModuleCreateInfo(
			sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			flags=0,
			codeSize=len(shader_spirv),
			pCode=shader_spirv
		)

		self.vkStage = vkCreateShaderModule(self.vkDevice, self.shader_create, None)
		
		# Create Shader stage
		self.shader_stage_create = VkPipelineShaderStageCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			stage=eval(setupDict["stage"]),
			module=self.vkStage,
			flags=0,
			pSpecializationInfo=None,
			pName='main')
	
	def getSize(self, bufftype):
		with open(os.path.join(here, "derivedtypes.json"), 'r') as f:
			derivedDict = json.loads(f.read())
		with open(os.path.join(here, "ctypes.json"), 'r') as f:
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
		vkDestroyShaderModule(self.vkDevice, self.vkStage, None)
		