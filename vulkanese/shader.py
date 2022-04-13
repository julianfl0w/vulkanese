import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *
from buffer import *
from pathlib import Path

class Shader(Sinode):
	def __init__(self, pipeline, setupDict):
		Sinode.__init__(self, pipeline)
		self.vkDevice = pipeline.device.vkDevice
		self.setupDict = setupDict
		self.pipeline  = pipeline
		self.outputWidthPixels  = setupDict["outputWidthPixels"]
		self.outputHeightPixels = setupDict["outputHeightPixels"]
		
		print("creating shader with description")
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
				
		inlocation = 0
		outlocation = 0
		
		# novel INPUT buffers belong to THIS shader (others are linked)
		for bufferName, bufferDict in setupDict["buffers"].items():
			if type(bufferDict) is not dict:
				continue
			if bufferDict.get("qualifier") is None:
				bufferDict["qualifier"] = "" 
				
			bufferMatch = False
			for existingBuffer in pipeline.getAllBuffers():
				print(bufferDict["name"] + " : " + existingBuffer.setupDict["name"])
				if bufferDict["name"] == existingBuffer.setupDict["name"]:
					print(bufferDict["name"] + " exists already. linking")
					bufferDict["location"] = existingBuffer.setupDict["location"]
					bufferDict["type"] = existingBuffer.setupDict["type"]
					bufferMatch = existingBuffer
			
			if bufferMatch:
				for k, v in bufferDict.items():
					bufferMatch.setupDict[k] = v
				shader_spirv += bufferMatch.getDeclaration()
				
			else:
				if bufferDict["name"] == "pixelIn":
					bufferDict["location"] = 1
				#elif "out" in bufferDict["qualifier"]:
				#	bufferDict["location"] = outlocation
				#	outlocation += self.getSize(bufferDict["type"])
				else:
					bufferDict["location"] = inlocation
					inlocation += self.getSize(bufferDict["type"])
					
				if "VERTEX" in setupDict["stage"]:
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
			
		print("---final shader code---")
		print(shader_spirv)
		print("--- end shader code ---")
		
		print("compiling shader")
		compShadersPath = os.path.join(here, "compiledShaders")
		compShadersPath = "compiledShaders"
		Path(compShadersPath).mkdir(parents=True, exist_ok=True)
		basename = os.path.basename(setupDict["header"])
		outfilename = os.path.join(compShadersPath, basename.replace(".header", ""))
		with open(outfilename, 'w+') as f:
			f.write(shader_spirv)
		
		os.system("glslc " + outfilename)
		# POS always outputs to "a.spv"
		with open("a.spv", 'rb') as f:
			shader_spirv = f.read()
			
		# Create shader
		self.shader_create = VkShaderModuleCreateInfo(
			sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			flags=0,
			codeSize=len(shader_spirv),
			pCode=shader_spirv
		)

		self.vkShader = vkCreateShaderModule(self.vkDevice, self.shader_create, None)
		
		# Create shader stage
		self.shader_stage_create = VkPipelineShaderStageCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			stage=eval(setupDict["stage"]),
			module=self.vkShader,
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
		print("destroying shader")
		Sinode.release(self)
		vkDestroyShaderModule(self.vkDevice, self.vkShader, None)
		