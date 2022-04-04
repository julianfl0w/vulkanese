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
				
		location = 0
		# novel INPUT buffers belong to THIS shader (others are linked)
		for bufferName, bufferDict in setupDict["inBuffers"].items():
			if type(bufferDict) is not dict:
				continue
				
			existsAlready = False
			for existingBuffer in pipeline.getAllBuffers():
				print(bufferDict["name"] + " : " + existingBuffer.setupDict["name"])
				if bufferDict["name"] == existingBuffer.setupDict["name"]:
					print(bufferDict["name"] + " exists already. linking")
					bufferDict["location"] = existingBuffer.setupDict["location"]
					bufferDict["type"] = existingBuffer.setupDict["type"]
					existsAlready = True
				
			if not existsAlready:
				bufferDict["location"] = location
				location += self.getSize(bufferDict["type"])
				if "VERTEX" in setupDict["stage"]:
					newBuffer     = VertexBuffer(pipeline.device, bufferDict)
				else:
					newBuffer     = Buffer(pipeline.device, bufferDict)
					
				self.buffers [bufferName] = newBuffer
				self.children += [newBuffer]
			
				if bufferDict["name"] == "INDEX":
					self.pipeline.indexBuffer = newBuffer
			
			shader_spirv += "layout (location = " + str(bufferDict["location"]) + ") in " + bufferDict["type"] + " " + bufferDict["name"] + ";\n"
					
		location = 0
		# ALL the OUTPUT buffers are owned by THIS shader
		for bufferName, bufferDict in setupDict["outBuffers"].items():
			if type(bufferDict) is not dict:
				continue
			print(bufferDict)
			print("adding outbuff " + bufferDict["name"])
			bufferDict["location"] = location
			location += self.getSize(bufferDict["type"])
			shader_spirv  = shader_spirv.replace("LOCATION_" + bufferDict["name"], str(bufferDict["location"]))
			newBuffer     = Buffer(pipeline.device, bufferDict)
			self.buffers [bufferName] = newBuffer
			self.children += [newBuffer]
			shader_spirv += "layout (location = " + str(bufferDict["location"]) + ") out " + bufferDict["type"] + " " + bufferDict["name"] + ";\n"
				
		with open(setupDict["main"]) as f:
			shader_spirv += f.read()
			
		print("---final shader code---")
		print(shader_spirv)
		print("--- end shader code ---")
		
		print("compiling shader")
		compShadersPath = os.path.join(here, "compiledShaders")
		compShadersPath = "compiledShaders"
		Path(compShadersPath).mkdir(parents=True, exist_ok=True)
		basename = os.path.basename(setupDict["path"])
		outfilename = os.path.join(compShadersPath, basename.replace(".template", ""))
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
		