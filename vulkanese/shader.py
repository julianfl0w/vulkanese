import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *
from buffer import *
from pathlib import Path

class Shader(PrintClass):
	def __init__(self, pipeline, shaderDict):
		PrintClass.__init__(self)
		self.vkDevice = pipeline.device.vkDevice
		self.outputWidthPixels  = shaderDict["outputWidthPixels"]
		self.outputHeightPixels = shaderDict["outputHeightPixels"]
		
		print("creating shader with description")
		print(json.dumps(shaderDict, indent=4))

		
		self.buffers = {}
		# apply template if shader is not precompiled
		if shaderDict["path"].endswith("template"):
			with open(shaderDict["path"], 'r') as f:
				shader_spirv = f.read()
				
			for dataName, bufferSize in shaderDict["buffers"].items():
				shader_spirv = shader_spirv.replace("LOCATION_" + dataName, str(pipeline.resourceIndex))
				pipeline.resourceIndex += 1
				newBuffer = Buffer(pipeline.device, bufferSize, dataName)
				self.buffers[dataName] = newBuffer
				self.children += [newBuffer]
			
			print("---final shader code---")
			print(shader_spirv)
			print("--- end shader code ---")
			
			print("compiling shader")
			compShadersPath = os.path.join(here, "compiledShaders")
			compShadersPath = "compiledShaders"
			Path(compShadersPath).mkdir(parents=True, exist_ok=True)
			basename = os.path.basename(shaderDict["path"])
			outfilename = os.path.join(compShadersPath, basename.replace(".template", ""))
			with open(outfilename, 'w+') as f:
				f.write(shader_spirv)
			
			os.system("glslc " + outfilename)
			# POS always outputs to "a.spv"
			with open("a.spv", 'rb') as f:
				shader_spirv = f.read()
			
		else:
			self.path = os.path.join(here, shaderDict["path"])
			with open(self.path, 'rb') as f:
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
			stage=eval(shaderDict["stage"]),
			module=self.vkShader,
			flags=0,
			pSpecializationInfo=None,
			pName='main')
		
		self.children += [self.vkShader, self.shader_stage_create]
		
	def release(self):
		print("destroying shader")
		PrintClass.release(self)
		vkDestroyShaderModule(self.vkDevice, self.vkShader, None)
		