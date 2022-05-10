import os
import time
import json
from vulkan import *
from surface import *
from stage import *
from renderpass import *
from pipelines import *
from commandbuffer import *
from vutil import *
from vulkanese import *
from PIL import Image as pilImage
import faulthandler
import cffi
import logging
import jvulkan
from pprint import pprint

from inspect import currentframe


def jlog(instr):
	cf = currentframe()
	ln = cf.f_back.f_lineno
	print(str(ln) + ": " + str(instr))

# WORKAROUND SEGFAULT
from ctypes import *
ctypes_vulkanlib = CDLL("libvulkan.so.1.3.211")

faulthandler.enable()

here = os.path.dirname(os.path.abspath(__file__))
def getVulkanesePath():
	return here

from enum import Enum
class StageIndices(Enum):
	eRaygen           = 0
	eMiss             = 1
	eMiss2            = 2
	eClosestHit       = 3
	eShaderGroupCount = 4
	
# WORKAROUND SEGFAULT
class jbufferDeviceAddressInfo(Structure):
	_fields_ = [("sType", c_long),
			   ("pNext", c_void_p),
			   ("buffer", c_void_p)]
	
def getAddressFromString(inobj):
	return eval(inobj.__str__().split(" ")[-1][:-1])
	
class RaytracePipeline(Pipeline):
	def __init__(self, device, setupDict):
		
		logger = logging.getLogger('dtfsdfm')
		logger.setLevel(logging.DEBUG)

		handler = logging.StreamHandler(sys.stdout)
		handler.setLevel(logging.DEBUG)
		formatter = logging.Formatter('{%(filename)s:%(lineno)d %(message)s}')
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		self.logger = logger

		Pipeline.__init__(self, device, setupDict)
		
		self.stages = [s.shader_stage_create for s in self.stageDict.values()]
		
		# in raytracing, there may be alternative shaders
		# for example, an occlusion miss shader and a diffraction one
		# there are always 4 stages:
		#  Raygen
		#  Miss
		#  Hit
		#  Callable
		# Each of these has its own SBT, represented as a strided region
		missCount = 1
		hitCount  = 1
		handleCount = 1 + missCount + hitCount
		
		self.SBTDict = {}
		baseSBT = {"stride": 0, "size": 0}
		self.SBTDict["gen"     ] = baseSBT.copy()
		self.SBTDict["callable"] = baseSBT.copy()
		self.SBTDict["miss"    ] = baseSBT.copy()
		self.SBTDict["hit"     ] = baseSBT.copy()
		self.SBTDict["gen"     ]["stage"] = VK_SHADER_STAGE_RAYGEN_BIT_KHR
		self.SBTDict["callable"]["stage"] = VK_SHADER_STAGE_CALLABLE_BIT_KHR
		self.SBTDict["miss"    ]["stage"] = VK_SHADER_STAGE_MISS_BIT_KHR
		self.SBTDict["hit"     ]["stage"] = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
		
		for stage, stageDict in self.SBTDict.items():
			jlog("processing RayTrace stage " + stage)
			for shader in self.stageDict.values():
				if eval(shader.stage) & stageDict["stage"]:
					#stageDict["size"  ] += shader.setupDict["SIZEBYTES"]
					stageDict["size"  ] += 65536
					stageDict["stride"]  = max(stageDict["stride"], 65536)
					
			# Allocate a buffer for storing the SBT.
			bufferDict = {
				"name"           : "SBT",
				"usage"          : "VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |  \
					VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR ",
				"descriptorSet"  : "global",
				"rate"           : "VK_VERTEX_INPUT_RATE_VERTEX",
				#"memProperties"  : "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT",
				"memProperties"  : "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT",
				#"sharingMode"    : "VK_SHARING_MODE_EXCLUSIVE",
				"sharingMode"    : "0",
				"SIZEBYTES"      : 65536,
				"qualifier"      : "in",
				"type"           : "vec3",
				"format"         : "VK_FORMAT_R32G32B32_SFLOAT",
				#"stage"          : "VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR",
				"stage"          : "VK_SHADER_STAGE_RAYGEN_BIT_KHR",
				"stride"         : 1
			}
			
			stageDict["buffer"] = Buffer(self.device, bufferDict)

			jlog("getting device address")
			
			#vkGetBufferDeviceAddressKHR = vkGetInstanceProcAddr(self.instance.vkInstance, 'vkGetBufferDeviceAddressKHR')

			#jlog(stageDict["buffer"].bufferDeviceAddressInfo)
			#jlog("SHIT " + str(vkGetBufferDeviceAddressKHR))
			devaddr = int(ffi.cast("uintptr_t", self.vkDevice))
			devaddr = c_void_p(devaddr)
			#bufaddr = int(ffi.cast("uintptr_t", stageDict["buffer"].bufferDeviceAddressInfo))
			bufaddr = stageDict["buffer"].bufferDeviceAddressInfo
			jlog(devaddr)
			jlog(hex(devaddr.value))
			jlog(bufaddr)
			jlog(bufaddr.sType)
			jlog(bufaddr.pNext)
			jlog(bufaddr.buffer)
			jlog(stageDict["buffer"].setupDict["name"])
			jvulkan.vkGetBufferDeviceAddress(
				{"device" : devaddr, 
				 "pInfo"  : pointer(bufaddr)}
			)
			#deviceAddress = vkGetBufferDeviceAddressKHR(
			#	self.vkDevice, 
			#	stageDict["buffer"].bufferDeviceAddressInfo
			#)
			#
			#	
			## WORKAROUND SEGFAULT
			#jlog(VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO)
			#fuckingBullshit = jbufferDeviceAddressInfo()
			#fuckingBullshit.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO
			#fuckingBullshit.pNext = None
			#BSAddress = getAddressFromString(stageDict["buffer"].vkBuffer)
			#jlog(hex(BSAddress))
			#fuckingBullshit.buffer = BSAddress
#
			#jlog(ctypes_vulkanlib)
			#jlog(ctypes_vulkanlib.vkEnumerateInstanceVersion)
			#ctypes_vulkanlib.vkEnumerateInstanceVersion.argtypes = [POINTER(c_int)]
			#CTYPES_INT_PTR = POINTER(c_int)
			#ret = CTYPES_INT_PTR()
			#ctypes_vulkanlib.vkEnumerateInstanceVersion(ret)
			#jlog(ret)
			#jlog(self.vkDevice)
			#jlog("BS " + str(ffi.addressof(stageDict["buffer"].bufferDeviceAddressInfo)))
			#BULLSHITFUCKINGBUFFERDEVICEFUCKINGADDRESSDUMBASSINFOSTUPIDFUCKINGPOINTER =  getAddressFromString(ffi.addressof(stageDict["buffer"].bufferDeviceAddressInfo))
			#jlog("FUCK " +hex(BULLSHITFUCKINGBUFFERDEVICEFUCKINGADDRESSDUMBASSINFOSTUPIDFUCKINGPOINTER))
			#ctypes_vulkanlib.vkGetBufferDeviceAddress.argtypes = [c_void_p,c_void_p]
			#jlog(ctypes_vulkanlib.vkGetBufferDeviceAddress)
			#jlog(vkGetBufferDeviceAddressKHR)
			#deviceAddress = ctypes_vulkanlib.vkGetBufferDeviceAddress(
			#	getAddressFromString(self.vkDevice), 
			#	BULLSHITFUCKINGBUFFERDEVICEFUCKINGADDRESSDUMBASSINFOSTUPIDFUCKINGPOINTER
			#)

			#IT STILL FUCKING SEGFAULTS
			
			#deviceAddress = vkGetBufferDeviceAddressKHR(
			#	self.vkDevice, 
			#	stageDict["buffer"].bufferDeviceAddressInfo
			#)

			jlog("creating strided region")
			stageDict["vkStridedDeviceAddressRegion"] = \
			VkStridedDeviceAddressRegionKHR(
				deviceAddress = deviceAddress,
				stride        = self.setupDict["stride"],
				size          = self.setupDict["SIZEBYTES"]
			)
		
		# Get the shader group handles
		vkGetRayTracingShaderGroupHandlesKHR = vkGetInstanceProcAddr(self.instance.vkInstance, 'vkGetRayTracingShaderGroupHandlesKHR')
		result = vkGetRayTracingShaderGroupHandlesKHR(self.vkDevice, self.pipeline.vkPipeline, 0, handleCount, dataSize, handles.data());
		assert(result == VK_SUCCESS)

			
		# Shader groups
		# Intersection shaders allow arbitrary intersection geometry
		# For now they are unused. therefore VK_SHADER_UNUSED_KHR is appropriate
		self.shaderGroupCreateInfo = VkRayTracingShaderGroupCreateInfoKHR(
			sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
			pNext = None,
			type  = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
			generalShader      = VK_SHADER_UNUSED_KHR,
			closestHitShader   = VK_SHADER_UNUSED_KHR,
			anyHitShader       = VK_SHADER_UNUSED_KHR,
			intersectionShader = VK_SHADER_UNUSED_KHR,
			pShaderGroupCaptureReplayHandle = None
			)
		rtShaderGroups = [self.shaderGroupCreateInfo]
		
		# Push constant: we want to be able to update constants used by the shaders
		#pushConstant = vkPushConstantRange(
		#	VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
		#	0, sizeof(PushConstantRay)};
		#VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		#pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		#.pPushConstantRanges    = &pushConstant;
	
		# Assemble the shader stages and recursion depth info into the ray tracing pipeline
		# In this case, self.rtShaderGroups.size() == 4: we have one raygen group,
		# two miss shader groups, and one hit group.
		# The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
		# hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
		# as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
		# in the ray generation to avoid deep recursion.
		self.rayPipelineInfo = VkRayTracingPipelineCreateInfoKHR(
			sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
			pNext      = None, 
			flags      = 0,
			stageCount = len(self.stages),  # Stages are shaders
			pStages    = self.stages,
			groupCount = len(rtShaderGroups),
			pGroups    = rtShaderGroups,
			maxPipelineRayRecursionDepth = 2, # Ray depth
			layout               = self.pipelineLayout
			)

		vkCreateRayTracingPipelinesKHR = vkGetInstanceProcAddr(self.instance.vkInstance, 'vkCreateRayTracingPipelinesKHR')
		
		self.vkPipeline = vkCreateRayTracingPipelinesKHR(
			device = self.vkDevice,
			deferredOperation = None, 
			pipelineCache = None, 
			createInfoCount = 1,
			pCreateInfos = [self.rayPipelineInfo],
			pAllocator = None
		)
		
		
		# create the sbt after creating the pipeline
		self.sbt = ShaderBindingTable(self)
		
		# wrap it all up into a command buffer
		jlog("Creating commandBuffer")
		self.commandBuffer = RaytraceCommandBuffer(self)
		


