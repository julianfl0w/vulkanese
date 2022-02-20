
import ctypes
import os
import sdl2
import sdl2.ext
import time
import json
from vutil import *
from vulkan import *
from pipelines import *
from PIL import Image as pilImage

here = os.path.dirname(os.path.abspath(__file__))
def getVulkanesePath():
	return here

class Instance(PrintClass):
	def __init__(self):
		PrintClass.__init__(self)
		
		# ----------
		# Create instance
		appInfo = VkApplicationInfo(
			sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
			pApplicationName="Hello Triangle",
			applicationVersion=VK_MAKE_VERSION(1, 0, 0),
			pEngineName="No Engine",
			engineVersion=VK_MAKE_VERSION(1, 0, 0),
			apiVersion=VK_API_VERSION_1_0)

		extensions = vkEnumerateInstanceExtensionProperties(None)
		extensions = [e.extensionName for e in extensions]
		print("available extensions: ")
		for e in extensions:
			print("    " + e)

		self.layers = vkEnumerateInstanceLayerProperties()
		self.layers = [l.layerName for l in self.layers]
		print("available layers:")
		for l in self.layers:
			print("    " + l)

		if 'VK_LAYER_KHRONOS_validation' in self.layers:
			self.layers = ['VK_LAYER_KHRONOS_validation']
		else:
			self.layers = ['VK_LAYER_LUNARG_standard_validation']


		createInfo = VkInstanceCreateInfo(
			sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			flags=0,
			pApplicationInfo=appInfo,
			enabledExtensionCount=len(extensions),
			ppEnabledExtensionNames=extensions,
			enabledLayerCount=len(self.layers),
			ppEnabledLayerNames=self.layers)

		self.vkInstance = vkCreateInstance(createInfo, None)
		self.children += [self.vkInstance]
		
		# ----------
		# Debug instance
		vkCreateDebugReportCallbackEXT = vkGetInstanceProcAddr(
			self.vkInstance,
			"vkCreateDebugReportCallbackEXT")
		self.vkDestroyDebugReportCallbackEXT = vkGetInstanceProcAddr(
			self.vkInstance,
			"vkDestroyDebugReportCallbackEXT")

		def debugCallback(*args):
			print('DEBUG: ' + args[5] + ' ' + args[6])
			return 0

		debug_create = VkDebugReportCallbackCreateInfoEXT(
			sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
			flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
			pfnCallback=debugCallback)
		self.callback = vkCreateDebugReportCallbackEXT(self.vkInstance, debug_create, None)

	def getDeviceList(self):
		self.physical_devices            = vkEnumeratePhysicalDevices(self.vkInstance)
		self.physical_devices_features   = [vkGetPhysicalDeviceFeatures(physical_device)   for physical_device in self.physical_devices]
		self.physical_devices_properties = [vkGetPhysicalDeviceProperties(physical_device) for physical_device in self.physical_devices]
		return self.physical_devices_properties
			
		
	def getDevice(self, deviceIndex):
		newDev = Device(self,deviceIndex)
		self.children += [newDev]
		return newDev
		
	def release(self):
		print("destroying child devices")
		for d in self.children:
			try:
				d.release()
			except:
				pass
		print("destroying debug etc")
		self.vkDestroyDebugReportCallbackEXT(self.vkInstance, self.callback, None)
		print("destroying instance")
		vkDestroyInstance(self.vkInstance, None)
		
class Device(PrintClass):
			
	def applyLayout(self, setupDict):
		self.pipelines = []
		for pipelineDict in setupDict["pipelines"]:
			if pipelineDict["class"] == "raster":
				self.pipelines += [RasterPipeline(self, pipelineDict)]
			elif pipelineDict["class"] == "compute":
				self.pipelines += [ComputePipeline(self, pipelineDict)]
			else:
				self.pipelines += [RaytracePipeline(self, pipelineDict)]
		
		self.children += self.pipelines
		return self.pipelines
	
	def __init__(self, instance, deviceIndex):
		PrintClass.__init__(self)
		self.instance = instance
		self.deviceIndex = deviceIndex
		
		print("initializing device " + str(deviceIndex))
		self.physical_device = vkEnumeratePhysicalDevices(self.instance.vkInstance)[deviceIndex]
		
		
		print("Select queue family")
		# ----------
		# Select queue family
		vkGetPhysicalDeviceSurfaceSupportKHR = vkGetInstanceProcAddr(
			self.instance.vkInstance, 'vkGetPhysicalDeviceSurfaceSupportKHR')
		
		#queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self.physical_device)
		queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self.physical_device)
		
		print("%s available queue family" % len(queue_families))

		self.queue_family_graphic_index = -1
		self.queue_family_present_index = -1

		for i, queue_family in enumerate(queue_families):
			# Currently, we set present index like graphic index
			#support_present = vkGetPhysicalDeviceSurfaceSupportKHR(
			#	physicalDevice=physical_device,
			#	queueFamilyIndex=i,
			#	surface=sdl_surface_inst.surface)
			if (queue_family.queueCount > 0 and
			   queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT):
				self.queue_family_graphic_index = i
				self.queue_family_present_index = i
			# if queue_family.queueCount > 0 and support_present:
			#     self.queue_family_present_index = i

		print("indice of selected queue families, graphic: %s, presentation: %s\n" % (
			self.queue_family_graphic_index, self.queue_family_present_index))

		self.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE
		self.queueFamilyIndexCount = 0
		self.pQueueFamilyIndices = None

		if self.queue_family_graphic_index != self.queue_family_present_index:
			self.imageSharingMode = VK_SHARING_MODE_CONCURRENT
			self.queueFamilyIndexCount = 2
			self.pQueueFamilyIndices = [device.queue_family_graphic_index, device.queue_family_present_index]


		# ----------
		# Create logical device and queues
		extensions = vkEnumerateDeviceExtensionProperties(physicalDevice=self.physical_device, pLayerName=None)
		extensions = [e.extensionName for e in extensions]
		print("available device extensions: %s\n" % extensions)

		#only use the extensions necessary
		extensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]

		queues_create = [VkDeviceQueueCreateInfo(sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
												 queueFamilyIndex=i,
												 queueCount=1,
												 pQueuePriorities=[1],
												 flags=0)
						 for i in {self.queue_family_graphic_index,
								   self.queue_family_present_index}]

		self.device_create = VkDeviceCreateInfo(
			sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			pQueueCreateInfos=queues_create,
			queueCreateInfoCount=len(queues_create),
			pEnabledFeatures=self.instance.physical_devices_features[self.deviceIndex],
			flags=0,
			enabledLayerCount=len(self.instance.layers),
			ppEnabledLayerNames=self.instance.layers,
			enabledExtensionCount=len(extensions),
			ppEnabledExtensionNames=extensions
		)

		self.vkDevice = vkCreateDevice(self.physical_device, self.device_create, None)
		
		self.graphic_queue = vkGetDeviceQueue(
			device=self.vkDevice,
			queueFamilyIndex=self.queue_family_graphic_index,
			queueIndex=0)
		self.presentation_queue = vkGetDeviceQueue(
			device=self.vkDevice,
			queueFamilyIndex=self.queue_family_present_index,
			queueIndex=0)
	
		print("Logical device and graphic queue successfully created\n")
		
		# Create command pool
		command_pool_create = VkCommandPoolCreateInfo(
			sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			queueFamilyIndex=self.queue_family_graphic_index,
			flags=0)

		self.vkCommandPool = vkCreateCommandPool(self.vkDevice, command_pool_create, None)

		# create descriptor pool.
		# Our descriptor pool can only allocate a single storage buffer.
		descriptorPoolSize = VkDescriptorPoolSize(
			type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, # Do we need different types of pool?
			descriptorCount=1
		)
		descriptorPoolCreateInfo = VkDescriptorPoolCreateInfo(
			sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			maxSets=1,  # we only need to allocate one descriptor set from the pool.
			poolSizeCount=1,
			pPoolSizes=descriptorPoolSize
		)
		self.vkDescriptorPool = vkCreateDescriptorPool(self.vkDevice, descriptorPoolCreateInfo, None)

	
	def getFeatures(self):
	
		self.features   = vkGetPhysicalDeviceFeatures(self.physical_device)  
		self.properties = vkGetPhysicalDeviceProperties(self.physical_device)
		self.memoryProperties = vkGetPhysicalDeviceMemoryProperties(self.physical_device)
		return [self.features, self.properties, self.memoryProperties]
		
	def createShader(self, path, stage):
		return Shader(self.vkDevice, path, stage)
		
	def release(self):
		print("destroying command pool")
		vkDestroyCommandPool(self.vkDevice, self.vkCommandPool, None)
		
		print("destroying descriptor pool")
		vkDestroyDescriptorPool(self.vkDevice, self.vkDescriptorPool, None)
		
		print("destroying pipelines")
		for pipeline in self.pipelines: 
			pipeline.release()
		
		print("destroying device")
		vkDestroyDevice(self.vkDevice, None)
		

	
class DescriptorSet(PrintClass):
	def __init__(self, descriptorPool):
		PrintClass.__init__(self)
		self.descriptorPool = descriptorPool
		# Here we specify a descriptor set layout. This allows us to bind our descriptors to
		# resources in the shader.

		# Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
		# 0. This binds to
		#   layout(std140, binding = 0) buffer buf
		# in the compute shader.

		self.descriptorSetLayoutBinding = VkDescriptorSetLayoutBinding(
			binding=0,
			descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			descriptorCount=1,
			stageFlags=VK_SHADER_STAGE_COMPUTE_BIT
		)

		descriptorSetLayoutCreateInfo = VkDescriptorSetLayoutCreateInfo(
			sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			bindingCount=1,  # only a single binding in this descriptor set layout.
			pBindings=self.descriptorSetLayoutBinding
		)

		# Create the descriptor set layout.
		self.vkCreateDescriptorSetLayout = vkCreateDescriptorSetLayout(self.vkDevice, descriptorSetLayoutCreateInfo, None)
		self.children += [self.vkCreateDescriptorSetLayout]

		# So we will allocate a descriptor set here.
		descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
			sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			descriptorPool=self.vkDescriptorPool,
			descriptorSetCount=1,
			pSetLayouts=[self.vkCreateDescriptorSetLayout]
		)

		# allocate descriptor set.
		self.vkDescriptorSet = vkAllocateDescriptorSets(self.vkDevice, descriptorSetAllocateInfo)[0]
		self.children += [self.vkDescriptorSet]

		# Next, we need to connect our actual storage buffer with the descrptor.
		# We use vkUpdateDescriptorSets() to update the descriptor set.

		# Specify the buffer to bind to the descriptor.
		descriptorBufferInfo = VkDescriptorBufferInfo(
			buffer=self.vkBuffer,
			offset=0,
			range=self.vkBufferSize
		)

		writeDescriptorSet = VkWriteDescriptorSet(
			sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			dstSet=self.vkDescriptorSet,
			dstBinding=0,  # write to the first, and only binding.
			descriptorCount=1,
			descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			pBufferInfo=descriptorBufferInfo
		)

		# perform the update of the descriptor set.
		vkUpdateDescriptorSets(self.vkDevice, 1, [writeDescriptorSet], 0, None)

	def release():
		
		vkDestroyDescriptorSetLayout(self.device, self.descriptorSetLayout, None)
		vkDestroyDescriptorSet(self.device, self.descriptorSet, None)
