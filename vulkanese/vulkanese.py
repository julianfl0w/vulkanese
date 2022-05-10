
import ctypes
import os
import sdl2
import sdl2.ext
import time
import json
from vutil import *
from vulkan import *
from pipelines import *
from raytracepipeline import *
from descriptor import *
from PIL import Image as pilImage
import jvulkan

here = os.path.dirname(os.path.abspath(__file__))
def getVulkanesePath():
	return here

class Instance(Sinode):
	def __init__(self):
		Sinode.__init__(self, None)
		
		jlog("version number ")
		packedVersion = vkEnumerateInstanceVersion()
		#The variant is a 3-bit integer packed into bits 31-29.
		variant = (packedVersion >> 29) & 0x07
		#The major version is a 7-bit integer packed into bits 28-22.
		major = (packedVersion >> 22) & 0x7F
		#The minor version number is a 10-bit integer packed into bits 21-12.
		minor = (packedVersion >> 12) & 0x3FF
		#The patch version number is a 12-bit integer packed into bits 11-0.
		patch = (packedVersion >>  0) & 0xFFF
		jlog("Variant : " + str(variant))
		jlog("Major   : " + str(major))
		jlog("Minor   : " + str(minor))
		jlog("Patch   : " + str(patch))
		
		# ----------
		# Create instance
		appInfo = VkApplicationInfo(
			sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
			pApplicationName="Hello Triangle",
			applicationVersion=VK_MAKE_VERSION(1, 0, 0),
			pEngineName="No Engine",
			engineVersion=VK_MAKE_VERSION(1, 0, 0),
			apiVersion=VK_API_VERSION_1_0)
			
		appInfo_ctypes = jvulkan.VkApplicationInfo(
			sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
			pApplicationName=cast(create_string_buffer(b"Hello Triangle"),c_char_p),
			applicationVersion=VK_MAKE_VERSION(1, 0, 0),
			pEngineName=cast(create_string_buffer(b"No Engine"),c_char_p),
			engineVersion=VK_MAKE_VERSION(1, 0, 0),
			apiVersion=VK_API_VERSION_1_0)

		#extensions = vkEnumerateInstanceExtensionProperties(None)
		#extensions = [e.extensionName for e in extensions]
		FUCKJESUS = (jvulkan.VkExtensionProperties*1200)()
		print(type(FUCKJESUS))
		print(jvulkan.jvulkanLib.vkEnumerateInstanceExtensionProperties.argtypes)
		extensions_ctypes = jvulkan.vkEnumerateInstanceExtensionProperties({"pLayerName":None, "pProperties":FUCKJESUS})
		print(extensions_ctypes)
		jlog(extensions_ctypes["pPropertyCount"][0])
		print(extensions_ctypes["pProperties"])
		jlog(cast(extensions_ctypes["pProperties"], POINTER(VkExtensionProperties())))
		extensions_ctypes = [e.extensionName for e in extensions_ctypes["pProperties"]]
		jlog("available extensions: ")
		for e in extensions:
			jlog("    " + e)

		self.layers = vkEnumerateInstanceLayerProperties()
		self.layers = [l.layerName for l in self.layers]
		jlog("available layers:")
		for l in self.layers:
			jlog("    " + l)

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
			ppEnabledLayerNames=self.layers
			)

		createInfo_ctypes = jvulkan.VkInstanceCreateInfo(
			sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			flags=0,
			pApplicationInfo=appInfo_ctypes,
			enabledExtensionCount=len(extensions),
			ppEnabledExtensionNames=extensions,
			enabledLayerCount=len(self.layers),
			ppEnabledLayerNames=self.layers
			)

		self.vkInstance = jvulkan.vkCreateInstance(createInfo, None)
		
		# ----------
		# Debug instance
		vkCreateDebugReportCallbackEXT = vkGetInstanceProcAddr(
			self.vkInstance,
			"vkCreateDebugReportCallbackEXT")
		self.vkDestroyDebugReportCallbackEXT = vkGetInstanceProcAddr(
			self.vkInstance,
			"vkDestroyDebugReportCallbackEXT")

		def debugCallback(*args):
			jlog('DEBUG: ' + args[5] + ' ' + args[6])
			return 0

		debug_create = VkDebugReportCallbackCreateInfoEXT(
			sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
			flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
			pfnCallback=debugCallback)
		self.callback = vkCreateDebugReportCallbackEXT(self.vkInstance, debug_create, None)

	def getDeviceList(self):
		self.physical_devices            = vkEnumeratePhysicalDevices(self.vkInstance)
		return self.physical_devices
			
		
	def getDevice(self, deviceIndex):
		newDev = Device(self,deviceIndex)
		self.children += [newDev]
		return newDev
		
	def release(self):
		jlog("destroying child devices")
		for d in self.children:
			d.release()
		jlog("destroying debug etc")
		self.vkDestroyDebugReportCallbackEXT(self.vkInstance, self.callback, None)
		jlog("destroying instance")
		vkDestroyInstance(self.vkInstance, None)
		
class Device(Sinode):
	def nameSubdicts(self, key, value):
		if type(value) is dict:
			retdict = {}
			retdict["name"] = key
			for k, v in value.items():
				retdict[k] = self.nameSubdicts(k, v)
			return retdict
		else:
			return value
				
	def applyLayout(self, setupDict):
		self.setupDict = self.nameSubdicts("root", setupDict)
		jlog(json.dumps(self.setupDict, indent=2))
		self.pipelines = []
		for pipelineName, pipelineDict in self.setupDict.items():
			if pipelineDict == "root":
				continue
			if pipelineDict["class"] == "raster":
				self.pipelines += [RasterPipeline(self, pipelineDict)]
			elif pipelineDict["class"] == "compute":
				self.pipelines += [ComputePipeline(self, pipelineDict)]
			else:
				self.pipelines += [RaytracePipeline(self, pipelineDict)]
		self.descriptorPool.finalize()
		self.children += self.pipelines
		return self.pipelines
		
	def applyLayoutFile(self, filename):
		
		with open(filename, 'r') as f:
			setupDict = json.loads(f.read())

		# apply setup to device
		jlog("Applying the following layout:")
		jlog(json.dumps(setupDict, indent = 4))
		jlog("")
		return self.applyLayout(setupDict)

	
	def __init__(self, instance, deviceIndex):
		Sinode.__init__(self, instance)
		self.instance = instance
		self.vkInstance = instance.vkInstance
		self.deviceIndex = deviceIndex
		
		jlog("initializing device " + str(deviceIndex))
		self.physical_device = vkEnumeratePhysicalDevices(self.instance.vkInstance)[deviceIndex]
		self.physical_device_ctypes = jvulkan.vkEnumeratePhysicalDevices(self.instance.vkInstance)[deviceIndex]
		
		jlog("getting features list")
		
		vkGetPhysicalDeviceFeatures2   = vkGetInstanceProcAddr(self.vkInstance, 'vkGetPhysicalDeviceFeatures2KHR')
		vkGetPhysicalDeviceProperties2 = vkGetInstanceProcAddr(self.vkInstance, 'vkGetPhysicalDeviceProperties2KHR')

		self.pFeatures    = vkGetPhysicalDeviceFeatures (self.physical_device)
		jlog("pFeatures")
		jlog([self.pFeatures])
		
		self.pFeatures2   = jvulkan.vkGetPhysicalDeviceFeatures2(self.physical_device)
		jlog("pFeatures2")
		jlog(self.pFeatures2)
		
		self.pProperties  = vkGetPhysicalDeviceProperties (self.physical_device)
		jlog("pProperties")
		jlog(self.pProperties)
		self.pProperties2 = vkGetPhysicalDeviceProperties2(self.physical_device)
		jlog("pProperties2")
		jlog(self.pProperties2)
		
		
		jlog("Select queue family")
		# ----------
		# Select queue family
		vkGetPhysicalDeviceSurfaceSupportKHR = vkGetInstanceProcAddr(
			self.instance.vkInstance, 'vkGetPhysicalDeviceSurfaceSupportKHR')
		
		#queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self.physical_device)
		queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self.physical_device)
		
		jlog("%s available queue family" % len(queue_families))

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

		jlog("indice of selected queue families, graphic: %s, presentation: %s\n" % (
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
		jlog("available device extensions: %s\n" % extensions)

		#only use the extensions necessary
		extensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]

		queues_create = [VkDeviceQueueCreateInfo(sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
												 queueFamilyIndex=i,
												 queueCount=1,
												 pQueuePriorities=[1],
												 flags=0)
						 for i in {self.queue_family_graphic_index,
								   self.queue_family_present_index}]
		#jlog(self.pFeatures.pNext)
		#die
		
		self.deviceAddressFeatures = jvulkan.VkPhysicalDeviceBufferDeviceAddressFeatures(
			sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
			pNext = None,
			bufferDeviceAddress = True,
			bufferDeviceAddressCaptureReplay = False,
			bufferDeviceAddressMultiDevice = False
			)
		jlog(self.pFeatures2.pNext)
		self.pFeatures2.pNext = cast(self.deviceAddressFeatures, c_void_p)


		self.device_create = VkDeviceCreateInfo(
			sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			pNext = self.pFeatures2,
			pQueueCreateInfos    =queues_create,
			queueCreateInfoCount =len(queues_create),
			pEnabledFeatures     =self.pFeatures, # NEED TO PUT PFEATURES2 or something
			flags                =0,
			enabledLayerCount    =len(self.instance.layers),
			ppEnabledLayerNames  =self.instance.layers,
			enabledExtensionCount=len(extensions),
			ppEnabledExtensionNames=extensions
		)

		self.vkDevice = vkCreateDevice(
			physicalDevice = self.physical_device, 
			pCreateInfo    = self.device_create, 
			pAllocator     = None)
		
		self.graphic_queue = vkGetDeviceQueue(
			device=self.vkDevice,
			queueFamilyIndex=self.queue_family_graphic_index,
			queueIndex=0)
		self.presentation_queue = vkGetDeviceQueue(
			device=self.vkDevice,
			queueFamilyIndex=self.queue_family_present_index,
			queueIndex=0)
	
		jlog("Logical device and graphic queue successfully created\n")
		
		# Create command pool
		command_pool_create = VkCommandPoolCreateInfo(
			sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			queueFamilyIndex=self.queue_family_graphic_index,
			flags=0)

		self.vkCommandPool = vkCreateCommandPool(self.vkDevice, command_pool_create, None)

		self.descriptorPool = DescriptorPool(self)
		self.children += [self.descriptorPool]

	def getBinding(self, buffer, bindName):
		return self.descriptorPool.getBinding(buffer, bindName)
		
	def getFeatures(self):
	
		self.features   = vkGetPhysicalDeviceFeatures(self.physical_device)  
		self.properties = vkGetPhysicalDeviceProperties(self.physical_device)
		self.memoryProperties = vkGetPhysicalDeviceMemoryProperties(self.physical_device)
		return [self.features, self.properties, self.memoryProperties]
		
	def createShader(self, path, stage):
		return Shader(self.vkDevice, path, stage)
		
	def release(self):
		
		jlog("destroying children")
		for child in self.children: 
			child.release()
		
		jlog("destroying command pool")
		vkDestroyCommandPool(self.vkDevice, self.vkCommandPool, None)
		
		jlog("destroying device")
		vkDestroyDevice(self.vkDevice, None)
		