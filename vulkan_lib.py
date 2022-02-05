
import ctypes
import os
import sdl2
import sdl2.ext
import time
import json
from vulkan import *
here = os.path.dirname(os.path.abspath(__file__))


class Instance:
	def __init__(self):
		
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
		self.activeDevices = []

	def getDeviceList(self):
		self.physical_devices            = vkEnumeratePhysicalDevices(self.vkInstance)
		self.physical_devices_features   = [vkGetPhysicalDeviceFeatures(physical_device)   for physical_device in self.physical_devices]
		self.physical_devices_properties = [vkGetPhysicalDeviceProperties(physical_device) for physical_device in self.physical_devices]
		return self.physical_devices_properties
			
		
	def getDevice(self, deviceIndex):
		newDev = Device(self,deviceIndex)
		self.activeDevices += [newDev]
		return newDev
		
	def release(self):
		print("destroying child devices")
		for d in self.activeDevices:
			d.release()
		print("destroying debug etc")
		self.vkDestroyDebugReportCallbackEXT(self.vkInstance, self.callback, None)
		print("destroying instance")
		vkDestroyInstance(self.vkInstance, None)
		
		
class Device:
	def __init__(self, instance, deviceIndex):
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
		self.commandPool = []
		
		
	def createCommandPool(self):
		newCommandPool = CommandPool(self)
		self.commandPool += [newCommandPool]
		return newCommandPool
		
	def createShader(self, path, stage):
		return Shader(self.vkDevice, path, stage)
		
	def draw_frame(self):
		try:
			for commandPool in self.commandPool:
				commandPool.draw_frame()

		except VkNotReady:
			print('not ready')
			return


	def release(self):
		print("destroying all command pools")
		for i in self.commandPool:
			print("destroying command pool i")
			i.release()
		print("destroying device")
		vkDestroyDevice(self.vkDevice, None)

class Surface:
	def getEvents(self):
		return sdl2.ext.get_events()
				
	def surface_xlib(self):
		print("Create Xlib surface")
		vkCreateXlibSurfaceKHR = vkGetInstanceProcAddr(self.vkInstance, "vkCreateXlibSurfaceKHR")
		surface_create = VkXlibSurfaceCreateInfoKHR(
			sType=VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
			dpy=self.wm_info.info.x11.display,
			window=self.wm_info.info.x11.window,
			flags=0)
		return vkCreateXlibSurfaceKHR(self.vkInstance, surface_create, None)

	def surface_wayland(self):
		print("Create wayland surface")
		vkCreateWaylandSurfaceKHR = vkGetInstanceProcAddr(self.vkInstance, "vkCreateWaylandSurfaceKHR")
		surface_create = VkWaylandSurfaceCreateInfoKHR(
			sType=VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
			display=self.wm_info.info.wl.display,
			surface=self.wm_info.info.wl.surface,
			flags=0)
		return vkCreateWaylandSurfaceKHR(self.vkInstance, surface_create, None)

	def surface_win32(self):
		def get_instance(hWnd):
			"""Hack needed before SDL 2.0.6"""
			from cffi import FFI
			_ffi = FFI()
			_ffi.cdef('long __stdcall GetWindowLongA(void* hWnd, int nIndex);')
			_lib = _ffi.dlopen('User32.dll')
			return _lib.GetWindowLongA(_ffi.cast('void*', hWnd), -6)

		print("Create windows surface")
		vkCreateWin32SurfaceKHR = vkGetInstanceProcAddr(self.vkInstance, "vkCreateWin32SurfaceKHR")
		surface_create = VkWin32SurfaceCreateInfoKHR(
			sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
			hinstance=get_instance(self.wm_info.info.win.window),
			hwnd=self.wm_info.info.win.window,
			flags=0)
		return vkCreateWin32SurfaceKHR(self.vkInstance, surface_create, None)

	def __init__(self, instance, device, commandBuffer):
		self.running = True
		self.commandBuffer = commandBuffer
		
		self.WIDTH = 400
		self.HEIGHT = 400
	
		self.instance = instance
		self.vkInstance = instance.vkInstance
		self.vkDevice   = device.vkDevice
		
		# ----------
		# Init sdl2
		if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
			raise Exception(sdl2.SDL_GetError())

		window = sdl2.SDL_CreateWindow(
			'test'.encode('ascii'),
			sdl2.SDL_WINDOWPOS_UNDEFINED,
			sdl2.SDL_WINDOWPOS_UNDEFINED, self.WIDTH, self.HEIGHT, 0)

		if not window:
			raise Exception(sdl2.SDL_GetError())

		self.wm_info = sdl2.SDL_SysWMinfo()
		sdl2.SDL_VERSION(self.wm_info.version)
		sdl2.SDL_GetWindowWMInfo(window, ctypes.byref(self.wm_info))
		
		extensions = ['VK_KHR_surface', 'VK_EXT_debug_report']
		if self.wm_info.subsystem == sdl2.SDL_SYSWM_WINDOWS:
			extensions.append('VK_KHR_win32_surface')
		elif self.wm_info.subsystem == sdl2.SDL_SYSWM_X11:
			extensions.append('VK_KHR_xlib_surface')
		elif self.wm_info.subsystem == sdl2.SDL_SYSWM_WAYLAND:
			extensions.append('VK_KHR_wayland_surface')
		else:
			raise Exception("Platform not supported")

		self.vkDestroySurfaceKHR = vkGetInstanceProcAddr(instance.vkInstance, "vkDestroySurfaceKHR")
		
		surface_mapping = {
			sdl2.SDL_SYSWM_X11: self.surface_xlib,
			sdl2.SDL_SYSWM_WAYLAND: self.surface_wayland,
			sdl2.SDL_SYSWM_WINDOWS: self.surface_win32
		}

		self.vkSurface = surface_mapping[self.wm_info.subsystem]()

		# ----------
		# Create swapchain
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vkGetInstanceProcAddr(instance.vkInstance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
		vkGetPhysicalDeviceSurfaceFormatsKHR = vkGetInstanceProcAddr(instance.vkInstance, "vkGetPhysicalDeviceSurfaceFormatsKHR")
		vkGetPhysicalDeviceSurfacePresentModesKHR = vkGetInstanceProcAddr(instance.vkInstance, "vkGetPhysicalDeviceSurfacePresentModesKHR")

		surface_capabilities  = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice=device.physical_device, surface=self.vkSurface)
		self.surface_formats  = vkGetPhysicalDeviceSurfaceFormatsKHR     (physicalDevice=device.physical_device, surface=self.vkSurface)
		surface_present_modes = vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice=device.physical_device, surface=self.vkSurface)

		if not self.surface_formats or not surface_present_modes:
			raise Exception('No available swapchain')

		def get_surface_format(formats):
			for f in formats:
				if f.format == VK_FORMAT_UNDEFINED:
					return  f
				if (f.format == VK_FORMAT_B8G8R8A8_UNORM and
					f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR):
					return f
			return formats[0]

		def get_surface_present_mode(present_modes):
			for p in present_modes:
				if p == VK_PRESENT_MODE_MAILBOX_KHR:
					return p
			return VK_PRESENT_MODE_FIFO_KHR;

		def get_swap_extent(capabilities):
			uint32_max = 4294967295
			if capabilities.currentExtent.width != uint32_max:
				return VkExtent2D(width=capabilities.currentExtent.width,
								  height=capabilities.currentExtent.height)

			width = max(
				capabilities.minImageExtent.width,
				min(capabilities.maxImageExtent.width, WIDTH))
			height = max(
				capabilities.minImageExtent.height,
				min(capabilities.maxImageExtent.height, HEIGHT))
			actualExtent = VkExtent2D(width=width, height=height);
			return actualExtent


		self.surface_format = get_surface_format(self.surface_formats)
		present_mode = get_surface_present_mode(surface_present_modes)
		self.extent = get_swap_extent(surface_capabilities)
		imageCount = surface_capabilities.minImageCount + 1;
		if surface_capabilities.maxImageCount > 0 and imageCount > surface_capabilities.maxImageCount:
			imageCount = surface_capabilities.maxImageCount

		print('selected format: %s' % self.surface_format.format)
		print('%s available swapchain present modes' % len(surface_present_modes))


		imageSharingMode = VK_SHARING_MODE_EXCLUSIVE
		queueFamilyIndexCount = 0
		pQueueFamilyIndices = None

		if device.queue_family_graphic_index != device.queue_family_present_index:
			imageSharingMode = VK_SHARING_MODE_CONCURRENT
			queueFamilyIndexCount = 2
			pQueueFamilyIndices = [device.queue_family_graphic_index, device.queue_family_present_index]

		vkCreateSwapchainKHR       = vkGetInstanceProcAddr(instance.vkInstance, 'vkCreateSwapchainKHR')
		self.vkDestroySwapchainKHR = vkGetInstanceProcAddr(instance.vkInstance, 'vkDestroySwapchainKHR')
		vkGetSwapchainImagesKHR    = vkGetInstanceProcAddr(instance.vkInstance, 'vkGetSwapchainImagesKHR')

		swapchain_create = VkSwapchainCreateInfoKHR(
			sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			flags=0,
			surface=self.vkSurface,
			minImageCount=imageCount,
			imageFormat=self.surface_format.format,
			imageColorSpace=self.surface_format.colorSpace,
			imageExtent=self.extent,
			imageArrayLayers=1,
			imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			imageSharingMode=imageSharingMode,
			queueFamilyIndexCount=queueFamilyIndexCount,
			pQueueFamilyIndices=pQueueFamilyIndices,
			compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			presentMode=present_mode,
			clipped=VK_TRUE,
			oldSwapchain=None,
			preTransform=surface_capabilities.currentTransform)

		self.swapchain = vkCreateSwapchainKHR(device.vkDevice, swapchain_create, None)
		swapchain_images = vkGetSwapchainImagesKHR(device.vkDevice, self.swapchain)

		# Create image view for each image in swapchain
		self.image_views = []
		for image in swapchain_images:
			subresourceRange = VkImageSubresourceRange(
				aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
				baseMipLevel=0,
				levelCount=1,
				baseArrayLayer=0,
				layerCount=1)

			components = VkComponentMapping(
				r=VK_COMPONENT_SWIZZLE_IDENTITY,
				g=VK_COMPONENT_SWIZZLE_IDENTITY,
				b=VK_COMPONENT_SWIZZLE_IDENTITY,
				a=VK_COMPONENT_SWIZZLE_IDENTITY)

			imageview_create = VkImageViewCreateInfo(
				sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				image=image,
				flags=0,
				viewType=VK_IMAGE_VIEW_TYPE_2D,
				format=self.surface_format.format,
				components=components,
				subresourceRange=subresourceRange)

			self.image_views.append(vkCreateImageView(device.vkDevice, imageview_create, None))


		print("%s images view created" % len(self.image_views))

	def release(self):
		print("destroying surface")
		self.vkDestroySwapchainKHR(self.vkDevice, self.swapchain, None)
		self.vkDestroySurfaceKHR(self.vkInstance, self.vkSurface, None)
		for i in self.image_views:
			vkDestroyImageView(self.vkDevice, i, None)
			
		
	
class CommandPool:
	def __init__(self, device):
		self.device = device
		self.vkDevice = device.vkDevice
		# Create command pools
		command_pool_create = VkCommandPoolCreateInfo(
			sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			queueFamilyIndex=self.device.queue_family_graphic_index,
			flags=0)

		self.vkCommandPool = vkCreateCommandPool(self.vkDevice, command_pool_create, None)

		self.commandBuffer = []
	
	def draw_frame(self):
		for commandBuffer in self.commandBuffer:
			commandBuffer.draw_frame()
	
	def createCommandBuffer(self, setupDict):
		newCommandBuffer = CommandBuffer(self, setupDict)
		self.commandBuffer += [newCommandBuffer]
		return newCommandBuffer
		
	def release(self):
		print("destroying command buffs")
		for b in self.commandBuffer:
			b.release()
		print("destroying command pool")
		vkDestroyCommandPool(self.vkDevice, self.vkCommandPool, None)
		
		
class CommandBuffer:
	def __init__(self, commandPool, setupDict):
		self.commandPool  = commandPool
		self.device       = commandPool.device
		self.vkDevice     = commandPool.vkDevice
		self.outputWidthPixels  = setupDict["outputWidthPixels"]
		self.outputHeightPixels = setupDict["outputHeightPixels"]
		
		print("GOOOOOO")
		print(setupDict["outputClass"])
		if "surface" in setupDict["outputClass"]:
			self.createSurface()
		self.createPipeline(setupDict["pipeline"])
	
		print("Creating buffers of size " + str(self.outputWidthPixels) + 
			", " + str(self.outputHeightPixels))
		# Create command buffers, one for each image in the triple-buffer (swapchain + framebuffer)
		self.command_buffers_create = VkCommandBufferAllocateInfo(
			sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			commandPool=self.commandPool.vkCommandPool,
			level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			commandBufferCount=len(self.surface.image_views))

		self.command_buffers = vkAllocateCommandBuffers(self.vkDevice, self.command_buffers_create)
		
		self.framebuffers = []
	
		# Create semaphore
		semaphore_create = VkSemaphoreCreateInfo(
			sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			flags=0)
		self.semaphore_image_available = vkCreateSemaphore(self.vkDevice, semaphore_create, None)
		self.semaphore_render_finished = vkCreateSemaphore(self.vkDevice, semaphore_create, None)

		self.vkAcquireNextImageKHR = vkGetInstanceProcAddr(self.device.instance.vkInstance, "vkAcquireNextImageKHR")
		self.vkQueuePresentKHR = vkGetInstanceProcAddr(self.device.instance.vkInstance, "vkQueuePresentKHR")

		wait_semaphores = [self.semaphore_image_available]
		wait_stages = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
		signal_semaphores = [self.semaphore_render_finished]

		self.submit_create = VkSubmitInfo(
			sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
			waitSemaphoreCount=len(wait_semaphores),
			pWaitSemaphores=wait_semaphores,
			pWaitDstStageMask=wait_stages,
			commandBufferCount=1,
			pCommandBuffers=[self.command_buffers[0]],
			signalSemaphoreCount=len(signal_semaphores),
			pSignalSemaphores=signal_semaphores)

		# optimization to avoid creating a new array each time
		self.submit_list = ffi.new('VkSubmitInfo[1]', [self.submit_create])

		self.present_create = VkPresentInfoKHR(
			sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			waitSemaphoreCount=1,
			pWaitSemaphores=signal_semaphores,
			swapchainCount=1,
			pSwapchains=[self.surface.swapchain],
			pImageIndices=[0],
			pResults=None)
			
		# Record command buffer
		for i, command_buffer in enumerate(self.command_buffers):
			print("recording command_buffer " + str(i))
			command_buffer_begin_create = VkCommandBufferBeginInfo(
				sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
				pInheritanceInfo=None)

			vkBeginCommandBuffer(command_buffer, command_buffer_begin_create)

			# Create Graphics render pass
			render_area = VkRect2D(offset=VkOffset2D(x=0, y=0),
								   extent=self.pipeline.extent)
			color = VkClearColorValue(float32=[0, 1, 0, 1])
			clear_value = VkClearValue(color=color)

			# Framebuffers creation
			attachments = [self.surface.image_views[i]]
			framebuffer_create = VkFramebufferCreateInfo(
				sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				flags=0,
				renderPass=self.pipeline.render_pass,
				attachmentCount=len(attachments),
				pAttachments=attachments,
				width=self.outputWidthPixels,
				height=self.outputHeightPixels,
				layers=1)
				
			thisFramebuffer = vkCreateFramebuffer(self.vkDevice, framebuffer_create, None)
			self.framebuffers.append(thisFramebuffer)
			
			self.render_pass_begin_create = VkRenderPassBeginInfo(
				sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				renderPass=self.pipeline.render_pass,
				framebuffer=thisFramebuffer,
				renderArea=render_area,
				clearValueCount=1,
				pClearValues=[clear_value])

			vkCmdBeginRenderPass(command_buffer, self.render_pass_begin_create, VK_SUBPASS_CONTENTS_INLINE)

			# Bind pipeline
			vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.vkPipeline)

			# Draw
			vkCmdDraw(command_buffer, 3, 1, 0, 0)

			# End
			vkCmdEndRenderPass(command_buffer)
			vkEndCommandBuffer(command_buffer)
			
	def getEvents(self):
		return self.surface.getEvents()
	
	def createSurface(self):
		newSurface   = Surface(self.device.instance, self.device, self)
		self.surface = newSurface
		return newSurface
		
	def createPipeline(self, setupDict):
		newPipeline   = Pipeline(self, setupDict)
		self.pipeline = newPipeline
		return newPipeline
		
		
	def release(self):
		print("destroying framebuffers")
		for f in self.framebuffers:
			print("destroying framebuffer f")
			vkDestroyFramebuffer(self.vkDevice, f, None)

		print("destroying semaphore")
		vkDestroySemaphore(self.vkDevice, self.semaphore_image_available, None)
		vkDestroySemaphore(self.vkDevice, self.semaphore_render_finished, None)
		
		self.surface.release()
		self.pipeline.release()
		

	def draw_frame(self):
		image_index = self.vkAcquireNextImageKHR(self.vkDevice, self.surface.swapchain, UINT64_MAX, self.semaphore_image_available, None)

		self.submit_create.pCommandBuffers[0] = self.command_buffers[image_index]
		vkQueueSubmit(self.device.graphic_queue, 1, self.submit_list, None)

		self.present_create.pImageIndices[0] = image_index
		self.vkQueuePresentKHR(self.device.presentation_queue, self.present_create)

		# Fix #55 but downgrade performance -1000FPS)
		vkQueueWaitIdle(self.device.presentation_queue)
		
		
class Pipeline:

	def __init__(self, command_buffer, setupDict):
		self.vkDevice = command_buffer.vkDevice
		self.command_buffer = command_buffer
		self.outputWidthPixels  = setupDict["outputWidthPixels"]
		self.outputHeightPixels = setupDict["outputHeightPixels"]
		
		# Create render pass
		color_attachement = VkAttachmentDescription(
			flags=0,
			format=command_buffer.surface.surface_format.format,
			samples=VK_SAMPLE_COUNT_1_BIT,
			loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
			storeOp=VK_ATTACHMENT_STORE_OP_STORE,
			stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
			initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
			finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)

		color_attachement_reference = VkAttachmentReference(
			attachment=0,
			layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)

		sub_pass = VkSubpassDescription(
			flags=0,
			pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
			inputAttachmentCount=0,
			pInputAttachments=None,
			pResolveAttachments=None,
			pDepthStencilAttachment=None,
			preserveAttachmentCount=0,
			pPreserveAttachments=None,
			colorAttachmentCount=1,
			pColorAttachments=[color_attachement_reference])

		dependency = VkSubpassDependency(
			dependencyFlags=0,
			srcSubpass=VK_SUBPASS_EXTERNAL,
			dstSubpass=0,
			srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			srcAccessMask=0,
			dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)

		render_pass_create = VkRenderPassCreateInfo(
			flags=0,
			sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			attachmentCount=1,
			pAttachments=[color_attachement],
			subpassCount=1,
			pSubpasses=[sub_pass],
			dependencyCount=1,
			pDependencies=[dependency])

		self.render_pass = vkCreateRenderPass(self.vkDevice, render_pass_create, None)
		
		# Add Shaders
		self.shaders = []
		for shaderDict in setupDict["shaders"]:
			self.shaders += [Shader(self.vkDevice, shaderDict)]
		
		# Create graphic pipeline
		vertex_input_create = VkPipelineVertexInputStateCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			flags=0,
			vertexBindingDescriptionCount=0,
			pVertexBindingDescriptions=None,
			vertexAttributeDescriptionCount=0,
			pVertexAttributeDescriptions=None)

		input_assembly_create = VkPipelineInputAssemblyStateCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			flags=0,
			topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			primitiveRestartEnable=VK_FALSE)
		viewport = VkViewport(
			x=0., y=0., width=float(self.outputWidthPixels), height=float(self.outputHeightPixels),
			minDepth=0., maxDepth=1.)

		scissor_offset = VkOffset2D(x=0, y=0)
		self.extent = VkExtent2D(width=self.outputWidthPixels,
						height=self.outputHeightPixels)
		scissor = VkRect2D(offset=scissor_offset, extent=self.extent)
		viewport_state_create = VkPipelineViewportStateCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			flags=0,
			viewportCount=1,
			pViewports=[viewport],
			scissorCount=1,
			pScissors=[scissor])

		rasterizer_create = VkPipelineRasterizationStateCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			flags=0,
			depthClampEnable=VK_FALSE,
			rasterizerDiscardEnable=VK_FALSE,
			polygonMode=VK_POLYGON_MODE_FILL,
			lineWidth=1,
			cullMode=VK_CULL_MODE_BACK_BIT,
			frontFace=VK_FRONT_FACE_CLOCKWISE,
			depthBiasEnable=VK_FALSE,
			depthBiasConstantFactor=0.,
			depthBiasClamp=0.,
			depthBiasSlopeFactor=0.)

		multisample_create = VkPipelineMultisampleStateCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			flags=0,
			sampleShadingEnable=VK_FALSE,
			rasterizationSamples=VK_SAMPLE_COUNT_1_BIT,
			minSampleShading=1,
			pSampleMask=None,
			alphaToCoverageEnable=VK_FALSE,
			alphaToOneEnable=VK_FALSE)

		color_blend_attachement = VkPipelineColorBlendAttachmentState(
			colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
			blendEnable=VK_FALSE,
			srcColorBlendFactor=VK_BLEND_FACTOR_ONE,
			dstColorBlendFactor=VK_BLEND_FACTOR_ZERO,
			colorBlendOp=VK_BLEND_OP_ADD,
			srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE,
			dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO,
			alphaBlendOp=VK_BLEND_OP_ADD)

		color_blend_create = VkPipelineColorBlendStateCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			flags=0,
			logicOpEnable=VK_FALSE,
			logicOp=VK_LOGIC_OP_COPY,
			attachmentCount=1,
			pAttachments=[color_blend_attachement],
			blendConstants=[0, 0, 0, 0])

		push_constant_ranges = VkPushConstantRange(
			stageFlags=0,
			offset=0,
			size=0)

		pipeline_layout_create = VkPipelineLayoutCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			flags=0,
			setLayoutCount=0,
			pSetLayouts=None,
			pushConstantRangeCount=0,
			pPushConstantRanges=[push_constant_ranges])

		self.pipeline_layout = vkCreatePipelineLayout(self.vkDevice, pipeline_layout_create, None)

		
		# Finally create graphic pipeline
		self.pipeline_create = VkGraphicsPipelineCreateInfo(
			sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			flags=0,
			stageCount=len(self.shaders),
			pStages=[s.shader_stage_create for s in self.shaders],
			pVertexInputState=vertex_input_create,
			pInputAssemblyState=input_assembly_create,
			pTessellationState=None,
			pViewportState=viewport_state_create,
			pRasterizationState=rasterizer_create,
			pMultisampleState=multisample_create,
			pDepthStencilState=None,
			pColorBlendState=color_blend_create,
			pDynamicState=None,
			layout=self.pipeline_layout,
			renderPass=self.render_pass,
			subpass=0,
			basePipelineHandle=None,
			basePipelineIndex=-1)

		pipelines = vkCreateGraphicsPipelines(self.vkDevice, None, 1, [self.pipeline_create], None)
		self.vkPipeline = pipelines[0]
	
	def release(self):
		print("destroying pipeline")
		for shader in self.shaders:
			shader.release()
		vkDestroyRenderPass(self.vkDevice, self.render_pass, None)
		vkDestroyPipeline(self.vkDevice, self.vkPipeline, None)
		vkDestroyPipelineLayout(self.vkDevice, self.pipeline_layout, None)
		
		
		
class Shader:
	def __init__(self, vkDevice, shaderDict):
		self.vkDevice = vkDevice
		self.outputWidthPixels  = shaderDict["outputWidthPixels"]
		self.outputHeightPixels = shaderDict["outputHeightPixels"]
		
		print("creating shader with description")
		print(json.dumps(shaderDict, indent=4))
		self.path = os.path.join(here, shaderDict["path"])

		with open(self.path, 'rb') as f:
			shader_spirv = f.read()

		# Create shader
		shader_create = VkShaderModuleCreateInfo(
			sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			flags=0,
			codeSize=len(shader_spirv),
			pCode=shader_spirv
		)

		self.shader_module = vkCreateShaderModule(vkDevice, shader_create, None)

		# Create shader stage
		self.shader_stage_create = VkPipelineShaderStageCreateInfo(
			sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			stage=eval(shaderDict["stage"]),
			module=self.shader_module,
			flags=0,
			pSpecializationInfo=None,
			pName='main')

	def release(self):
		print("destroying shader")
		vkDestroyShaderModule(self.vkDevice, self.shader_module, None)
		