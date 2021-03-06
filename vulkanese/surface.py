from vulkan import *
from vutil import *
import sdl2
import sdl2.ext
import ctypes

class Surface(Sinode):
	def getEvents(self):
		return sdl2.ext.get_events()
	
	def get_surface_format(self):
		for f in self.formats:
			print("FORMAT")
			print(f.format)
			if f.format == VK_FORMAT_UNDEFINED:
				print("FORMAT UNDEFINED")
				return  f
			if (f.format == VK_FORMAT_B8G8R8A8_UNORM and
				f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR):
				print("FORMAT VK_FORMAT_B8G8R8A8_UNORM")
				return f
		return self.formats[0]

	def get_surface_present_mode(self):
		for p in self.present_modes:
			if p == VK_PRESENT_MODE_MAILBOX_KHR:
				return p
		return VK_PRESENT_MODE_FIFO_KHR;

	def get_swap_extent(self):
		uint32_max = 4294967295
		if self.capabilities.currentExtent.width != uint32_max:
			return VkExtent2D(width=self.capabilities.currentExtent.width,
							  height=self.capabilities.currentExtent.height)

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

	def __init__(self, instance, device, pipeline):
		Sinode.__init__(self, device)
		self.running = True
		self.pipeline = pipeline
		
		self.WIDTH = pipeline.setupDict["outputWidthPixels"]
		self.HEIGHT = pipeline.setupDict["outputHeightPixels"]
		self.extent = VkExtent2D(width=self.WIDTH,
							  height=self.HEIGHT )
	
		self.instance = instance
		self.vkInstance = instance.vkInstance
		self.vkDevice   = device.vkDevice
		self.device     = device
		
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
		self.children += [self.vkSurface]

		# ----------
		# Create swapchain
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vkGetInstanceProcAddr(instance.vkInstance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
		vkGetPhysicalDeviceSurfaceFormatsKHR      = vkGetInstanceProcAddr(instance.vkInstance, "vkGetPhysicalDeviceSurfaceFormatsKHR")
		vkGetPhysicalDeviceSurfacePresentModesKHR = vkGetInstanceProcAddr(instance.vkInstance, "vkGetPhysicalDeviceSurfacePresentModesKHR")

		self.capabilities  = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice=device.physical_device, surface=self.vkSurface)
		self.formats       = vkGetPhysicalDeviceSurfaceFormatsKHR     (physicalDevice=device.physical_device, surface=self.vkSurface)
		self.present_modes = vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice=device.physical_device, surface=self.vkSurface)

		if not self.formats or not self.present_modes:
			raise Exception('No available swapchain')

			width = max(
				capabilities.minImageExtent.width,
				min(capabilities.maxImageExtent.width, WIDTH))
			height = max(
				capabilities.minImageExtent.height,
				min(capabilities.maxImageExtent.height, HEIGHT))
			actualExtent = VkExtent2D(width=width, height=height);
			return actualExtent


		self.surface_format = self.get_surface_format()
		present_mode = self.get_surface_present_mode()
		self.extent  = self.get_swap_extent()
		imageCount   = self.capabilities.minImageCount + 1;
		if self.capabilities.maxImageCount > 0 and imageCount > self.capabilities.maxImageCount:
			imageCount = self.capabilities.maxImageCount

		print('selected format: %s' % self.surface_format.format)
		print('selected colorspace: %s' % self.surface_format.colorSpace)
		print('%s available swapchain present modes' % len(self.present_modes))
		print("image count " + str(imageCount))
		self.vkDestroySwapchainKHR = vkGetInstanceProcAddr(instance.vkInstance, 'vkDestroySwapchainKHR')
		
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
			imageSharingMode=device.imageSharingMode,
			queueFamilyIndexCount=device.queueFamilyIndexCount,
			pQueueFamilyIndices=device.pQueueFamilyIndices,
			compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			presentMode=present_mode,
			clipped=VK_TRUE,
			oldSwapchain=None,
			preTransform=self.capabilities.currentTransform)

		vkCreateSwapchainKHR    = vkGetInstanceProcAddr(instance.vkInstance, 'vkCreateSwapchainKHR')
		self.swapchain          = vkCreateSwapchainKHR(device.vkDevice, swapchain_create, None)
		vkGetSwapchainImagesKHR = vkGetInstanceProcAddr(instance.vkInstance, 'vkGetSwapchainImagesKHR')
		self.vkSwapchainImages  = vkGetSwapchainImagesKHR(device.vkDevice, self.swapchain)
		print("swapchain images " + str(self.vkSwapchainImages))
		self.children += [self.vkSwapchainImages]

		# preesentation creator
		self.present_create = VkPresentInfoKHR(
			sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			waitSemaphoreCount=1,
			pWaitSemaphores=self.pipeline.signal_semaphores,
			swapchainCount=1,
			pSwapchains=[self.swapchain],
			pImageIndices=[0],
			pResults=None)

	def release(self):
		print("destroying surface")
		self.vkDestroySwapchainKHR(self.vkDevice, self.swapchain, None)
		self.vkDestroySurfaceKHR(self.vkInstance, self.vkSurface, None)
			
		
	