import ctypes
import os

# import sdl2
# import sdl2.ext
import time
import json
import vulkan as vk
import sinode

from . import shader
from . import descriptor
from . import device
from . import buffer

vulkanesehome = os.path.dirname(os.path.abspath(__file__))


class Instance(sinode.Sinode):
    def __init__(self, verbose=False):
        sinode.Sinode.__init__(self, None)
        self.verbose = verbose
        self.debug("version number ")
        packedVersion = vk.vkEnumerateInstanceVersion()
        # The variant is a 3-bit integer packed into bits 31-29.
        variant = (packedVersion >> 29) & 0x07
        # The major version is a 7-bit integer packed into bits 28-22.
        major = (packedVersion >> 22) & 0x7F
        # The minor version number is a 10-bit integer packed into bits 21-12.
        minor = (packedVersion >> 12) & 0x3FF
        # The patch version number is a 12-bit integer packed into bits 11-0.
        patch = (packedVersion >> 0) & 0xFFF

        self.debug("Variant : " + str(variant))
        self.debug("Major   : " + str(major))
        self.debug("Minor   : " + str(minor))
        self.debug("Patch   : " + str(patch))

        # ----------
        # Create instance
        appInfo = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Hello Triangle",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 3, 0),
        )

        extensions = vk.vkEnumerateInstanceExtensionProperties(None)
        extensions = [e.extensionName for e in extensions]
        # self.debug("available extensions: ")
        # for e in extensions:
        #    self.debug("    " + e)

        self.layers = vk.vkEnumerateInstanceLayerProperties()
        self.layers = [l.layerName for l in self.layers]
        # self.debug("available layers:")
        # for l in self.layers:
        #    self.debug("    " + l)

        if self.verbose:
            self.debug("Available layers " + json.dumps(self.layers, indent=2))

        if "VK_LAYER_KHRONOS_validation" in self.layers:
            self.layers = ["VK_LAYER_KHRONOS_validation"]
        elif "VK_LAYER_LUNARG_standard_validation" in self.layers:
            self.layers = ["VK_LAYER_LUNARG_standard_validation"]
        else:
            self.layers = []

        if self.verbose:
            self.debug("applying layers " + str(self.layers))
        createInfo = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            flags=0,
            pApplicationInfo=appInfo,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            enabledLayerCount=len(self.layers),
            ppEnabledLayerNames=self.layers,
        )

        self.vkInstance = vk.vkCreateInstance(createInfo, None)

        # ----------
        # Debug instance
        vkCreateDebugReportCallbackEXT = vk.vkGetInstanceProcAddr(
            self.vkInstance, "vkCreateDebugReportCallbackEXT"
        )
        self.vkDestroyDebugReportCallbackEXT = vk.vkGetInstanceProcAddr(
            self.vkInstance, "vkDestroyDebugReportCallbackEXT"
        )

        def debugCallback(*args):
            self.debug("DEBUG: " + args[5] + " " + args[6])
            return 0

        debug_create = vk.VkDebugReportCallbackCreateInfoEXT(
            sType=vk.VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=vk.VK_DEBUG_REPORT_ERROR_BIT_EXT | vk.VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=debugCallback,
        )
        self.callback = vkCreateDebugReportCallbackEXT(
            self.vkInstance, debug_create, None
        )

    def debug(self, *args):
        if self.verbose:
            print(args)

    def getDeviceList(self):
        self.physical_devices = vk.vkEnumeratePhysicalDevices(self.vkInstance)
        self.debug(type(self.physical_devices))
        devdict = {}
        for i, physical_device in enumerate(self.physical_devices):
            # subgroupProperties = VkPhysicalDeviceSubgroupProperties()
            pProperties = vk.vkGetPhysicalDeviceProperties(physical_device)
            device = self.getDevice(i)
            memprops = Device.getMemoryProperties(device)
            processorType = Device.getProcessorType(physical_device)
            limits = Device.getLimits(device)
            devdict[pProperties.deviceName] = {
                "processorType": processorType,
                "memProperties": memprops,
                "limits": limits,
            }
        return devdict

    def getDevice(self, deviceIndex):
        newDev = device.Device(self, deviceIndex)
        self.children += [newDev]
        return newDev

    def release(self):
        if self.verbose:
            self.debug("destroying child devices")
        for d in self.children:
            d.release()
        if self.verbose:
            self.debug("destroying debug etc")
        self.vkDestroyDebugReportCallbackEXT(self.vkInstance, self.callback, None)
        if self.verbose:
            self.debug("destroying instance")
        vkDestroyInstance(self.vkInstance, None)
