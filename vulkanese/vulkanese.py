import ctypes
import os
#import sdl2
#import sdl2.ext
import time
import json
from vulkan import *

# prepend local imports with .

if "vulkanese" in sys.modules:
    from pipelines import *
    from rasterpipeline import *
    from raytracepipeline import *
    from computepipeline import *
    from descriptor import *
    from device import *
else:
    from .pipelines import *
    from .rasterpipeline import *
    from .raytracepipeline import *
    from .computepipeline import *
    from .descriptor import *
    from .device import *

from PIL import Image as pilImage

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


class Instance(Sinode):
    def __init__(self, verbose=False):
        Sinode.__init__(self, None)
        self.verbose = verbose
        print("version number ")
        packedVersion = vkEnumerateInstanceVersion()
        # The variant is a 3-bit integer packed into bits 31-29.
        variant = (packedVersion >> 29) & 0x07
        # The major version is a 7-bit integer packed into bits 28-22.
        major = (packedVersion >> 22) & 0x7F
        # The minor version number is a 10-bit integer packed into bits 21-12.
        minor = (packedVersion >> 12) & 0x3FF
        # The patch version number is a 12-bit integer packed into bits 11-0.
        patch = (packedVersion >> 0) & 0xFFF
        if self.verbose:
            print("Variant : " + str(variant))
            print("Major   : " + str(major))
            print("Minor   : " + str(minor))
            print("Patch   : " + str(patch))

        # ----------
        # Create instance
        appInfo = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Hello Triangle",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_MAKE_VERSION(1, 3, 0),
        )

        extensions = vkEnumerateInstanceExtensionProperties(None)
        extensions = [e.extensionName for e in extensions]
        # print("available extensions: ")
        # for e in extensions:
        #    print("    " + e)

        self.layers = vkEnumerateInstanceLayerProperties()
        self.layers = [l.layerName for l in self.layers]
        # print("available layers:")
        # for l in self.layers:
        #    print("    " + l)

        if self.verbose:
            print("Available layers " + json.dumps(self.layers, indent=2))

        if "VK_LAYER_KHRONOS_validation" in self.layers:
            self.layers = ["VK_LAYER_KHRONOS_validation"]
        elif "VK_LAYER_LUNARG_standard_validation" in self.layers:
            self.layers = ["VK_LAYER_LUNARG_standard_validation"]
        else:
            self.layers = []

        if self.verbose:
            print("applying layers " + str(self.layers))
        createInfo = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            flags=0,
            pApplicationInfo=appInfo,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            enabledLayerCount=len(self.layers),
            ppEnabledLayerNames=self.layers,
        )

        self.vkInstance = vkCreateInstance(createInfo, None)

        # ----------
        # Debug instance
        vkCreateDebugReportCallbackEXT = vkGetInstanceProcAddr(
            self.vkInstance, "vkCreateDebugReportCallbackEXT"
        )
        self.vkDestroyDebugReportCallbackEXT = vkGetInstanceProcAddr(
            self.vkInstance, "vkDestroyDebugReportCallbackEXT"
        )

        def debugCallback(*args):
            print("DEBUG: " + args[5] + " " + args[6])
            return 0

        debug_create = VkDebugReportCallbackCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=debugCallback,
        )
        self.callback = vkCreateDebugReportCallbackEXT(
            self.vkInstance, debug_create, None
        )

    def getDeviceList(self):
        self.physical_devices = vkEnumeratePhysicalDevices(self.vkInstance)
        print(type(self.physical_devices))
        devdict = {}
        for i, physical_device in enumerate(self.physical_devices):
            pProperties = vkGetPhysicalDeviceProperties(physical_device)
            memprops = Device.getMemoryProperties(physical_device)
            processorType = Device.getProcessorType(physical_device)
            devdict[pProperties.deviceName] = {"processorType": processorType, "memProperties" : memprops}
        return devdict

    def getDevice(self, deviceIndex):
        newDev = Device(self, deviceIndex)
        self.children += [newDev]
        return newDev

    def release(self):
        if self.verbose:
            print("destroying child devices")
        for d in self.children:
            d.release()
        if self.verbose:
            print("destroying debug etc")
        self.vkDestroyDebugReportCallbackEXT(self.vkInstance, self.callback, None)
        if self.verbose:
            print("destroying instance")
        vkDestroyInstance(self.vkInstance, None)
