import ctypes
import os
import time
import json
from sinode import *
from vulkan import *

# prepend local imports with .
try:
    from vulkanese import *
    from descriptor import *
except:
    from .vulkanese import *
    from .descriptor import *
from PIL import Image as pilImage

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


class Device(Sinode):
    def ctypes2dict(props, depth=0):
        outDict = dict()
        type = ffi.typeof(props)
        if type.kind == "primitive":
            return props
        elif type.kind == "struct":
            for f in type.fields:
                fieldName, fieldType = f
                if fieldType.type.kind == "primitive":
                    outDict[fieldName] = eval("props." + fieldName)
                else:
                    outDict[fieldName] = Device.ctypes2dict(
                        eval("props." + fieldName), depth + 1
                    )
            return outDict
        elif type.kind == "array":
            return [Device.ctypes2dict(p, depth + 1) for p in props]
        else:
            print(" " * depth + type.kind)
            print(" " * depth + dir(type))
            die
        
    def __init__(self, instance, deviceIndex):
        Sinode.__init__(self, instance)
        self.instance = instance
        self.vkInstance = instance.vkInstance
        self.deviceIndex = deviceIndex

        print("initializing device " + str(deviceIndex))
        self.physical_device = vkEnumeratePhysicalDevices(self.instance.vkInstance)[
            deviceIndex
        ]

        self.memoryProperties = Device.getMemoryProperties(self.physical_device)
        
        print("getting features list")

        # vkGetPhysicalDeviceFeatures2 = vkGetInstanceProcAddr(
        #    self.vkInstance, "vkGetPhysicalDeviceFeatures2KHR"
        # )
        # vkGetPhysicalDeviceProperties2 = vkGetInstanceProcAddr(
        #    self.vkInstance, "vkGetPhysicalDeviceProperties2KHR"
        # )

        self.pFeatures = vkGetPhysicalDeviceFeatures(self.physical_device)

        # self.pFeatures2 = vkGetPhysicalDeviceFeatures2(self.physical_device)
        # print("pFeatures2")
        # print(self.pFeatures2)

        self.propertiesDict = Device.getLimits(self.physical_device)

        # self.pProperties2 = vkGetPhysicalDeviceProperties2(self.physical_device)
        # print("pProperties2")
        # print(self.pProperties2)

        print("Select queue family")
        # ----------
        # Select queue family
        vkGetPhysicalDeviceSurfaceSupportKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkGetPhysicalDeviceSurfaceSupportKHR"
        )

        # queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self.physical_device)
        queue_families = vkGetPhysicalDeviceQueueFamilyProperties(
            physicalDevice=self.physical_device
        )

        # print("%s available queue family" % len(queue_families))

        self.queue_family_graphic_index = -1
        self.queue_family_present_index = -1

        for i, queue_family in enumerate(queue_families):
            # Currently, we set present index like graphic index
            # support_present = vkGetPhysicalDeviceSurfaceSupportKHR(
            # 	physicalDevice=physical_device,
            # 	queueFamilyIndex=i,
            # 	surface=sdl_surface_inst.surface)
            if (
                queue_family.queueCount > 0
                and queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT
            ):
                self.queue_family_graphic_index = i
                self.queue_family_present_index = i
            # if queue_family.queueCount > 0 and support_present:
            #     self.queue_family_present_index = i

        print(
            "indice of selected queue families, graphic: %s, presentation: %s\n"
            % (self.queue_family_graphic_index, self.queue_family_present_index)
        )

        self.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE
        self.queueFamilyIndexCount = 0
        self.pQueueFamilyIndices = None

        if self.queue_family_graphic_index != self.queue_family_present_index:
            self.imageSharingMode = VK_SHARING_MODE_CONCURRENT
            self.queueFamilyIndexCount = 2
            self.pQueueFamilyIndices = [
                device.queue_family_graphic_index,
                device.queue_family_present_index,
            ]

        # ----------
        # Create logical device and queues
        extensions = vkEnumerateDeviceExtensionProperties(
            physicalDevice=self.physical_device, pLayerName=None
        )
        extensions = [e.extensionName for e in extensions]
        # print("available device extensions: %s\n" % extensions)

        # only use the extensions necessary
        extensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]

        queues_create = [
            VkDeviceQueueCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=i,
                queueCount=1,
                pQueuePriorities=[1],
                flags=0,
            )
            for i in {self.queue_family_graphic_index, self.queue_family_present_index}
        ]
        # print(self.pFeatures.pNext)
        # die
        self.device_create = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            # pNext=self.pFeatures2,
            pNext=None,
            pQueueCreateInfos=queues_create,
            queueCreateInfoCount=len(queues_create),
            pEnabledFeatures=self.pFeatures,  # NEED TO PUT PFEATURES2 or something
            flags=0,
            enabledLayerCount=len(self.instance.layers),
            ppEnabledLayerNames=self.instance.layers,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
        )

        self.vkDevice = vkCreateDevice(
            physicalDevice=self.physical_device,
            pCreateInfo=self.device_create,
            pAllocator=None,
        )

        self.graphic_queue = vkGetDeviceQueue(
            device=self.vkDevice,
            queueFamilyIndex=self.queue_family_graphic_index,
            queueIndex=0,
        )
        self.presentation_queue = vkGetDeviceQueue(
            device=self.vkDevice,
            queueFamilyIndex=self.queue_family_present_index,
            queueIndex=0,
        )
        self.compute_queue = vkGetDeviceQueue(
            device=self.vkDevice,
            queueFamilyIndex=self.getComputeQueueFamilyIndex(),
            queueIndex=0,
        )

        # Create command pool
        command_pool_create = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family_graphic_index,
            flags=0,
        )

        self.vkCommandPool = vkCreateCommandPool(
            self.vkDevice, command_pool_create, None
        )

        self.descriptorPool = DescriptorPool(self)
        self.children += [self.descriptorPool]

    def getMemoryProperties(physical_device):
        print("getting memory properties")
        memoryProperties = vkGetPhysicalDeviceMemoryProperties(
            physical_device
        )
        memoryPropertiesPre = Device.ctypes2dict(memoryProperties)

        # the following is complicated only because C/C++ is so basic
        # shorten the lists to however many there are
        memoryTypes = memoryPropertiesPre["memoryTypes"][
            : memoryPropertiesPre["memoryTypeCount"]
        ]
        print(memoryTypes)
        memoryHeaps = memoryPropertiesPre["memoryHeaps"][
            : memoryPropertiesPre["memoryHeapCount"]
        ]
        print(memoryHeaps)

        for mt in memoryTypes:
            mt["propertyFlagsString"] = []
        for mh in memoryHeaps:
            mh["flagsString"] = []

        heaps = []
        types = []
        # (this is so dumb)
        # get all keys that start with VK_MEMORY_PROPERTY_
        for k, v in globals().items():
            if (
                k.startswith("VK_MEMORY_PROPERTY_")
                and v is not None
                and not k.endswith("MAX_ENUM")
            ):
                for mt in memoryTypes:
                    if mt["propertyFlags"] & v:
                        mt["propertyFlagsString"] += [k]

        for k, v in globals().items():
            if (
                k.startswith("VK_MEMORY_HEAP_")
                and v is not None
                and not k.endswith("MAX_ENUM")
            ):
                for mh in memoryHeaps:
                    if mh["flags"] & v:
                        mh["flagsString"] += [k]
                        
        return {"memoryTypes":memoryTypes, "memoryHeaps":memoryHeaps}

    def nameSubdicts(self, key, value):
        if type(value) is dict:
            retdict = {}
            retdict["name"] = key
            for k, v in value.items():
                retdict[k] = self.nameSubdicts(k, v)
            return retdict
        else:
            return value

    # Returns the index of a queue family that supports compute operations.
    def getComputeQueueFamilyIndex(self):
        # Retrieve all queue families.
        queueFamilies = vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)

        # Now find a family that supports compute.
        for i, props in enumerate(queueFamilies):
            if props.queueCount > 0 and props.queueFlags & VK_QUEUE_COMPUTE_BIT:
                # found a queue with compute. We're done!
                return i

        return -1

    def getProcessorType(physical_device):
        
        pProperties = vkGetPhysicalDeviceProperties(physical_device)
        deviceType = pProperties.deviceType
        if deviceType == 0:
            deviceTypeStr = "VK_PHYSICAL_DEVICE_TYPE_OTHER"
        elif deviceType == 1:
            deviceTypeStr = "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"
        elif deviceType == 2:
            deviceTypeStr = "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
        elif deviceType == 3:
            deviceTypeStr = "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU"
        elif deviceType == 4:
            deviceTypeStr = "VK_PHYSICAL_DEVICE_TYPE_CPU"
        return deviceTypeStr
        
    def getLimits(physical_device):
        pProperties = vkGetPhysicalDeviceProperties(physical_device)
        print("Device Name: " + pProperties.deviceName)

        devPropsDict = {}
        
        devPropsDict["deviceType"] = Device.getProcessorType(physical_device)
        
        limitsDict = {}
        type = ffi.typeof(pProperties.limits)
        for fieldName, fieldType in type.fields:
            if fieldType.type.kind == "primitive":
                fieldValue = eval("pProperties.limits." + fieldName)
            else:
                lfieldValue = str(eval("pProperties.limits." + fieldName))

            limitsDict[fieldName] = fieldValue

        return limitsDict

    def getFeatures(self):

        self.features = vkGetPhysicalDeviceFeatures(self.physical_device)
        self.properties = vkGetPhysicalDeviceProperties(self.physical_device)
        self.memoryProperties = vkGetPhysicalDeviceMemoryProperties(
            self.physical_device
        )
        return [self.features, self.properties, self.memoryProperties]

    def createShader(self, path, stage):
        return Shader(self.vkDevice, path, stage)

    def release(self):

        print("destroying children")
        for child in self.children:
            child.release()

        print("destroying command pool")
        vkDestroyCommandPool(self.vkDevice, self.vkCommandPool, None)

        print("destroying device")
        vkDestroyDevice(self.vkDevice, None)
