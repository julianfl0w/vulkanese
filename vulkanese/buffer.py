import json
from . import sinode
import os

import vulkan as vk
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def glsltype2python(glsltype):
    if glsltype == "float":
        return np.float32
    if glsltype == "float32_t":
        return np.float32
    elif glsltype == "float64_t":
        return np.float64
    elif glsltype == "int":
        return np.int32
    elif glsltype == "uint":
        return np.uint32
    elif "vec" in glsltype:
        return np.float32
    else:
        self.device.instance.debug("type")
        self.device.instance.debug(glsltype)
        die

def glsltype2bytesize(glsltype):
    if glsltype == "float":
        return 4
    elif glsltype == "float32_t":
        return 4
    elif glsltype == "float64_t":
        return 8
    elif glsltype == "int":
        return 4
    elif glsltype == "uint":
        return 4
    elif glsltype == "vec2":
        # return 12
        return 4
    elif glsltype == "vec3":
        # return 12
        return 4
    elif glsltype == "vec4":
        # return 16
        return 4
    else:
        raise Exception("Unrecognized type: " + glsltype)


class Buffer(sinode.Sinode):
    currLocation = 0

    def __init__(
        self,
        device,
        name,
        location,
        dimensionVals,
        DEBUG=False,
        format=vk.VK_FORMAT_R64_SFLOAT,
        readFromCPU=True,
        usage=vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        qualifier="",
        memtype="float",
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=4,
        compress=True,
    ):
        self.stageFlags =stageFlags
        self.DEBUG = DEBUG
        self.format = format
        self.compress = compress
        self.dimensionVals = dimensionVals
        # this should be fixed in vulkan wrapper
        self.released = False
        self.usage = usage
        sinode.Sinode.__init__(self, device)
        self.device = device
        self.location = location
        self.vkDevice = device.vkDevice
        self.qualifier = qualifier
        self.type = memtype
        self.itemSize = glsltype2bytesize(self.type)
        self.pythonType = glsltype2python(self.type)
        self.getSkipval()

        # for vec3 etc, the size is already bakd in
        self.itemCount = int(np.prod(dimensionVals))
        self.sizeBytes = int(self.itemCount * self.itemSize * self.skipval)
        self.name = name
        

        self.device.instance.debug("creating buffer " + name)

        # We will now create a buffer with these options
        self.bufferCreateInfo = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=self.sizeBytes,  # buffer size in bytes.
            usage=usage,  # buffer is used as a storage buffer.
            sharingMode=sharingMode,  # buffer is exclusive to a single queue family at a time.
        )
        self.device.instance.debug(self.vkDevice)
        self.device.instance.debug(self.bufferCreateInfo)
        self.vkBuffer = vk.vkCreateBuffer(self.vkDevice, self.bufferCreateInfo, None)

        # But the buffer doesn't allocate memory for itself, so we must do that manually.

        # First, we find the memory requirements for the buffer.
        memoryRequirements = vk.vkGetBufferMemoryRequirements(
            self.vkDevice, self.vkBuffer
        )

        # There are several types of memory that can be allocated, and we must choose a memory type that:
        # 1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
        # 2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
        #    with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
        # Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
        # visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
        # this flag.
        index = device.findMemoryType(memoryRequirements.memoryTypeBits, memProperties)

        if index < 0:
            raise Exception("Requested memory type not available on this device")

        # Now use obtained memory requirements info to allocate the memory for the buffer.
        self.allocateInfo = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memoryRequirements.size,  # specify required memory.
            memoryTypeIndex=index,
        )

        # allocate memory on device.
        self.device.instance.debug("allocating")
        self.vkDeviceMemory = vk.vkAllocateMemory(
            self.vkDevice, self.allocateInfo, None
        )

        self.device.instance.debug("done allocating")

        self.device.instance.debug("mapping")
        # Map the buffer memory, so that we can read from it on the CPU.
        self.pmap = vk.vkMapMemory(
            device=self.vkDevice,
            memory=self.vkDeviceMemory,
            offset=0,
            size=self.sizeBytes,
            flags=0,
        )
        self.device.instance.debug("done mapping")

        # these debug prints take forever
        # self.device.instance.debug(len(self.pmap[:]))
        # self.device.instance.debug(len(np.zeros((self.itemCount * self.skipval), dtype=self.pythonType)))

        # sometimes you may want to unmap from CPU
        if not readFromCPU:
            vk.vkUnmapMemory(self.vkDevice, self.vkDeviceMemory)
            self.pmap = None

        self.device.instance.debug("binding to device")
        # Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
        vk.vkBindBufferMemory(
            device=self.vkDevice,
            buffer=self.vkBuffer,
            memory=self.vkDeviceMemory,
            memoryOffset=0,
        )
        self.device.instance.debug("done binding to device")
        
        self.vkMappedMemoryRange = vk.VkMappedMemoryRange(
            sType = vk.VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            pNext = None,
            memory=self.vkDeviceMemory,
            offset=0,
            size=int(self.sizeBytes)
        )
        
        # initialize to zero
        self.zeroInitialize()
        self.flush()
        self.device.instance.debug("done initializing")


        # NEEDED FOR RAYTRACING, FAILS BEFORE VULKAN 1.3
        # self.bufferDeviceAddressInfo = VkBufferDeviceAddressInfo(
        #    sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        #    pNext=None,
        #    buffer=self.vkBuffer,
        # )

        # Maintain an address pointer for feed operations
        self.addrPtr = 0

    def flush(self):
        return vk.vkFlushMappedMemoryRanges(device = self.device.vkDevice, memoryRangeCount = 1, pMemoryRanges=[self.vkMappedMemoryRange])
        
    def getDescriptorBinding(self):
        
        self.binding = self.descriptorSet.getBufferBinding()
        self.descriptorSet.buffers += [self]
        # descriptorCount is the number of descriptors contained in the binding,
        # accessed in a shader as an array, except if descriptorType is
        # VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK in which case descriptorCount
        # is the size in bytes of the inline uniform block
        # self.device.instance.debug("BUFFER DTYPE")
        # self.device.instance.debug(descriptorSet.type)
        self.descriptorSetLayoutBinding = vk.VkDescriptorSetLayoutBinding(
            binding=self.binding,
            descriptorType=self.descriptorSet.type,
            descriptorCount=1,
            stageFlags=self.stageFlags,
        )

        # Specify the buffer to bind to the descriptor.
        # Every buffer contains its own info for descriptor set
        # Next, we need to connect our actual storage buffer with the descrptor.
        # We use vkUpdateDescriptorSets() to update the descriptor set.
        self.descriptorBufferInfo = vk.VkDescriptorBufferInfo(
            buffer=self.vkBuffer, offset=0, range=self.sizeBytes
        )

        

    # in some cases, memory access from the shader must be in increments of 16 bytes
    # so if we have a 4-byte float, we need to skip every 4th memory element
    def getSkipval(self):
        if (
            self.compress
            and self.usage == vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            and (self.type == "float64_t" or self.type == "float")
        ):
            self.skipval = 1
        elif (
            not self.usage & vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT and not self.usage & vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT
        ) and self.itemSize <= 8:
            self.skipval = int(16 / self.itemSize)
        # vec3s (12 bytes) does not divide evenly into 16 :(
        # elif self.itemSize == 12:
        #    self.skipval = 4.0/3
        else:
            self.skipval = int(1)

    def debugSizeParams(self):
        self.device.instance.debug("itemCount " + str(self.itemCount))
        self.device.instance.debug("itemSize " + str(self.itemSize))
        self.device.instance.debug("skipval " + str(self.skipval))
        self.device.instance.debug("sizeBytes " + str(self.sizeBytes))

    def zeroInitialize(self):
        self.set(np.zeros((self.itemCount), dtype=self.pythonType))

    def oneInitialize(self):
        self.set(np.ones((self.itemCount), dtype=self.pythonType))

    def getAsNumpyArray(self, asComplex=False, flat=False, order="C"):
        # glsl to python
        flatArray = np.frombuffer(self.pmap, self.pythonType)
        # because GLSL only allows 16-byte access,
        # we need to skip a few values in the memory
        if asComplex:
            rcvdArray = list(flatArray.astype(float))
            rcvdArrayReal = rcvdArray[::4]
            rcvdArrayImag = 1j * flatArray[1::4].astype(complex)
            rcvdArrayComplex = rcvdArrayReal + rcvdArrayImag
            # finally, reshape according to the expected dims
            rcvdArray = np.array(rcvdArrayComplex).reshape(self.dimensionVals)
        elif self.type == "vec2":
            rcvdArrayList = list(flatArray.astype(float))
            rcvdArray = np.zeros(self.dimensionVals + [2])
            rcvdArray = np.append(
                np.expand_dims(rcvdArrayList[::4], 1),
                np.expand_dims(rcvdArrayList[1::4], 1),
                axis=1,
            )

        else:
            if self.compress:
                if flat:
                    rcvdArray = flatArray
                else:
                    rcvdArray = flatArray.reshape(self.dimensionVals, order=order)

            else:
                indices = np.arange(0, len(flatArray), self.skipval).astype(int)
                rcvdArray = np.array(flatArray[indices]).reshape(self.dimensionVals)
        return rcvdArray

    def saveAsImage(self, height, width, path="mandelbrot.png"):

        # Get the color data from the buffer, and cast it to bytes.
        # We save the data to a vector.
        st = time.time()

        pa = np.frombuffer(self.pmap, np.float32)
        pa = pa.reshape((height, width, 4))
        pa *= 255

        self.cpuDataConverTime = time.time() - st

        # Done reading, so unmap.
        # vkUnmapMemory(self.vkDevice, self.__bufferMemory)

        # Now we save the acquired color data to a .png.
        image = pilImage.fromarray(pa.astype(np.uint8))
        image.save(path)

    def release(self):
        if not self.released:
            self.device.instance.debug("destroying buffer " + self.name)
            vk.vkFreeMemory(self.vkDevice, self.vkDeviceMemory, None)
            vk.vkDestroyBuffer(self.vkDevice, self.vkBuffer, None)
            self.released = True

    def getDeclaration(self):
        if "uniform" in self.qualifier:
            return (
                "layout (location = "
                + str(self.location)
                + ", binding = "
                + str(self.binding)
                + ") "
                + self.qualifier
                + " "
                + self.type
                + " "
                + self.name
                + ";\n"
            )
        else:
            return (
                "layout (location = "
                + str(self.location)
                + ") "
                + self.qualifier
                + " "
                + self.type
                + " "
                + self.name
                + ";\n"
            )

    def getComputeDeclaration(self):
        if self.usage == vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:
            b = "uniform "
            std = "std140"
        else:
            b = "buffer "
            if self.compress:
                std = "std430"
            else:
                std = "std140"

        return (
            "layout("
            + std
            + ", set = "
            + str(self.descriptorSet.binding)
            + ", binding = "
            + str(self.binding)
            # + ", "
            # + "xfb_stride = " + str(self.stride)
            + ") "
            + b
            + self.name
            + "_buf\n{\n   "
            + self.qualifier
            + " "
            + self.type
            + " "
            + self.name
            + "["
            + str(int(self.sizeBytes / self.itemSize))
            + "];\n};\n"
        )

    def write(self, data):
        startByte = self.addrPtr
        endByte = self.addrPtr + len(data) * self.itemSize
        self.addrPtr = endByte

        self.pmap[startByte:endByte] = data
        return startByte

    def setByIndexVec(self, index, data):
        # self.device.instance.debug(self.name + " setting " + str(index) + " to " + str(data))
        startByte = index * self.itemSize * self.skipval
        self.pmap[startByte : startByte + 4] = np.real(data).astype(np.float32)
        self.pmap[startByte + 4 : startByte + 8] = np.imag(data).astype(np.float32)

        # self.device.instance.debug("setting " + str(index) + " to " + str(np.real(data).astype(np.float32)))
        # self.device.instance.debug("setting " + str(index) + ".i to " + str(np.imag(data).astype(np.float32)))

    def setByIndex(self, index, data):
        # self.device.instance.debug(self.name + " setting " + str(index) + " to " + str(data))
        startByte = index * self.itemSize * self.skipval
        endByte = index * self.itemSize * self.skipval + self.itemSize
        self.pmap[startByte:endByte] = np.array(data, dtype=self.pythonType)

    def setByIndexStart(self, startIndex, data):
        # if self.skipval != 1:
        #    raise ("You can only do this with new-format storage buffers!")
        # self.device.instance.debug(self.name + " setting " + str(index) + " to " + str(data))
        startByte = startIndex * self.itemSize * self.skipval
        endByte = startIndex * self.itemSize * self.skipval + self.itemSize * len(data)
        self.pmap[startByte:endByte] = np.array(data, dtype=self.pythonType)

    def getByIndex(self, index):
        # self.device.instance.debug(self.name + " setting " + str(index) + " to " + str(data))
        startByte = index * self.itemSize * self.skipval
        endByte = index * self.itemSize * self.skipval + self.itemSize
        return np.frombuffer(self.pmap[startByte:endByte], dtype=self.pythonType)

    def set(self, data, flush=True):
        # self.pmap[:] = data.astype(self.pythonType)
        try:
            if self.skipval == 1:
                self.pmap[:] = data.astype(self.pythonType).flatten()
            else:
                indices = np.arange(0, len(data), 1.0 / self.skipval).astype(int)
                data = data[indices]
                #print(self.pythonType)
                self.pmap[:] = data.astype(self.pythonType).flatten()

        except:
            self.device.instance.debug("WRONG SIZE")
            self.device.instance.debug("pmap (bytes): " + str(len(self.pmap[:])))
            self.device.instance.debug(
                "data (bytes): " + str(len(data) * self.itemSize)
            )
            raise Exception("Wrong Size")
        
        if flush:
            self.flush()
        
    def fill(self, value):
        # self.pmap[: data.size * data.itemSize] = data
        a = np.array([value])
        # self.pmap[:a] = data[:int(a/(data.itemSize))]
        self.pmap[:] = np.tile(a, int(len(self.pmap[:]) / a.itemSize))

    def getSize(self):
        with open(os.path.join(here, "derivedtypes.json"), "r") as f:
            derivedDict = json.loads(f.read())
        with open(os.path.join(here, "ctypes.json"), "r") as f:
            cDict = json.loads(f.read())
        size = 0
        if self.type in derivedDict.keys():
            for subtype in derivedDict[self.type]:
                size += self.getSize(subtype)
        else:
            size += 1
        return int(size)


class StorageBuffer(Buffer):
    def __init__(
        self,
        device,
        name,
        dimensionVals,
        DEBUG=False,
        qualifier="",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        descriptorSet=None,
        memtype="float",
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=12,
        compress=True,
    ):
        self.descriptorSet = descriptorSet
        if descriptorSet is None:
            self.descriptorSet = device.descriptorPool.descSetGlobal
        Buffer.__init__(
            self,
            DEBUG=False,
            device=device,
            name=name,
            location=0,
            dimensionVals=dimensionVals,
            format=vk.VK_FORMAT_R64_SFLOAT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            memProperties=memProperties,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            qualifier=qualifier,
            memtype=memtype,
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=4,
            compress=compress,
        )
        self.getDescriptorBinding()

class DebugBuffer(StorageBuffer):
    def __init__(
        self,
        device,
        name,
        dimIndexNames,
        dimensionVals,
        memtype="vec3",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=12,
    ):
        self.dimIndexNames = dimIndexNames
        StorageBuffer.__init__(
            self,
            DEBUG=True,
            device=device,
            name=name,
            location=0,
            dimensionVals=dimensionVals,
            format=VK_FORMAT_R64_SFLOAT,
            readFromCPU=True,
            memProperties=memProperties,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            qualifier="",
            memtype=memtype,
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=12,
            compress=True,
        )

class VertexBuffer(Buffer):
    def __init__(
        self,
        device,
        name,
        dimensionVals,
        location,
        DEBUG=False,
        qualifier="",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        memtype="float",
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=12,
        compress=False,
    ):
        # self.location = Buffer.currLocation
        # Buffer.currLocation+=1

        Buffer.__init__(
            self,
            DEBUG=False,
            device=device,
            name=name,
            location=location,
            dimensionVals=dimensionVals,
            format=vk.VK_FORMAT_R32G32B32_SFLOAT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            memProperties=memProperties,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
            qualifier=qualifier,
            memtype=memtype,
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=stride,
            compress=compress,
        )
        self.binding = location
        # the following are only needed for vertex buffers
        # VK_VERTEX_INPUT_RATE_VERTEX: Move to the next data entry after each vertex
        # VK_VERTEX_INPUT_RATE_INSTANCE: Move to the next data entry after each instance

        # we will standardize its bindings with a attribute description
        self.attributeDescription = vk.VkVertexInputAttributeDescription(
            binding=self.binding, location=self.location, format=self.format, offset=0
        )
        # ^^ Consider VK_FORMAT_R32G32B32A32_SFLOAT  ?? ^^
        self.bindingDescription = vk.VkVertexInputBindingDescription(
            binding=self.binding, stride=stride, inputRate=rate  # 4 bytes/element
        )


class IndexBuffer(Buffer):
    def __init__(
        self,
        device,
        dimensionVals,
        DEBUG=False,
        qualifier="",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        # | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=4,
    ):
        Buffer.__init__(
            self,
            DEBUG=False,
            device=device,
            name="index",
            location=0,
            dimensionVals=dimensionVals,
            format=vk.VK_FORMAT_R32_UINT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            memProperties=memProperties,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_ALL_GRAPHICS,
            qualifier=qualifier,
            memtype="uint",
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=4,
            compress=True,
        )


class FragmentBuffer(Buffer):
    def __init__(
        self,
        device,
        name,
        dimensionVals,
        DEBUG=False,
        qualifier="",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        memtype="float",
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=12,
        compress=False,
    ):
        Buffer.__init__(
            self,
            DEBUG=False,
            device=device,
            name=name,
            location=0,
            dimensionVals=dimensionVals,
            format=vk.VK_FORMAT_R64_SFLOAT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            memProperties=memProperties,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            qualifier=qualifier,
            memtype=memtype,
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=stride,
            compress=compress,
        )


class UniformBuffer(Buffer):
    def __init__(
        self,
        device,
        name,
        dimensionVals,
        DEBUG=False,
        qualifier="",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        memtype="float",
        rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        stride=12,
    ):
        self.descriptorSet = descriptorSet
        if descriptorSet is None:
            self.descriptorSet = device.descriptorPool.descSetUniform
        Buffer.__init__(
            self,
            DEBUG=False,
            device=device,
            name=name,
            location=0,
            descriptorSet=device.descriptorPool.descSetUniform,
            dimensionVals=dimensionVals,
            format=vk.VK_FORMAT_R64_SFLOAT,
            readFromCPU=True,
            usage=vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            memProperties=memProperties,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            qualifier=qualifier,
            memtype=memtype,
            rate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            stride=12,
            compress=False,
        )
        self.getDescriptorBinding()


class AccelerationStructure(Buffer):
    def __init__(self, setupDict, shader):
        Descriptorset.__init__(self, shader)
        self.pipeline = shader.pipeline
        self.pipelineDict = self.pipeline.setupDict
        self.vkCommandPool = self.pipeline.device.vkCommandPool
        self.device = self.pipeline.device
        self.vkDevice = self.pipeline.device.vkDevice
        self.outputWidthPixels = self.pipeline.outputWidthPixels
        self.outputHeightPixels = self.pipeline.outputHeightPixels


class AccelerationStructureNV(AccelerationStructure):
    def __init__(self, setupDict, shader):
        AccelerationStructure.__init__(self, setupDict, shader)

        # We need to get the compactedSize with a query

        # // Get the size result back
        # std::vector<VkDeviceSize> compactSizes(m_blas.size());
        # vkGetQueryPoolResults(m_device, queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
        # 											compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);

        # just playing. we will guess that b***h

        # Provided by VK_NV_ray_tracing
        self.asCreateInfo = VkAccelerationStructureCreateInfoNV(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
            pNext=None,
            compactedSize=642000,  # VkDeviceSize
        )

        # Provided by VK_NV_ray_tracing
        self.vkAccelerationStructure = vkCreateAccelerationStructureNV(
            device=self.vkDevice, pCreateInfo=self.asCreateInfo, pAllocator=None
        )


# If type is VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV then geometryCount must be 0
class TLASNV(AccelerationStructureNV):
    def __init__(self, setupDict, shader):
        AccelerationStructureNV.__init__(self, setupDict, shader)

        for blasName, blasDict in setupDict["blas"].items():
            newBlas = BLASNV(blasDict, shader)
            self.children += [newBlas]

        # Provided by VK_NV_ray_tracing
        self.asInfo = VkAccelerationStructureInfoNV(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
            pNext=None,  # const void*
            type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=len(self.children),  # uint32_t
            geometryCount=0,  # uint32_t
            pGeometries=None,  # const VkGeometryNV*
        )


class Geometry(sinode.Sinode):
    def __init__(self, setupDict, blas, initialMesh):
        sinode.Sinode.__init__(self, setupDict, blas)
        buffSetupDict = {}
        buffSetupDict["vertex"] = [[0, 1, 0], [1, 1, 1], [1, 1, 0]]
        buffSetupDict["index"] = [[0, 1, 2]]
        buffSetupDict["aabb"] = [[0, 1, 2]]
        self.vertexBuffer = Buffer(
            self.lookUp("device"), buffSetupDict["vertex"].flatten()
        )
        self.indexBuffer = Buffer(
            self.lookUp("device"), buffSetupDict["index"].flatten()
        )
        self.aabb = Buffer(self.lookUp("device"), buffSetupDict["aabb"].flatten())

        # ccw rotation
        theta = 0
        self.vkTransformMatrix = VkTransformMatrixKHR(
            # float    matrix[3][4];
            [cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1]
        )

        self.geometryTriangles = VkGeometryTrianglesNV(
            sType=VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
            pNext=None,
            vertexData=self.buffer.vkBuffer,
            vertexOffset=0,
            vertexCount=len(buffSetupDict["vertex"].flatten()),
            vertexStride=12,
            vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
            indexData=self.indexBuffer.vkBuffer,
            indexOffset=0,
            indexCount=len(buffSetupDict["index"].flatten()),
            indexType=VK_INDEX_TYPE_UINT32,
            transformData=self.vkTransformMatrix,
            transformOffset=0,
        )

        self.aabbs = VkGeometryAABBNV(
            sType=VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV,
            pNext=None,
            aabbData=self.aabb.vkBuffer,
            numAABBs=1,
            stride=4,
            offset=0,
        )

        self.geometryData = VkGeometryDataNV(
            triangles=self.geometryTriangles, aabbs=self.aabbs
        )

        # possible flags:

        # VK_GEOMETRY_OPAQUE_BIT_KHR = 0x00000001,
        # VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR = 0x00000002,
        # // Provided by VK_NV_ray_tracing
        # VK_GEOMETRY_OPAQUE_BIT_NV = VK_GEOMETRY_OPAQUE_BIT_KHR,
        # // Provided by VK_NV_ray_tracing
        # VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_NV

        # VK_GEOMETRY_OPAQUE_BIT_KHR indicates that this geometry does
        # not invoke the any-hit shaders even if present in a hit group.

        self.vkGeometry = VkGeometryNV(
            sType=VK_STRUCTURE_TYPE_GEOMETRY_NV,
            pNext=None,
            geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            geometry=self.geometryData,
            flags=VK_GEOMETRY_OPAQUE_BIT_KHR,
        )


# If type is VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV then instanceCount must be 0
class BLASNV(AccelerationStructureNV):
    def __init__(self, setupDict, shader, initialMesh):
        AccelerationStructureNV.__init__(self, setupDict, shader)

        self.geometry = Geometry(initialMesh, self)

        # Provided by VK_NV_ray_tracing
        self.asInfo = VkAccelerationStructureInfoNV(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
            pNext=None,  # const void*
            type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=0,  # uint32_t
            geometryCount=1,  # uint32_t
            pGeometries=[self.geometry.vkGeometry],  # const VkGeometryNV*
        )


class AccelerationStructureKHR(AccelerationStructure):
    def __init__(self, setupDict, shader):
        AccelerationStructure.__init__(self, setupDict, shader)

        # Identify the above data as containing opaque triangles.
        asGeom = VkAccelerationStructureGeometryKHR(
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            flags=VK_GEOMETRY_OPAQUE_BIT_KHR,
            triangles=geometry.triangles,
        )

        # The entire array will be used to build the BLAS.
        offset = VkAccelerationStructureBuildRangeInfoKHR(
            firstVertex=0, primitiveCount=53324234, primitiveOffset=0, transformOffset=0
        )

        # Provided by VK_NV_ray_tracing
        pCreateInfo = VkAccelerationStructureCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,  # VkStructureType
            pNext=None,  # const void*
            compactedSize=642000,  # VkDeviceSize
        )

        # Provided by VK_NV_ray_tracing
        self.vkAccelerationStructure = vkCreateAccelerationStructureNV(
            device=self.vkDevice, pCreateInfo=self.asCreateInfo, pAllocator=None
        )


# If type is VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_ then instanceCount must be 0
class BLAS(AccelerationStructure):
    def __init__(self, setupDict, shader, initialMesh):
        AccelerationStructure.__init__(self, setupDict, shader)

        self.geometry = Geometry(initialMesh, self)

        # Provided by VK__ray_tracing
        self.asInfo = VkAccelerationStructureInfo(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_,
            pNext=None,  # const void*
            type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=0,  # uint32_t
            geometryCount=1,  # uint32_t
            pGeometries=[self.geometry.vkGeometry],  # const VkGeometry*
        )


# If type is VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_ then geometryCount must be 0
class TLAS(AccelerationStructure):
    def __init__(self, setupDict, shader):
        AccelerationStructure.__init__(self, setupDict, shader)

        for blasName, blasDict in setupDict["blas"].items():
            newBlas = BLAS(blasDict, shader)
            self.children += [newBlas]

        # Provided by VK__ray_tracing
        self.asInfo = VkAccelerationStructureInfo(
            sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_,
            pNext=None,  # const void*
            type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            instanceCount=len(self.children),  # uint32_t
            geometryCount=0,  # uint32_t
            pGeometries=None,  # const VkGeometry*
        )
