import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *

class Buffer(PrintClass):

	# find memory type with desired properties.
	def findMemoryType(self, memoryTypeBits, properties):
		memoryProperties = vkGetPhysicalDeviceMemoryProperties(self.device.physical_device)

		# How does this search work?
		# See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
		for i, mt in enumerate(memoryProperties.memoryTypes):
			if memoryTypeBits & (1 << i) and (mt.propertyFlags & properties) == properties:
				return i

		return -1

	def __init__(self, device, setupDict):
		PrintClass.__init__(self)
		self.setupDict= setupDict
		self.device   = device
		self.vkDevice = device.vkDevice
		self.setupDict["SIZEBYTES"]= setupDict["SIZEBYTES"]
		
		# We will now create a buffer with these options
		bufferCreateInfo = VkBufferCreateInfo(
			sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			size =setupDict["SIZEBYTES"],  # buffer size in bytes.
			usage=eval(setupDict["usage"]),  # buffer is used as a storage buffer.
			sharingMode=eval(setupDict["sharingMode"])  # buffer is exclusive to a single queue family at a time.
		)
		self.vkBuffer = vkCreateBuffer(self.vkDevice, bufferCreateInfo, None)
		self.children += [self.vkBuffer]

		# we will standardize its bindings with a attribute description
		self.attributeDescription = VkVertexInputAttributeDescription(
			binding  = setupDict["binding"],
			location = setupDict["location"],
			format   = eval(setupDict["format"]), # single, 4 bytes
			offset   = 0
		)
		# ^^ Consider VK_FORMAT_R32G32B32A32_SFLOAT  ?? ^^ 
		
		# But the buffer doesn't allocate memory for itself, so we must do that manually.

		# First, we find the memory requirements for the buffer.
		memoryRequirements = vkGetBufferMemoryRequirements(self.vkDevice, self.vkBuffer)

		# There are several types of memory that can be allocated, and we must choose a memory type that:
		# 1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
		# 2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
		#    with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
		# Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
		# visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
		# this flag.
		index = self.findMemoryType(memoryRequirements.memoryTypeBits,
									VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		# Now use obtained memory requirements info to allocate the memory for the buffer.
		allocateInfo = VkMemoryAllocateInfo(
			sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			allocationSize=memoryRequirements.size,  # specify required memory.
			memoryTypeIndex=index
		)

		# allocate memory on device.
		self.vkBufferMemory = vkAllocateMemory(self.vkDevice, allocateInfo, None)
		self.children += [self.vkBufferMemory]

		# Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
		vkBindBufferMemory(self.vkDevice, self.vkBuffer, self.vkBufferMemory, 0)
		
		# Map the buffer memory, so that we can read from it on the CPU.
		self.pmap = vkMapMemory(self.vkDevice, self.vkBufferMemory, 0, self.setupDict["SIZEBYTES"], 0)

		self.bindingDescription = VkVertexInputBindingDescription(
			binding   = setupDict["binding"],
			stride    = setupDict["stride"], #4 bytes/element
			inputRate = VK_VERTEX_INPUT_RATE_VERTEX)
			
		#VK_VERTEX_INPUT_RATE_VERTEX: Move to the next data entry after each vertex
		#VK_VERTEX_INPUT_RATE_INSTANCE: Move to the next data entry after each instance

		

	def saveAsImage(self, height, width, path = 'mandelbrot.png'):

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
		print("destroying buffer " + self.setupDict["name"])
		vkFreeMemory(self.vkDevice, self.vkBufferMemory, None)
		vkDestroyBuffer(self.vkDevice, self.vkBuffer, None)
	