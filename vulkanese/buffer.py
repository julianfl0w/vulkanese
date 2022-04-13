import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *
import numpy as np 

class Buffer(Sinode):

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
		Sinode.__init__(self, device)
		self.setupDict= setupDict
		self.device   = device
		self.vkDevice = device.vkDevice
		self.size = setupDict["SIZEBYTES"]
		
		print("creating buffer with description")
		print(json.dumps(setupDict, indent=2))
		
		# We will now create a buffer with these options
		bufferCreateInfo = VkBufferCreateInfo(
			sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			size =setupDict["SIZEBYTES"],  # buffer size in bytes.
			usage=eval(setupDict["usage"]),  # buffer is used as a storage buffer.
			sharingMode=eval(setupDict["sharingMode"])  # buffer is exclusive to a single queue family at a time.
		)
		self.vkBuffer = vkCreateBuffer(self.vkDevice, bufferCreateInfo, None)
		self.children += [self.vkBuffer]

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
		index = self.findMemoryType(memoryRequirements.memoryTypeBits, eval(setupDict["memProperties"]))
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
		
		
	def getDeclaration(self):
		bufferDict = self.setupDict
		return "layout (location = " + str(bufferDict["location"]) + ") " + bufferDict["qualifier"] + " " + bufferDict["type"] + " " + bufferDict["name"] + ";\n"
	
class VertexBuffer(Buffer):
	def __init__(self, device, setupDict):
		Buffer.__init__(self, device, setupDict)
		
		outfilename = os.path.join("resources", "standard_bindings.json")
		with open(outfilename, 'r') as f:
			bindDict = json.loads(f.read())
			
		self.binding = self.getAncestor("device").getBinding(self, setupDict["descriptorSet"])
		self.setupDict["binding"] = self.binding 
		
		# we will standardize its bindings with a attribute description
		self.attributeDescription = VkVertexInputAttributeDescription(
			binding  = self.binding,
			location = setupDict["location"],
			format   = eval(setupDict["format"]), # single, 4 bytes
			offset   = 0
		)
		# ^^ Consider VK_FORMAT_R32G32B32A32_SFLOAT  ?? ^^ 
		self.bindingDescription = VkVertexInputBindingDescription(
			binding   = self.binding,
			stride    = setupDict["stride"], #4 bytes/element
			inputRate = eval(setupDict["rate"]))
		
		# Every buffer contains its own info for descriptor set
		# Next, we need to connect our actual storage buffer with the descrptor.
		# We use vkUpdateDescriptorSets() to update the descriptor set.
		self.descriptorBufferInfo = VkDescriptorBufferInfo(
			buffer=self.vkBuffer,
			offset=0,
			range=setupDict["SIZEBYTES"]
		)
		
		#VK_VERTEX_INPUT_RATE_VERTEX: Move to the next data entry after each vertex
		#VK_VERTEX_INPUT_RATE_INSTANCE: Move to the next data entry after each instance
		
	def getDeclaration(self):
		bufferDict = self.setupDict
		if "uniform" in bufferDict["qualifier"]:
			return "layout (location = " + str(bufferDict["location"]) + ", binding = " + str(bufferDict["binding"]) + ") " + bufferDict["qualifier"] + " " + bufferDict["type"] + " " + bufferDict["name"] + ";\n"
		else:
			return "layout (location = " + str(bufferDict["location"]) + ") " + bufferDict["qualifier"] + " " + bufferDict["type"] + " " + bufferDict["name"] + ";\n"
			

class DescriptorSetBuffer(Buffer):
	def __init__(self, device, setupDict):
		Buffer.__init__(self, device, setupDict)
		
class PushConstantsBuffer(DescriptorSetBuffer):
	def __init__(self, device, setupDict):
		DescriptorSetBuffer.__init__(self, device, setupDict)
		
class UniformBuffer(DescriptorSetBuffer):
	def __init__(self, device, setupDict):
		DescriptorSetBuffer.__init__(self, device, setupDict)
		
class UniformTexelBuffer(DescriptorSetBuffer):
	def __init__(self, device, setupDict):
		DescriptorSetBuffer.__init__(self, device, setupDict)
		
class SampledImageBuffer(DescriptorSetBuffer):
	def __init__(self, device, setupDict):
		DescriptorSetBuffer.__init__(self, device, setupDict)
		
class StorageBuffer(DescriptorSetBuffer):
	def __init__(self, device, setupDict):
		DescriptorSetBuffer.__init__(self, device, setupDict)

class StorageTexelBuffer(DescriptorSetBuffer):
	def __init__(self, device, setupDict):
		DescriptorSetBuffer.__init__(self, device, setupDict)
		
class StorageImageBuffer(DescriptorSetBuffer):
	def __init__(self, device, setupDict):
		DescriptorSetBuffer.__init__(self, device, setupDict)
		

class AccelerationStructure(DescriptorSetBuffer):
	def __init__(self, setupDict, shader):
		DescriptorSetBuffer.__init__(self, shader)
		self.pipeline           = shader.pipeline
		self.pipelineDict       = self.pipeline.setupDict
		self.vkCommandPool      = self.pipeline.device.vkCommandPool
		self.device             = self.pipeline.device
		self.vkDevice           = self.pipeline.device.vkDevice
		self.outputWidthPixels  = self.pipeline.setupDict["outputWidthPixels"]
		self.outputHeightPixels = self.pipeline.setupDict["outputHeightPixels"]

class AccelerationStructureNV(AccelerationStructure):
	def __init__(self, setupDict, shader):
		AccelerationStructure.__init__(self, setupDict, shader)
	
		# We need to get the compactedSize with a query
		
		#// Get the size result back
		#std::vector<VkDeviceSize> compactSizes(m_blas.size());
		#vkGetQueryPoolResults(m_device, queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
		#											compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);
		
		# just playing. we will guess that b***h

		# Provided by VK_NV_ray_tracing
		self.asCreateInfo = VkAccelerationStructureCreateInfoNV(
				sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
				pNext         = None,  
				compactedSize = 642000   # VkDeviceSize
		)

		# Provided by VK_NV_ray_tracing
		self.vkAccelerationStructure = vkCreateAccelerationStructureNV(
				device      = self.vkDevice,
				pCreateInfo = self.asCreateInfo,
				pAllocator  = None )
				
#If type is VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV then geometryCount must be 0
class TLASNV(AccelerationStructureNV):
	def __init__(self, setupDict, shader):
		AccelerationStructureNV.__init__(self, setupDict, shader)
		
		for blasName, blasDict in setupDict["blas"].items():
			newBlas = BLASNV(blasDict, shader)
			self.children += [newBlas]
			
		# Provided by VK_NV_ray_tracing
		self.asInfo = VkAccelerationStructureInfoNV (
				sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV, 
				pNext          = None,   # const void*                            
				type           = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR , 
				flags          = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
				instanceCount  = len(self.children),   # uint32_t                               
				geometryCount  = 0,   # uint32_t                               
				pGeometries    = None,   # const VkGeometryNV*                    
		)
		
class Geometry(Sinode):
	def __init__(self, setupDict, blas, initialMesh):
		Sinode.__init__(self, setupDict, blas)
		buffSetupDict = {}                         
		buffSetupDict["vertex"] = [[0,1,0], [1,1,1], [1,1,0]] 
		buffSetupDict["index" ] = [[0,1,2]]                   
		buffSetupDict["aabb"  ] = [[0,1,2]]                    
		self.vertexBuffer = Buffer(self.lookUp("device"), buffSetupDict["vertex"].flatten())
		self.indexBuffer = Buffer(self.lookUp("device"), buffSetupDict["index"].flatten())
		self.aabb        = Buffer(self.lookUp("device"), buffSetupDict["aabb"].flatten())
		
		
		# ccw rotation
		theta = 0
		self.vkTransformMatrix = VkTransformMatrixKHR(
			#float    matrix[3][4];
			[cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1]
		)

		self.geometryTriangles = VkGeometryTrianglesNV(
			sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV, 
			pNext = None, 
			vertexData   = self.buffer.vkBuffer,
			vertexOffset = 0,
			vertexCount  = len(buffSetupDict["vertex"].flatten()),
			vertexStride = 12, 
			vertexFormat = VK_FORMAT_R32G32B32_SFLOAT, 
			indexData    = self.indexBuffer.vkBuffer, 
			indexOffset  = 0, 
			indexCount   = len(buffSetupDict["index"].flatten()),
			indexType    = VK_INDEX_TYPE_UINT32, 
			transformData = self.vkTransformMatrix,
			transformOffset = 0
		)
		
		self.aabbs = VkGeometryAABBNV(
			sType = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV, 
			pNext = None,
			aabbData = self.aabb.vkBuffer,
			numAABBs = 1, 
			stride   = 4,
			offset   = 0
		)

		self.geometryData = VkGeometryDataNV(
			triangles = self.geometryTriangles, 
			aabbs     = self.aabbs
		)
		
		# possible flags: 
		
		#VK_GEOMETRY_OPAQUE_BIT_KHR = 0x00000001,
		#VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR = 0x00000002,
		#// Provided by VK_NV_ray_tracing
		#VK_GEOMETRY_OPAQUE_BIT_NV = VK_GEOMETRY_OPAQUE_BIT_KHR,
		#// Provided by VK_NV_ray_tracing
		#VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_NV 
		
		#VK_GEOMETRY_OPAQUE_BIT_KHR indicates that this geometry does 
		# not invoke the any-hit shaders even if present in a hit group.

		self.vkGeometry = VkGeometryNV(
			sType = VK_STRUCTURE_TYPE_GEOMETRY_NV, 
			pNext = None, 
			geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR, 
			geometry = self.geometryData, 
			flags = VK_GEOMETRY_OPAQUE_BIT_KHR
		)
		

#If type is VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV then instanceCount must be 0
class BLASNV(AccelerationStructureNV):
	def __init__(self, setupDict, shader, initialMesh):
		AccelerationStructureNV.__init__(self, setupDict, shader)
		
		self.geometry = Geometry(initialMesh, self)
	
		# Provided by VK_NV_ray_tracing
		self.asInfo = VkAccelerationStructureInfoNV (
				sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
				pNext          = None,   # const void*                            
				type           = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR , 
				flags          = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
				instanceCount  = 0,   # uint32_t                               
				geometryCount  = 1,   # uint32_t                               
				pGeometries    = [self.geometry.vkGeometry],   # const VkGeometryNV*                    
		)


class AccelerationStructureKHR(AccelerationStructure):
	def __init__(self, setupDict, shader):
		AccelerationStructure.__init__(self, setupDict, shader)
		
		# Identify the above data as containing opaque triangles.
		asGeom = VkAccelerationStructureGeometryKHR (
			VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
			geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
			flags              = VK_GEOMETRY_OPAQUE_BIT_KHR,
			triangles          = geometry.triangles)

		# The entire array will be used to build the BLAS.
		offset = VkAccelerationStructureBuildRangeInfoKHR(
			firstVertex     = 0,
			primitiveCount  = 53324234,
			primitiveOffset = 0,
			transformOffset = 0
		)
	

		# Provided by VK_NV_ray_tracing
		pCreateInfo = VkAccelerationStructureCreateInfoKHR(
				sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,   # VkStructureType                  
				pNext         = None,   # const void*                      
				compactedSize = 642000    # VkDeviceSize
		)

		# Provided by VK_NV_ray_tracing
		self.vkAccelerationStructure = vkCreateAccelerationStructureNV(
				device      = self.vkDevice,
				pCreateInfo = self.asCreateInfo,
				pAllocator  = None );
				
#If type is VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_ then instanceCount must be 0
class BLAS(AccelerationStructure):
	def __init__(self, setupDict, shader, initialMesh):
		AccelerationStructure.__init__(self, setupDict, shader)
		
		self.geometry = Geometry(initialMesh, self)
	
		# Provided by VK__ray_tracing
		self.asInfo = VkAccelerationStructureInfo (
				sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_,
				pNext          = None,   # const void*                            
				type           = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR , 
				flags          = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
				instanceCount  = 0,   # uint32_t                               
				geometryCount  = 1,   # uint32_t                               
				pGeometries    = [self.geometry.vkGeometry],   # const VkGeometry*                    
		)


#If type is VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_ then geometryCount must be 0
class TLAS(AccelerationStructure):
	def __init__(self, setupDict, shader):
		AccelerationStructure.__init__(self, setupDict, shader)
		
		for blasName, blasDict in setupDict["blas"].items():
			newBlas = BLAS(blasDict, shader)
			self.children += [newBlas]
			
		# Provided by VK__ray_tracing
		self.asInfo = VkAccelerationStructureInfo (
				sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_, 
				pNext          = None,   # const void*                            
				type           = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR , 
				flags          = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
				instanceCount  = len(self.children),   # uint32_t                               
				geometryCount  = 0,   # uint32_t                               
				pGeometries    = None,   # const VkGeometry*                    
		)
		