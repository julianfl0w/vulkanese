import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *

class AccelerationStructure(Sinode):
	def __init__(self, setupDict, shader):
		Sinode.__init__(self, shader)
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
				pAllocator  = None );
				
#If type is VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV then geometryCount must be 0
class TLASNV(AccelerationStructureNV)
	def __init__(self, setupDict, shader):
		AccelerationStructureNV.__init__(self, setupDict, shader)
		
		for blasName, blasDict in setupDict["blas"]
			newBlas = BLASNV(blasDict, shader)
			self.children += [newBlas]
			
		# Provided by VK_NV_ray_tracing
		self.asInfo = VkAccelerationStructureInfoNV (
				sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV, 
				pNext          = None,   # const void*                            
				type           = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR , 
				flags          = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
				instanceCount  = ,   # uint32_t                               
				geometryCount  = 0,   # uint32_t                               
				pGeometries    = ,   # const VkGeometryNV*                    
		)

#If type is VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV then instanceCount must be 0
class BLASNV(AccelerationStructureNV)
	def __init__(self, setupDict, shader):
		AccelerationStructureNV.__init__(self, setupDict, shader)
		# Provided by VK_NV_ray_tracing
		self.asInfo = VkAccelerationStructureInfoNV (
				sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
				pNext          = None,   # const void*                            
				type           = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR , 
				flags          = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
				instanceCount  = 0,   # uint32_t                               
				geometryCount  = ,   # uint32_t                               
				pGeometries    = ,   # const VkGeometryNV*                    
		)


class AccelerationStructureKHR(AccelerationStructure):
	def __init__(self, setupDict, shader):
		AccelerationStructure.__init__(self, setupDict, shader)
		
		# Identify the above data as containing opaque triangles.
		asGeom = VkAccelerationStructureGeometryKHR (
			VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
			geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
			flags              = VK_GEOMETRY_OPAQUE_BIT_KHR,
			geometry.triangles = triangles)

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
				