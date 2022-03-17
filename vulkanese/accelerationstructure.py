import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *

class AccelerationStructure(PrintClass):
	def __init__(self, pipeline, vertices, indices):
		PrintClass.__init__(self)
		self.pipeline           = pipeline
		self.pipelineDict       = pipeline.setupDict
		self.vkCommandPool      = pipeline.device.vkCommandPool
		self.device             = pipeline.device
		self.vkDevice           = pipeline.device.vkDevice
		self.outputWidthPixels  = pipeline.setupDict["outputWidthPixels"]
		self.outputHeightPixels = pipeline.setupDict["outputHeightPixels"]
		
		# BLAS builder requires raw device addresses.
		BLAS_VERTEX_BUFFER = Buffer(53324234)
		BLAS_INDEX_BUFFER = Buffer(53324234)

		# Describe buffer as array of VertexObj.
		triangles = VkAccelerationStructureGeometryTrianglesDataKHR(
			VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR
			vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT,  # vec3 vertex position data.
			vertexData.deviceAddress = BLAS_VERTEX_BUFFER.pmap,
			vertexStride             = sizeof(VertexObj),
			# Describe index data (32-bit unsigned int)
			indexType               = VK_INDEX_TYPE_UINT32,
			indexData.deviceAddress = indexAddress,
			# Indicate identity transform by setting transformData to null device pointer.
			#triangles.transformData = {},
			maxVertex = model.nbVertices)

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
	
		# BLAS - Storing each primitive in a geometry
		self.rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
  
	def createTopLevelAS():
		std::vector<VkAccelerationStructureInstanceKHR> tlas;
		tlas.reserve(self.instances.size());
		for(const HelloVulkan::ObjInstance& inst : self.instances)
		{
		VkAccelerationStructureInstanceKHR rayInst{};
		rayInst.transform                      = nvvk::toTransformMatrixKHR(inst.transform);  # Position of the instance
		rayInst.instanceCustomIndex            = inst.objIndex;                               # gl_InstanceCustomIndexEXT
		rayInst.accelerationStructureReference = self.rtBuilder.getBlasDeviceAddress(inst.objIndex);
		rayInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		rayInst.mask                           = 0xFF;       #  Only be hit if rayMask & instance.mask != 0
		rayInst.instanceShaderBindingTableRecordOffset = 0;  # We will use the same hit group for all objects
		tlas.emplace_back(rayInst);
		}
		self.rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
	
	def createRtDescriptorSet():
		# Top-level acceleration structure, usable by both the ray generation and the closest hit (to shoot shadow rays)
		self.rtDescSetLayoutBind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
									   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);  # TLAS
		self.rtDescSetLayoutBind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
									   VK_SHADER_STAGE_RAYGEN_BIT_KHR);  # Output image

		self.rtDescPool      = self.rtDescSetLayoutBind.createPool(self.device);
		self.rtDescSetLayout = self.rtDescSetLayoutBind.createLayout(self.device);

		VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		allocateInfo.descriptorPool     = self.rtDescPool;
		allocateInfo.descriptorSetCount = 1;
		allocateInfo.pSetLayouts        = &self.rtDescSetLayout;
		vkAllocateDescriptorSets(self.device, &allocateInfo, &self.rtDescSet);


		VkAccelerationStructureKHR                   tlas = self.rtBuilder.getAccelerationStructure();
		VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
		descASInfo.accelerationStructureCount = 1;
		descASInfo.pAccelerationStructures    = &tlas;
		VkDescriptorImageInfo imageInfo{{}, self.offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};

		std::vector<VkWriteDescriptorSet> writes;
		writes.emplace_back(self.rtDescSetLayoutBind.makeWrite(self.rtDescSet, RtxBindings::eTlas, &descASInfo));
		writes.emplace_back(self.rtDescSetLayoutBind.makeWrite(self.rtDescSet, RtxBindings::eOutImage, &imageInfo));
		vkUpdateDescriptorSets(self.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);