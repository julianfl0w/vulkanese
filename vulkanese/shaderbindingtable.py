import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *

class ShaderBindingTable(PrintClass):
	def __init__(self, pipeline):
		PrintClass.__init__(self)
		self.pipeline = pipeline
		self.pipelineDict = pipeline.setupDict
		self.vkCommandPool  = pipeline.device.vkCommandPool
		self.device       = pipeline.device
		self.vkDevice     = pipeline.device.vkDevice
		self.outputWidthPixels  = self.pipelineDict["outputWidthPixels"]
		self.outputHeightPixels = self.pipelineDict["outputHeightPixels"]
		self.commandBufferCount = 0
		
		missCount = 2
		hitCount  = 1
		handleCount = 1 + missCount + hitCount;
		handleSize  = self.rtProperties.shaderGroupHandleSize;

		# The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
		handleSizeAligned = nvh::align_up(handleSize, self.rtProperties.shaderGroupHandleAlignment);

		self.rgenRegion.stride = nvh::align_up(handleSizeAligned, self.rtProperties.shaderGroupBaseAlignment);
		self.rgenRegion.size   = self.rgenRegion.stride;  # The size member of pRayGenShaderBindingTable must be equal to its stride member
		self.missRegion.stride = handleSizeAligned;
		self.missRegion.size   = nvh::align_up(missCount * handleSizeAligned, self.rtProperties.shaderGroupBaseAlignment);
		self.hitRegion.stride  = handleSizeAligned;
		self.hitRegion.size    = nvh::align_up(hitCount * handleSizeAligned, self.rtProperties.shaderGroupBaseAlignment);

		# Get the shader group handles
		            dataSize = handleCount * handleSize;
		std::vector<uint8_t> handles(dataSize);
		auto result = vkGetRayTracingShaderGroupHandlesKHR(self.device, self.rtPipeline, 0, handleCount, dataSize, handles.data());
		assert(result == VK_SUCCESS);

		# Allocate a buffer for storing the SBT.
		VkDeviceSize sbtSize = self.rgenRegion.size + self.missRegion.size + self.hitRegion.size + self.callRegion.size;
		self.rtSBTBuffer        = self.alloc.createBuffer(sbtSize,
										   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
											   | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
										   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		self.debug.setObjectName(self.rtSBTBuffer.buffer, std::string("SBT"));  # Give it a debug name for NSight.

		# Find the SBT addresses of each group
		VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, self.rtSBTBuffer.buffer};
		VkDeviceAddress           sbtAddress = vkGetBufferDeviceAddress(self.device, &info);
		self.rgenRegion.deviceAddress           = sbtAddress;
		self.missRegion.deviceAddress           = sbtAddress + self.rgenRegion.size;
		self.hitRegion.deviceAddress            = sbtAddress + self.rgenRegion.size + self.missRegion.size;

		# Helper to retrieve the handle data
		auto getHandle = [&](int i) { return handles.data() + i * handleSize; };

		# Map the SBT buffer and write in the handles.
		auto*    pSBTBuffer = reinterpret_cast<uint8_t*>(self.alloc.map(self.rtSBTBuffer));
		uint8_t* pData{nullptr};
		handleIdx{0};
		# Raygen
		pData = pSBTBuffer;
		memcpy(pData, getHandle(handleIdx++), handleSize);
		# Miss
		pData = pSBTBuffer + self.rgenRegion.size;
		for(c = 0; c < missCount; c++)
		{
		memcpy(pData, getHandle(handleIdx++), handleSize);
		pData += self.missRegion.stride;
		}
		# Hit
		pData = pSBTBuffer + self.rgenRegion.size + self.missRegion.size;
		for(c = 0; c < hitCount; c++)
		{
		memcpy(pData, getHandle(handleIdx++), handleSize);
		pData += self.hitRegion.stride;
		}

		self.alloc.unmap(self.rtSBTBuffer);
		self.alloc.finalizeAndReleaseStaging();