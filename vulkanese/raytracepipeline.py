import ctypes
import os
import time
import json
from vulkan import *
from surface import *
from shader import *
from renderpass import *
from pipeline import *
from commandbuffer import *
from vutil import *
from vulkanese import *
from PIL import Image as pilImage

here = os.path.dirname(os.path.abspath(__file__))
def getVulkanesePath():
	return here

from enum import Enum
class StageIndices(Enum):
	eRaygen           = 0
	eMiss             = 1
	eMiss2            = 2
	eClosestHit       = 3
	eShaderGroupCount = 4
	
class RaytracePipeline(Pipeline):
	def __init__():

		# All stages
		std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
		VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
		stage.pName = "main";  # All the same entry point
		# Raygen
		stage.module = nvvk::createShaderModule(self.device, nvh::loadFile("spv/raytrace.rgen.spv", true, defaultSearchPaths, true));
		stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
		stages[eRaygen] = stage;
		# Miss
		stage.module = nvvk::createShaderModule(self.device, nvh::loadFile("spv/raytrace.rmiss.spv", true, defaultSearchPaths, true));
		stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
		stages[eMiss] = stage;
		# The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
		stage.module =
		  nvvk::createShaderModule(self.device, nvh::loadFile("spv/raytraceShadow.rmiss.spv", true, defaultSearchPaths, true));
		stage.stage    = VK_SHADER_STAGE_MISS_BIT_KHR;
		stages[eMiss2] = stage;
		# Hit Group - Closest Hit
		stage.module = nvvk::createShaderModule(self.device, nvh::loadFile("spv/raytrace.rchit.spv", true, defaultSearchPaths, true));
		stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
		stages[eClosestHit] = stage;


		# Shader groups
		VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
		group.anyHitShader       = VK_SHADER_UNUSED_KHR;
		group.closestHitShader   = VK_SHADER_UNUSED_KHR;
		group.generalShader      = VK_SHADER_UNUSED_KHR;
		group.intersectionShader = VK_SHADER_UNUSED_KHR;

		# Raygen
		group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		group.generalShader = eRaygen;
		self.rtShaderGroups.push_back(group);

		# Miss
		group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		group.generalShader = eMiss;
		self.rtShaderGroups.push_back(group);

		# Shadow Miss
		group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		group.generalShader = eMiss2;
		self.rtShaderGroups.push_back(group);

		# closest hit shader
		group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
		group.generalShader    = VK_SHADER_UNUSED_KHR;
		group.closestHitShader = eClosestHit;
		self.rtShaderGroups.push_back(group);

		# Push constant: we want to be able to update constants used by the shaders
		VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
									   0, sizeof(PushConstantRay)};


		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

		# Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
		std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {self.rtDescSetLayout, self.descSetLayout};
		pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
		pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();

		vkCreatePipelineLayout(self.device, &pipelineLayoutCreateInfo, nullptr, &self.rtPipelineLayout);


		# Assemble the shader stages and recursion depth info into the ray tracing pipeline
		VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
		rayPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());  # Stages are shaders
		rayPipelineInfo.pStages    = stages.data();

		# In this case, self.rtShaderGroups.size() == 4: we have one raygen group,
		# two miss shader groups, and one hit group.
		rayPipelineInfo.groupCount = static_cast<uint32_t>(self.rtShaderGroups.size());
		rayPipelineInfo.pGroups    = self.rtShaderGroups.data();

		# The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
		# hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
		# as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
		# in the ray generation to avoid deep recursion.
		rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  # Ray depth
		rayPipelineInfo.layout                       = self.rtPipelineLayout;

		vkCreateRayTracingPipelinesKHR(self.device, {}, {}, 1, &rayPipelineInfo, nullptr, &self.rtPipeline);


		for(auto& s : stages)
			vkDestroyShaderModule(self.device, s.module, nullptr);
			
	def onResize(w, h):
		createOffscreenRender();
		updatePostDescriptorSet();
		updateRtDescriptorSet();

	def createOffscreenRender():
		{
		self.alloc.destroy(self.offscreenColor);
		self.alloc.destroy(self.offscreenDepth);

		# Creating the color image
		{
		auto colorCreateInfo = nvvk::makeImage2DCreateInfo(self.size, self.offscreenColorFormat,
													   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
														   | VK_IMAGE_USAGE_STORAGE_BIT);


		nvvk::Image           image  = self.alloc.createImage(colorCreateInfo);
		VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
		VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
		self.offscreenColor                        = self.alloc.createTexture(image, ivInfo, sampler);
		self.offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		}

		# Creating the depth buffer
		auto depthCreateInfo = nvvk::makeImage2DCreateInfo(self.size, self.offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
		{
		nvvk::Image image = self.alloc.createImage(depthCreateInfo);


		depthStencilView = VkImageViewCreateInfo(#{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
			viewType         = VK_IMAGE_VIEW_TYPE_2D,
			format           = self.offscreenDepthFormat,
			subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
			image            = image.image
			)

		self.offscreenDepth = self.alloc.createTexture(image, depthStencilView);
		}

		# Setting the image layout for both color and depth
		{
		nvvk::cmdBarrierImageLayout(cmdBuf, self.offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, self.offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
								VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

		genCmdBuf.submitAndWait(cmdBuf);
		}

		# Creating a renderpass for the offscreen
		if(!self.offscreenRenderPass)
		{
		self.offscreenRenderPass = nvvk::createRenderPass(self.device, {self.offscreenColorFormat}, self.offscreenDepthFormat, 1, true,
												   true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
		}


		# Creating the frame buffer for offscreen
		std::vector<VkImageView> attachments = {self.offscreenColor.descriptor.imageView, self.offscreenDepth.descriptor.imageView};

		vkDestroyFramebuffer(self.device, self.offscreenFramebuffer, nullptr);
		info = VkFramebufferCreateInfo( #{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
			renderPass      = self.offscreenRenderPass,
			attachmentCount = 2,
			pAttachments    = attachments.data(),
			width           = self.size.width,
			height          = self.size.height,
			layers          = 1)
		vkCreateFramebuffer(self.device, &info, nullptr, &self.offscreenFramebuffer);
		}


	
	def createPostPipeline():
		# Push constants in the fragment shader
		pushConstantRanges = VkPushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float));

		# Creating the pipeline layout
		#{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		createInfo = VkPipelineLayoutCreateInfo( 
		createInfo.setLayoutCount         = 1,
		createInfo.pSetLayouts            = &self.postDescSetLayout,
		createInfo.pushConstantRangeCount = 1,
		createInfo.pPushConstantRanges    = &pushConstantRanges)
		
		vkCreatePipelineLayout(self.device, &createInfo, nullptr, &self.postPipelineLayout);


		# Pipeline: completely generic, no vertices
		nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(self.device, self.postPipelineLayout, self.renderPass);
		pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_VERTEX_BIT);
		pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
		pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
		self.postPipeline                                = pipelineGenerator.createPipeline();
		self.debug.setObjectName(self.postPipeline, "post");
  
  
	def initRayTracing()
		# Requesting ray tracing properties
		VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
		prop2.pNext = &self.rtProperties;
		vkGetPhysicalDeviceProperties2(self.physicalDevice, &prop2);

		self.rtBuilder.setup(self.device, &self.alloc, self.graphicsQueueIndex)
	

	
	def updateRtDescriptorSet():
		# (1) Output buffer
		VkDescriptorImageInfo imageInfo{{}, self.offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
		VkWriteDescriptorSet  wds = self.rtDescSetLayoutBind.makeWrite(self.rtDescSet, RtxBindings::eOutImage, &imageInfo);
		vkUpdateDescriptorSets(self.device, 1, &wds, 0, nullptr);
		
		
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
		hSizeAligned = nvh::align_up(handleSize, self.rtProperties.shaderGroupHandleAlignment);

		self.rgenRegion = VkStridedDeviceAddressRegionKHR(
			stride = n::align_up(hSizeAligned, self.rtProperties.shaderGroupBaseAlignment),
			size   = self.rgenRegion.stride;  # The size member of pRayGenShaderBindingTable must be equal to its stride member
		);
		self.missRegion = VkStridedDeviceAddressRegionKHR(
			stride = hSizeAligned,
			size   = nvh::align_up(missCount * hSizeAligned, self.rtProperties.shaderGroupBaseAlignment))
			
		self.hitRegion = VkStridedDeviceAddressRegionKHR(
			stride = hSizeAligned;
			size   = nvh::align_up(hitCount * hSizeAligned, self.rtProperties.shaderGroupBaseAlignment))

		# Get the shader group handles
		dataSize = handleCount * handleSize;
		std::vector<uint8_t> handles(dataSize);
		auto result = vkGetRayTracingShaderGroupHandlesKHR(self.device, self.rtPipeline, 0, handleCount, dataSize, handles.data());
		assert(result == VK_SUCCESS);

		# Allocate a buffer for storing the SBT.
		VkDeviceSize sbtSize = self.rgenRegion.size + self.missRegion.size + self.hitRegion.size + self.callRegion.size;
		self.rtSBTBuffer = \
		Buffer(sbtSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		);

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