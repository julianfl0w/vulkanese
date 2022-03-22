import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *

class DescriptorPool(Sinode):
	def __init__(self, pipeline, MAX_FRAMES_IN_FLIGHT = 3):
		Sinode.__init__(self, pipeline)
		self.pipeline = pipeline
		self.pipelineDict  = pipeline.setupDict
		self.vkCommandPool = pipeline.device.vkCommandPool
		self.device        = pipeline.device
		self.vkDevice      = pipeline.device.vkDevice
		
		#We first need to describe which descriptor types our descriptor sets are going to contain and how many of them, using VkDescriptorPoolSize structures.
		
		self.poolSize = VkDescriptorPoolSize(
			type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			descriptorCount = MAX_FRAMES_IN_FLIGHT)

		#We will allocate one of these descriptors for every frame. This pool size structure is referenced by the main VkDescriptorPoolCreateInfo:
		#Aside from the maximum number of individual descriptors that are available, we also need to specify the maximum number of descriptor sets that may be allocated:
		self.poolInfo = VkDescriptorPoolCreateInfo(
			sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			poolSizeCount = 1,
			pPoolSizes = [self.poolSize],
			maxSets = MAX_FRAMES_IN_FLIGHT)


		#The structure has an optional flag similar to command pools that determines if individual descriptor sets can be freed or not: VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT. We're not going to touch the descriptor set after creating it, so we don't need this flag. You can leave flags to its default value of 0.

		self.vkDescriptorPool = VkCreateDescriptorPool(self.device, [poolInfo], 0)
		
		#The descriptor set number 0 will be used for engine-global resources, and bound once per frame.
		self.engineGlobalDS = DescriptorSet(self, binding = 0)
		#The descriptor set number 1 will be used for per-pass resources, and bound once per pass. 
		self.PerPassDS   = DescriptorSet(self, binding = 1)
		
		#The descriptor set number 2 will be used for material resources, 
		self.materialDS  = DescriptorSet(self, binding = 2)
		
		#and the number 3 will be used for per-object resources. 
		self.perObjectDS = DescriptorSet(self, binding = 3)
		
		#This way, the inner render loops will only be binding descriptor sets 2 and 3, and performance will be high.

class DescriptorSet(Sinode):
	def __init__(self, descriptorPool, binding, MAX_FRAMES_IN_FLIGHT = 3):
		Sinode.__init__(self, descriptorPool)

		self.descriptorPool = descriptorPool
		# Here we specify a descriptor set layout. This allows us to bind our descriptors to
		# resources in the shader.

		# Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
		# 0. This binds to
		#   layout(std140, binding = 0) buffer buf
		# in the compute shader.

		self.descriptorSetLayoutBinding = VkDescriptorSetLayoutBinding(
			binding=binding,
			descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			descriptorCount=1,
			stageFlags=VK_SHADER_STAGE_COMPUTE_BIT
		)

		descriptorSetLayoutCreateInfo = VkDescriptorSetLayoutCreateInfo(
			sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			bindingCount=1,  # only a single binding in this descriptor set layout.
			pBindings=self.descriptorSetLayoutBinding
		)

		# Create the descriptor set layout.
		self.vkCreateDescriptorSetLayout = vkCreateDescriptorSetLayout(self.vkDevice, descriptorSetLayoutCreateInfo, None)

		# So we will allocate a descriptor set here.
		descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
			sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			descriptorPool=self.vkDescriptorPool,
			descriptorSetCount=1,
			pSetLayouts=[self.vkCreateDescriptorSetLayout]
		)

		# allocate descriptor set.
		self.vkDescriptorSet = vkAllocateDescriptorSets(self.vkDevice, descriptorSetAllocateInfo)[0]
		self.children += [self.vkDescriptorSet]

		# Next, we need to connect our actual storage buffer with the descrptor.
		# We use vkUpdateDescriptorSets() to update the descriptor set.
		self.descriptorBufferInfo = VkDescriptorBufferInfo(
			buffer=self.vkBuffer,
			offset=0,
			range=setupDict["SIZEBYTES"]
		)
		
		self.writeDescriptorSet = VkWriteDescriptorSet(
			sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			dstSet=self.vkDescriptorSet,
			dstBinding=0,  # write to the first, and only binding.
			descriptorCount=1,
			descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			pBufferInfo=descriptorBufferInfo
		)

		# perform the update of the descriptor set.
		vkUpdateDescriptorSets(self.vkDevice, 1, [writeDescriptorSet], 0, None)

	def release():
		
		vkDestroyDescriptorSetLayout(self.device, self.descriptorSetLayout, None)
		vkDestroyDescriptorSet(self.device, self.descriptorSet, None)

	def updateDescriptorSet():
		std::vector<VkWriteDescriptorSet> writes;

		# Camera matrices and scene description
		VkDescriptorBufferInfo dbiUnif{self.bGlobals.buffer, 0, VK_WHOLE_SIZE};
		writes += [self.descSetLayoutBind.makeWrite(self.descSet, SceneBindings::eGlobals, &dbiUnif));

		VkDescriptorBufferInfo dbiSceneDesc{self.bObjDesc.buffer, 0, VK_WHOLE_SIZE};
		writes += [self.descSetLayoutBind.makeWrite(self.descSet, SceneBindings::eObjDescs, &dbiSceneDesc));

		# All texture samplers
		std::vector<VkDescriptorImageInfo> diit;
		for(auto& texture : self.textures)
		{
		diit += [texture.descriptor);
		}
		writes += [self.descSetLayoutBind.makeWriteArray(self.descSet, SceneBindings::eTextures, diit.data()));

		# Writing the information
		vkUpdateDescriptorSets(self.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	def updatePostDescriptorSet():
		VkWriteDescriptorSet writeDescriptorSets = self.postDescSetLayoutBind.makeWrite(self.postDescSet, 0, &self.offscreenColor.descriptor);
		vkUpdateDescriptorSets(self.device, 1, &writeDescriptorSets, 0, nullptr);
		
	def createPostDescriptor():
		self.postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
		self.postDescSetLayout = self.postDescSetLayoutBind.createLayout(self.device);
		self.postDescPool      = self.postDescSetLayoutBind.createPool(self.device);
		self.postDescSet       = nvvk::allocateDescriptorSet(self.device, self.postDescPool, self.postDescSetLayout)