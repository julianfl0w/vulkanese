import json
from vutil import *
import os
here = os.path.dirname(os.path.abspath(__file__))
from vulkan import *

class DescriptorPool(Sinode):
	def __init__(self, device, MAX_FRAMES_IN_FLIGHT = 3):
		Sinode.__init__(self, device)
		self.vkDevice = device.vkDevice
		self.MAX_FRAMES_IN_FLIGHT = MAX_FRAMES_IN_FLIGHT
		
		self.descSetDict = {}
		#The descriptor set number 0 will be used for engine-global resources, and bound once per frame.
		self.descSetDict["global"] = DescriptorSet(self, binding = 0)
		#The descriptor set number 1 will be used for per-pass resources, and bound once per pass. 
		self.descSetDict["perPass"]   = DescriptorSet(self, binding = 1)
		
		#The descriptor set number 2 will be used for material resources, 
		self.descSetDict["material"]  = DescriptorSet(self, binding = 2)
		
		#and the number 3 will be used for per-object resources. 
		self.descSetDict["perObject"] = DescriptorSet(self, binding = 3)
	
	def getBinding(self, buffer, bindname):
		print("Allocating ")
		print(buffer)
		print(bindname)
		return self.descSetDict[bindname].attachBuffer(buffer)
		
	#We first need to describe which descriptor types our descriptor sets are going to contain and how many of them, using VkDescriptorPoolSize structures.
	def finalize(self):
		# create descriptor pool.
		self.poolSize = VkDescriptorPoolSize(
			type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			descriptorCount = 4)

		#We will allocate one of these descriptors for every frame. This pool size structure is referenced by the main VkDescriptorPoolCreateInfo:
		#Aside from the maximum number of individual descriptors that are available, we also need to specify the maximum number of descriptor sets that may be allocated:
		self.poolInfo = VkDescriptorPoolCreateInfo(
			sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			poolSizeCount = 1,
			pPoolSizes = [self.poolSize],
			maxSets = 4) # imposed by some gpus 
		
		#The structure has an optional flag similar to command pools that determines if individual descriptor sets can be freed or not: VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT. We're not going to touch the descriptor set after creating it, so we don't need this flag. You can leave flags to its default value of 0.
		self.vkDescriptorPool = vkCreateDescriptorPool(self.vkDevice, [self.poolInfo], 0)
		
		#This way, the inner render loops will only be binding descriptor sets 2 and 3, and performance will be high.
		for descriptor in self.descSetDict.values():
			descriptor.finalize()
			
			
		# Establish the create info
		descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
			sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			descriptorPool=self.vkDescriptorPool,
			descriptorSetCount=len(self.descSetDict.values()),
			pSetLayouts=[d.vkCreateDescriptorSetLayout for d in self.descSetDict.values()]
		)
		
		# create the allocate descriptor set.
		self.vkDescriptorSets = vkAllocateDescriptorSets(self.vkDevice, descriptorSetAllocateInfo)
		
		for i, d in enumerate(self.descSetDict.values()):
			d.vkDescriptorSet = self.vkDescriptorSets[i]
			
		#self.writeDescriptorSet = VkWriteDescriptorSet(
		#	sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		#	dstSet=self.vkDescriptorSet,
		#	dstBinding=0,  # write to the first, and only binding.
		#	descriptorCount=1,
		#	descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		#	pBufferInfo=[buffer.descriptorBufferInfo for buffer in self.buffers.values()]
		#)

class DescriptorSet(Sinode):
	def __init__(self, descriptorPool, binding, MAX_FRAMES_IN_FLIGHT = 3):
		Sinode.__init__(self, descriptorPool)
		self.vkDevice = descriptorPool.vkDevice
		self.descriptorPool = descriptorPool
		self.buffers = {}
		
	def attachBuffer(self, buffer):
		# this gets set in buffer
		#buffer.binding = len(self.buffers.values())
		thisIndex = len(self.buffers.values())
		self.buffers[buffer.setupDict["name"]] = buffer
		return thisIndex
		
	def finalize(self):
		# Here we specify a descriptor set layout. This allows us to bind our descriptors to
		# resources in the shader.

		# Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
		# 0. This binds to
		#   layout(std140, binding = 0) buffer buf
		# in the compute shader.
		
		for buffname, buffer in self.buffers.items():
			flags = 0
			try:
				flags = eval(buffer.parent.setupDict["stage"])
			except:
				pass
			buffer.descriptorSetLayoutBinding = VkDescriptorSetLayoutBinding(
				binding=buffer.binding,
				descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				descriptorCount=1,
				stageFlags=flags
				)


		# Establish the create info
		descriptorSetLayoutCreateInfo = VkDescriptorSetLayoutCreateInfo(
			sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			bindingCount=len(self.buffers.values()),  
			pBindings=[buffer.descriptorSetLayoutBinding for buffer in self.buffers.values()]
		)

		# Create the descriptor set layout.
		self.vkCreateDescriptorSetLayout = vkCreateDescriptorSetLayout(self.vkDevice, descriptorSetLayoutCreateInfo, None)


		# perform the update of the descriptor set.
		#vkUpdateDescriptorSets(self.vkDevice, 1, [self.writeDescriptorSet], 0, None)

	def release():
		
		vkDestroyDescriptorSetLayout(self.device, self.descriptorSetLayout, None)
		vkDestroyDescriptorSet(self.device, self.descriptorSet, None)

#	def updateDescriptorSet():
#		std::vector<VkWriteDescriptorSet> writes;
#
#		# Camera matrices and scene description
#		VkDescriptorBufferInfo dbiUnif{self.bGlobals.buffer, 0, VK_WHOLE_SIZE};
#		writes += [self.descSetLayoutBind.makeWrite(self.descSet, "engineGlobal", &dbiUnif));
#
#		VkDescriptorBufferInfo dbiSceneDesc{self.bObjDesc.buffer, 0, VK_WHOLE_SIZE};
#		writes += [self.descSetLayoutBind.makeWrite(self.descSet, "perObject", &dbiSceneDesc));
#
#		# All texture samplers
#		std::vector<VkDescriptorImageInfo> diit;
#		for(auto& texture : self.textures)
#		{
#		diit += [texture.descriptor);
#		}
#		writes += [self.descSetLayoutBind.makeWriteArray(self.descSet, "perPass", diit.data()));
#
#		# Writing the information
#		vkUpdateDescriptorSets(self.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

#	def updatePostDescriptorSet():
#		VkWriteDescriptorSet writeDescriptorSets = self.postDescSetLayoutBind.makeWrite(self.postDescSet, 0, &self.offscreenColor.descriptor);
#		vkUpdateDescriptorSets(self.device, 1, &writeDescriptorSets, 0, nullptr);
		
#	def createPostDescriptor():
#		self.postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
#		self.postDescSetLayout = self.postDescSetLayoutBind.createLayout(self.device);
#		self.postDescPool      = self.postDescSetLayoutBind.createPool(self.device);
#		self.postDescSet       = nvvk::allocateDescriptorSet(self.device, self.postDescPool, self.postDescSetLayout)