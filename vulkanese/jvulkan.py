
from enum import Enum
from ctypes import *
jvulkanLib = CDLL("/home/julian/Documents/sodll/vulkan/libvulkan.so") 

def cdataStr(instr):
	return ffi.new("char[]", instr.encode('ascii'))

# Generate Constants (ex VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR)
typeDef = {
  "__int128_t": "__int128",
  "__uint128_t": "unsigned __int128",
  "__NSConstantString": "struct __NSConstantString_tag",
  "__builtin_ms_va_list": "char *",
  "__builtin_va_list": "struct __va_list_tag [1]",
  "ptrdiff_t": "long",
  "size_t": "unsigned long",
  "wchar_t": "int",
  "max_align_t": "struct max_align_t",
  "__u_char": "unsigned char",
  "__u_short": "unsigned short",
  "__u_int": "unsigned int",
  "__u_long": "unsigned long",
  "__int8_t": "signed char",
  "__uint8_t": "unsigned char",
  "__int16_t": "short",
  "__uint16_t": "unsigned short",
  "__int32_t": "int",
  "__uint32_t": "unsigned int",
  "__int64_t": "long",
  "__uint64_t": "unsigned long",
  "__int_least8_t": "__int8_t",
  "__uint_least8_t": "__uint8_t",
  "__int_least16_t": "__int16_t",
  "__uint_least16_t": "__uint16_t",
  "__int_least32_t": "__int32_t",
  "__uint_least32_t": "__uint32_t",
  "__int_least64_t": "__int64_t",
  "__uint_least64_t": "__uint64_t",
  "__quad_t": "long",
  "__u_quad_t": "unsigned long",
  "__intmax_t": "long",
  "__uintmax_t": "unsigned long",
  "__dev_t": "unsigned long",
  "__uid_t": "unsigned int",
  "__gid_t": "unsigned int",
  "__ino_t": "unsigned long",
  "__ino64_t": "unsigned long",
  "__mode_t": "unsigned int",
  "__nlink_t": "unsigned long",
  "__off_t": "long",
  "__off64_t": "long",
  "__pid_t": "int",
  "__fsid_t": "struct __fsid_t",
  "__clock_t": "long",
  "__rlim_t": "unsigned long",
  "__rlim64_t": "unsigned long",
  "__id_t": "unsigned int",
  "__time_t": "long",
  "__useconds_t": "unsigned int",
  "__suseconds_t": "long",
  "__daddr_t": "int",
  "__key_t": "int",
  "__clockid_t": "int",
  "__timer_t": "void *",
  "__blksize_t": "long",
  "__blkcnt_t": "long",
  "__blkcnt64_t": "long",
  "__fsblkcnt_t": "unsigned long",
  "__fsblkcnt64_t": "unsigned long",
  "__fsfilcnt_t": "unsigned long",
  "__fsfilcnt64_t": "unsigned long",
  "__fsword_t": "long",
  "__ssize_t": "long",
  "__syscall_slong_t": "long",
  "__syscall_ulong_t": "unsigned long",
  "__loff_t": "__off64_t",
  "__caddr_t": "char *",
  "__intptr_t": "long",
  "__socklen_t": "unsigned int",
  "__sig_atomic_t": "int",
  "int8_t": "__int8_t",
  "int16_t": "__int16_t",
  "int32_t": "__int32_t",
  "int64_t": "__int64_t",
  "uint8_t": "__uint8_t",
  "uint16_t": "__uint16_t",
  "uint32_t": "__uint32_t",
  "uint64_t": "__uint64_t",
  "int_least8_t": "__int_least8_t",
  "int_least16_t": "__int_least16_t",
  "int_least32_t": "__int_least32_t",
  "int_least64_t": "__int_least64_t",
  "uint_least8_t": "__uint_least8_t",
  "uint_least16_t": "__uint_least16_t",
  "uint_least32_t": "__uint_least32_t",
  "uint_least64_t": "__uint_least64_t",
  "int_fast8_t": "signed char",
  "int_fast16_t": "long",
  "int_fast32_t": "long",
  "int_fast64_t": "long",
  "uint_fast8_t": "unsigned char",
  "uint_fast16_t": "unsigned long",
  "uint_fast32_t": "unsigned long",
  "uint_fast64_t": "unsigned long",
  "intptr_t": "long",
  "uintptr_t": "unsigned long",
  "intmax_t": "__intmax_t",
  "uintmax_t": "__uintmax_t",
  "VkBool32": "uint32_t",
  "VkDeviceAddress": "uint64_t",
  "VkDeviceSize": "uint64_t",
  "VkFlags": "uint32_t",
  "VkSampleMask": "uint32_t",
  "VkBuffer": "struct VkBuffer_T *",
  "VkImage": "struct VkImage_T *",
  "VkInstance": "struct VkInstance_T *",
  "VkPhysicalDevice": "struct VkPhysicalDevice_T *",
  "VkDevice": "struct VkDevice_T *",
  "VkQueue": "struct VkQueue_T *",
  "VkSemaphore": "struct VkSemaphore_T *",
  "VkCommandBuffer": "struct VkCommandBuffer_T *",
  "VkFence": "struct VkFence_T *",
  "VkDeviceMemory": "struct VkDeviceMemory_T *",
  "VkEvent": "struct VkEvent_T *",
  "VkQueryPool": "struct VkQueryPool_T *",
  "VkBufferView": "struct VkBufferView_T *",
  "VkImageView": "struct VkImageView_T *",
  "VkShaderModule": "struct VkShaderModule_T *",
  "VkPipelineCache": "struct VkPipelineCache_T *",
  "VkPipelineLayout": "struct VkPipelineLayout_T *",
  "VkPipeline": "struct VkPipeline_T *",
  "VkRenderPass": "struct VkRenderPass_T *",
  "VkDescriptorSetLayout": "struct VkDescriptorSetLayout_T *",
  "VkSampler": "struct VkSampler_T *",
  "VkDescriptorSet": "struct VkDescriptorSet_T *",
  "VkDescriptorPool": "struct VkDescriptorPool_T *",
  "VkFramebuffer": "struct VkFramebuffer_T *",
  "VkCommandPool": "struct VkCommandPool_T *",
  "VkResult": "enum VkResult",
  "VkStructureType": "enum VkStructureType",
  "VkImageLayout": "enum VkImageLayout",
  "VkObjectType": "enum VkObjectType",
  "VkPipelineCacheHeaderVersion": "enum VkPipelineCacheHeaderVersion",
  "VkVendorId": "enum VkVendorId",
  "VkSystemAllocationScope": "enum VkSystemAllocationScope",
  "VkInternalAllocationType": "enum VkInternalAllocationType",
  "VkFormat": "enum VkFormat",
  "VkImageTiling": "enum VkImageTiling",
  "VkImageType": "enum VkImageType",
  "VkPhysicalDeviceType": "enum VkPhysicalDeviceType",
  "VkQueryType": "enum VkQueryType",
  "VkSharingMode": "enum VkSharingMode",
  "VkComponentSwizzle": "enum VkComponentSwizzle",
  "VkImageViewType": "enum VkImageViewType",
  "VkBlendFactor": "enum VkBlendFactor",
  "VkBlendOp": "enum VkBlendOp",
  "VkCompareOp": "enum VkCompareOp",
  "VkDynamicState": "enum VkDynamicState",
  "VkFrontFace": "enum VkFrontFace",
  "VkVertexInputRate": "enum VkVertexInputRate",
  "VkPrimitiveTopology": "enum VkPrimitiveTopology",
  "VkPolygonMode": "enum VkPolygonMode",
  "VkStencilOp": "enum VkStencilOp",
  "VkLogicOp": "enum VkLogicOp",
  "VkBorderColor": "enum VkBorderColor",
  "VkFilter": "enum VkFilter",
  "VkSamplerAddressMode": "enum VkSamplerAddressMode",
  "VkSamplerMipmapMode": "enum VkSamplerMipmapMode",
  "VkDescriptorType": "enum VkDescriptorType",
  "VkAttachmentLoadOp": "enum VkAttachmentLoadOp",
  "VkAttachmentStoreOp": "enum VkAttachmentStoreOp",
  "VkPipelineBindPoint": "enum VkPipelineBindPoint",
  "VkCommandBufferLevel": "enum VkCommandBufferLevel",
  "VkIndexType": "enum VkIndexType",
  "VkSubpassContents": "enum VkSubpassContents",
  "VkAccessFlagBits": "enum VkAccessFlagBits",
  "VkAccessFlags": "VkFlags",
  "VkImageAspectFlagBits": "enum VkImageAspectFlagBits",
  "VkImageAspectFlags": "VkFlags",
  "VkFormatFeatureFlagBits": "enum VkFormatFeatureFlagBits",
  "VkFormatFeatureFlags": "VkFlags",
  "VkImageCreateFlagBits": "enum VkImageCreateFlagBits",
  "VkImageCreateFlags": "VkFlags",
  "VkSampleCountFlagBits": "enum VkSampleCountFlagBits",
  "VkSampleCountFlags": "VkFlags",
  "VkImageUsageFlagBits": "enum VkImageUsageFlagBits",
  "VkImageUsageFlags": "VkFlags",
  "VkInstanceCreateFlagBits": "enum VkInstanceCreateFlagBits",
  "VkInstanceCreateFlags": "VkFlags",
  "VkMemoryHeapFlagBits": "enum VkMemoryHeapFlagBits",
  "VkMemoryHeapFlags": "VkFlags",
  "VkMemoryPropertyFlagBits": "enum VkMemoryPropertyFlagBits",
  "VkMemoryPropertyFlags": "VkFlags",
  "VkQueueFlagBits": "enum VkQueueFlagBits",
  "VkQueueFlags": "VkFlags",
  "VkDeviceCreateFlags": "VkFlags",
  "VkDeviceQueueCreateFlagBits": "enum VkDeviceQueueCreateFlagBits",
  "VkDeviceQueueCreateFlags": "VkFlags",
  "VkPipelineStageFlagBits": "enum VkPipelineStageFlagBits",
  "VkPipelineStageFlags": "VkFlags",
  "VkMemoryMapFlags": "VkFlags",
  "VkSparseMemoryBindFlagBits": "enum VkSparseMemoryBindFlagBits",
  "VkSparseMemoryBindFlags": "VkFlags",
  "VkSparseImageFormatFlagBits": "enum VkSparseImageFormatFlagBits",
  "VkSparseImageFormatFlags": "VkFlags",
  "VkFenceCreateFlagBits": "enum VkFenceCreateFlagBits",
  "VkFenceCreateFlags": "VkFlags",
  "VkSemaphoreCreateFlags": "VkFlags",
  "VkEventCreateFlagBits": "enum VkEventCreateFlagBits",
  "VkEventCreateFlags": "VkFlags",
  "VkQueryPipelineStatisticFlagBits": "enum VkQueryPipelineStatisticFlagBits",
  "VkQueryPipelineStatisticFlags": "VkFlags",
  "VkQueryPoolCreateFlags": "VkFlags",
  "VkQueryResultFlagBits": "enum VkQueryResultFlagBits",
  "VkQueryResultFlags": "VkFlags",
  "VkBufferCreateFlagBits": "enum VkBufferCreateFlagBits",
  "VkBufferCreateFlags": "VkFlags",
  "VkBufferUsageFlagBits": "enum VkBufferUsageFlagBits",
  "VkBufferUsageFlags": "VkFlags",
  "VkBufferViewCreateFlags": "VkFlags",
  "VkImageViewCreateFlagBits": "enum VkImageViewCreateFlagBits",
  "VkImageViewCreateFlags": "VkFlags",
  "VkShaderModuleCreateFlags": "VkFlags",
  "VkPipelineCacheCreateFlagBits": "enum VkPipelineCacheCreateFlagBits",
  "VkPipelineCacheCreateFlags": "VkFlags",
  "VkColorComponentFlagBits": "enum VkColorComponentFlagBits",
  "VkColorComponentFlags": "VkFlags",
  "VkPipelineCreateFlagBits": "enum VkPipelineCreateFlagBits",
  "VkPipelineCreateFlags": "VkFlags",
  "VkPipelineShaderStageCreateFlagBits": "enum VkPipelineShaderStageCreateFlagBits",
  "VkPipelineShaderStageCreateFlags": "VkFlags",
  "VkShaderStageFlagBits": "enum VkShaderStageFlagBits",
  "VkCullModeFlagBits": "enum VkCullModeFlagBits",
  "VkCullModeFlags": "VkFlags",
  "VkPipelineVertexInputStateCreateFlags": "VkFlags",
  "VkPipelineInputAssemblyStateCreateFlags": "VkFlags",
  "VkPipelineTessellationStateCreateFlags": "VkFlags",
  "VkPipelineViewportStateCreateFlags": "VkFlags",
  "VkPipelineRasterizationStateCreateFlags": "VkFlags",
  "VkPipelineMultisampleStateCreateFlags": "VkFlags",
  "VkPipelineDepthStencilStateCreateFlagBits": "enum VkPipelineDepthStencilStateCreateFlagBits",
  "VkPipelineDepthStencilStateCreateFlags": "VkFlags",
  "VkPipelineColorBlendStateCreateFlagBits": "enum VkPipelineColorBlendStateCreateFlagBits",
  "VkPipelineColorBlendStateCreateFlags": "VkFlags",
  "VkPipelineDynamicStateCreateFlags": "VkFlags",
  "VkPipelineLayoutCreateFlagBits": "enum VkPipelineLayoutCreateFlagBits",
  "VkPipelineLayoutCreateFlags": "VkFlags",
  "VkShaderStageFlags": "VkFlags",
  "VkSamplerCreateFlagBits": "enum VkSamplerCreateFlagBits",
  "VkSamplerCreateFlags": "VkFlags",
  "VkDescriptorPoolCreateFlagBits": "enum VkDescriptorPoolCreateFlagBits",
  "VkDescriptorPoolCreateFlags": "VkFlags",
  "VkDescriptorPoolResetFlags": "VkFlags",
  "VkDescriptorSetLayoutCreateFlagBits": "enum VkDescriptorSetLayoutCreateFlagBits",
  "VkDescriptorSetLayoutCreateFlags": "VkFlags",
  "VkAttachmentDescriptionFlagBits": "enum VkAttachmentDescriptionFlagBits",
  "VkAttachmentDescriptionFlags": "VkFlags",
  "VkDependencyFlagBits": "enum VkDependencyFlagBits",
  "VkDependencyFlags": "VkFlags",
  "VkFramebufferCreateFlagBits": "enum VkFramebufferCreateFlagBits",
  "VkFramebufferCreateFlags": "VkFlags",
  "VkRenderPassCreateFlagBits": "enum VkRenderPassCreateFlagBits",
  "VkRenderPassCreateFlags": "VkFlags",
  "VkSubpassDescriptionFlagBits": "enum VkSubpassDescriptionFlagBits",
  "VkSubpassDescriptionFlags": "VkFlags",
  "VkCommandPoolCreateFlagBits": "enum VkCommandPoolCreateFlagBits",
  "VkCommandPoolCreateFlags": "VkFlags",
  "VkCommandPoolResetFlagBits": "enum VkCommandPoolResetFlagBits",
  "VkCommandPoolResetFlags": "VkFlags",
  "VkCommandBufferUsageFlagBits": "enum VkCommandBufferUsageFlagBits",
  "VkCommandBufferUsageFlags": "VkFlags",
  "VkQueryControlFlagBits": "enum VkQueryControlFlagBits",
  "VkQueryControlFlags": "VkFlags",
  "VkCommandBufferResetFlagBits": "enum VkCommandBufferResetFlagBits",
  "VkCommandBufferResetFlags": "VkFlags",
  "VkStencilFaceFlagBits": "enum VkStencilFaceFlagBits",
  "VkStencilFaceFlags": "VkFlags",
  "VkExtent2D": "struct VkExtent2D",
  "VkExtent3D": "struct VkExtent3D",
  "VkOffset2D": "struct VkOffset2D",
  "VkOffset3D": "struct VkOffset3D",
  "VkRect2D": "struct VkRect2D",
  "VkBaseInStructure": "struct VkBaseInStructure",
  "VkBaseOutStructure": "struct VkBaseOutStructure",
  "VkBufferMemoryBarrier": "struct VkBufferMemoryBarrier",
  "VkDispatchIndirectCommand": "struct VkDispatchIndirectCommand",
  "VkDrawIndexedIndirectCommand": "struct VkDrawIndexedIndirectCommand",
  "VkDrawIndirectCommand": "struct VkDrawIndirectCommand",
  "VkImageSubresourceRange": "struct VkImageSubresourceRange",
  "VkImageMemoryBarrier": "struct VkImageMemoryBarrier",
  "VkMemoryBarrier": "struct VkMemoryBarrier",
  "VkPipelineCacheHeaderVersionOne": "struct VkPipelineCacheHeaderVersionOne",
  "PFN_vkAllocationFunction": "void *(*)(void *, size_t, size_t, VkSystemAllocationScope)",
  "PFN_vkFreeFunction": "void (*)(void *, void *)",
  "PFN_vkInternalAllocationNotification": "void (*)(void *, size_t, VkInternalAllocationType, VkSystemAllocationScope)",
  "PFN_vkInternalFreeNotification": "void (*)(void *, size_t, VkInternalAllocationType, VkSystemAllocationScope)",
  "PFN_vkReallocationFunction": "void *(*)(void *, void *, size_t, size_t, VkSystemAllocationScope)",
  "PFN_vkVoidFunction": "void (*)(void)",
  "VkAllocationCallbacks": "struct VkAllocationCallbacks",
  "VkApplicationInfo": "struct VkApplicationInfo",
  "VkFormatProperties": "struct VkFormatProperties",
  "VkImageFormatProperties": "struct VkImageFormatProperties",
  "VkInstanceCreateInfo": "struct VkInstanceCreateInfo",
  "VkMemoryHeap": "struct VkMemoryHeap",
  "VkMemoryType": "struct VkMemoryType",
  "VkPhysicalDeviceFeatures": "struct VkPhysicalDeviceFeatures",
  "VkPhysicalDeviceLimits": "struct VkPhysicalDeviceLimits",
  "VkPhysicalDeviceMemoryProperties": "struct VkPhysicalDeviceMemoryProperties",
  "VkPhysicalDeviceSparseProperties": "struct VkPhysicalDeviceSparseProperties",
  "VkPhysicalDeviceProperties": "struct VkPhysicalDeviceProperties",
  "VkQueueFamilyProperties": "struct VkQueueFamilyProperties",
  "VkDeviceQueueCreateInfo": "struct VkDeviceQueueCreateInfo",
  "VkDeviceCreateInfo": "struct VkDeviceCreateInfo",
  "VkExtensionProperties": "struct VkExtensionProperties",
  "VkLayerProperties": "struct VkLayerProperties",
  "VkSubmitInfo": "struct VkSubmitInfo",
  "VkMappedMemoryRange": "struct VkMappedMemoryRange",
  "VkMemoryAllocateInfo": "struct VkMemoryAllocateInfo",
  "VkMemoryRequirements": "struct VkMemoryRequirements",
  "VkSparseMemoryBind": "struct VkSparseMemoryBind",
  "VkSparseBufferMemoryBindInfo": "struct VkSparseBufferMemoryBindInfo",
  "VkSparseImageOpaqueMemoryBindInfo": "struct VkSparseImageOpaqueMemoryBindInfo",
  "VkImageSubresource": "struct VkImageSubresource",
  "VkSparseImageMemoryBind": "struct VkSparseImageMemoryBind",
  "VkSparseImageMemoryBindInfo": "struct VkSparseImageMemoryBindInfo",
  "VkBindSparseInfo": "struct VkBindSparseInfo",
  "VkSparseImageFormatProperties": "struct VkSparseImageFormatProperties",
  "VkSparseImageMemoryRequirements": "struct VkSparseImageMemoryRequirements",
  "VkFenceCreateInfo": "struct VkFenceCreateInfo",
  "VkSemaphoreCreateInfo": "struct VkSemaphoreCreateInfo",
  "VkEventCreateInfo": "struct VkEventCreateInfo",
  "VkQueryPoolCreateInfo": "struct VkQueryPoolCreateInfo",
  "VkBufferCreateInfo": "struct VkBufferCreateInfo",
  "VkBufferViewCreateInfo": "struct VkBufferViewCreateInfo",
  "VkImageCreateInfo": "struct VkImageCreateInfo",
  "VkSubresourceLayout": "struct VkSubresourceLayout",
  "VkComponentMapping": "struct VkComponentMapping",
  "VkImageViewCreateInfo": "struct VkImageViewCreateInfo",
  "VkShaderModuleCreateInfo": "struct VkShaderModuleCreateInfo",
  "VkPipelineCacheCreateInfo": "struct VkPipelineCacheCreateInfo",
  "VkSpecializationMapEntry": "struct VkSpecializationMapEntry",
  "VkSpecializationInfo": "struct VkSpecializationInfo",
  "VkPipelineShaderStageCreateInfo": "struct VkPipelineShaderStageCreateInfo",
  "VkComputePipelineCreateInfo": "struct VkComputePipelineCreateInfo",
  "VkVertexInputBindingDescription": "struct VkVertexInputBindingDescription",
  "VkVertexInputAttributeDescription": "struct VkVertexInputAttributeDescription",
  "VkPipelineVertexInputStateCreateInfo": "struct VkPipelineVertexInputStateCreateInfo",
  "VkPipelineInputAssemblyStateCreateInfo": "struct VkPipelineInputAssemblyStateCreateInfo",
  "VkPipelineTessellationStateCreateInfo": "struct VkPipelineTessellationStateCreateInfo",
  "VkViewport": "struct VkViewport",
  "VkPipelineViewportStateCreateInfo": "struct VkPipelineViewportStateCreateInfo",
  "VkPipelineRasterizationStateCreateInfo": "struct VkPipelineRasterizationStateCreateInfo",
  "VkPipelineMultisampleStateCreateInfo": "struct VkPipelineMultisampleStateCreateInfo",
  "VkStencilOpState": "struct VkStencilOpState",
  "VkPipelineDepthStencilStateCreateInfo": "struct VkPipelineDepthStencilStateCreateInfo",
  "VkPipelineColorBlendAttachmentState": "struct VkPipelineColorBlendAttachmentState",
  "VkPipelineColorBlendStateCreateInfo": "struct VkPipelineColorBlendStateCreateInfo",
  "VkPipelineDynamicStateCreateInfo": "struct VkPipelineDynamicStateCreateInfo",
  "VkGraphicsPipelineCreateInfo": "struct VkGraphicsPipelineCreateInfo",
  "VkPushConstantRange": "struct VkPushConstantRange",
  "VkPipelineLayoutCreateInfo": "struct VkPipelineLayoutCreateInfo",
  "VkSamplerCreateInfo": "struct VkSamplerCreateInfo",
  "VkCopyDescriptorSet": "struct VkCopyDescriptorSet",
  "VkDescriptorBufferInfo": "struct VkDescriptorBufferInfo",
  "VkDescriptorImageInfo": "struct VkDescriptorImageInfo",
  "VkDescriptorPoolSize": "struct VkDescriptorPoolSize",
  "VkDescriptorPoolCreateInfo": "struct VkDescriptorPoolCreateInfo",
  "VkDescriptorSetAllocateInfo": "struct VkDescriptorSetAllocateInfo",
  "VkDescriptorSetLayoutBinding": "struct VkDescriptorSetLayoutBinding",
  "VkDescriptorSetLayoutCreateInfo": "struct VkDescriptorSetLayoutCreateInfo",
  "VkWriteDescriptorSet": "struct VkWriteDescriptorSet",
  "VkAttachmentDescription": "struct VkAttachmentDescription",
  "VkAttachmentReference": "struct VkAttachmentReference",
  "VkFramebufferCreateInfo": "struct VkFramebufferCreateInfo",
  "VkSubpassDescription": "struct VkSubpassDescription",
  "VkSubpassDependency": "struct VkSubpassDependency",
  "VkRenderPassCreateInfo": "struct VkRenderPassCreateInfo",
  "VkCommandPoolCreateInfo": "struct VkCommandPoolCreateInfo",
  "VkCommandBufferAllocateInfo": "struct VkCommandBufferAllocateInfo",
  "VkCommandBufferInheritanceInfo": "struct VkCommandBufferInheritanceInfo",
  "VkCommandBufferBeginInfo": "struct VkCommandBufferBeginInfo",
  "VkBufferCopy": "struct VkBufferCopy",
  "VkImageSubresourceLayers": "struct VkImageSubresourceLayers",
  "VkBufferImageCopy": "struct VkBufferImageCopy",
  "VkClearColorValue": "union VkClearColorValue",
  "VkClearDepthStencilValue": "struct VkClearDepthStencilValue",
  "VkClearValue": "union VkClearValue",
  "VkClearAttachment": "struct VkClearAttachment",
  "VkClearRect": "struct VkClearRect",
  "VkImageBlit": "struct VkImageBlit",
  "VkImageCopy": "struct VkImageCopy",
  "VkImageResolve": "struct VkImageResolve",
  "VkRenderPassBeginInfo": "struct VkRenderPassBeginInfo",
  "PFN_vkCreateInstance": "VkResult (*)(const VkInstanceCreateInfo *, const VkAllocationCallbacks *, VkInstance *)",
  "PFN_vkDestroyInstance": "void (*)(VkInstance, const VkAllocationCallbacks *)",
  "PFN_vkEnumeratePhysicalDevices": "VkResult (*)(VkInstance, uint32_t *, VkPhysicalDevice *)",
  "PFN_vkGetPhysicalDeviceFeatures": "void (*)(VkPhysicalDevice, VkPhysicalDeviceFeatures *)",
  "PFN_vkGetPhysicalDeviceFormatProperties": "void (*)(VkPhysicalDevice, VkFormat, VkFormatProperties *)",
  "PFN_vkGetPhysicalDeviceImageFormatProperties": "VkResult (*)(VkPhysicalDevice, VkFormat, VkImageType, VkImageTiling, VkImageUsageFlags, VkImageCreateFlags, VkImageFormatProperties *)",
  "PFN_vkGetPhysicalDeviceProperties": "void (*)(VkPhysicalDevice, VkPhysicalDeviceProperties *)",
  "PFN_vkGetPhysicalDeviceQueueFamilyProperties": "void (*)(VkPhysicalDevice, uint32_t *, VkQueueFamilyProperties *)",
  "PFN_vkGetPhysicalDeviceMemoryProperties": "void (*)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *)",
  "PFN_vkGetInstanceProcAddr": "PFN_vkVoidFunction (*)(VkInstance, const char *)",
  "PFN_vkGetDeviceProcAddr": "PFN_vkVoidFunction (*)(VkDevice, const char *)",
  "PFN_vkCreateDevice": "VkResult (*)(VkPhysicalDevice, const VkDeviceCreateInfo *, const VkAllocationCallbacks *, VkDevice *)",
  "PFN_vkDestroyDevice": "void (*)(VkDevice, const VkAllocationCallbacks *)",
  "PFN_vkEnumerateInstanceExtensionProperties": "VkResult (*)(const char *, uint32_t *, VkExtensionProperties *)",
  "PFN_vkEnumerateDeviceExtensionProperties": "VkResult (*)(VkPhysicalDevice, const char *, uint32_t *, VkExtensionProperties *)",
  "PFN_vkEnumerateInstanceLayerProperties": "VkResult (*)(uint32_t *, VkLayerProperties *)",
  "PFN_vkEnumerateDeviceLayerProperties": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkLayerProperties *)",
  "PFN_vkGetDeviceQueue": "void (*)(VkDevice, uint32_t, uint32_t, VkQueue *)",
  "PFN_vkQueueSubmit": "VkResult (*)(VkQueue, uint32_t, const VkSubmitInfo *, VkFence)",
  "PFN_vkQueueWaitIdle": "VkResult (*)(VkQueue)",
  "PFN_vkDeviceWaitIdle": "VkResult (*)(VkDevice)",
  "PFN_vkAllocateMemory": "VkResult (*)(VkDevice, const VkMemoryAllocateInfo *, const VkAllocationCallbacks *, VkDeviceMemory *)",
  "PFN_vkFreeMemory": "void (*)(VkDevice, VkDeviceMemory, const VkAllocationCallbacks *)",
  "PFN_vkMapMemory": "VkResult (*)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void **)",
  "PFN_vkUnmapMemory": "void (*)(VkDevice, VkDeviceMemory)",
  "PFN_vkFlushMappedMemoryRanges": "VkResult (*)(VkDevice, uint32_t, const VkMappedMemoryRange *)",
  "PFN_vkInvalidateMappedMemoryRanges": "VkResult (*)(VkDevice, uint32_t, const VkMappedMemoryRange *)",
  "PFN_vkGetDeviceMemoryCommitment": "void (*)(VkDevice, VkDeviceMemory, VkDeviceSize *)",
  "PFN_vkBindBufferMemory": "VkResult (*)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize)",
  "PFN_vkBindImageMemory": "VkResult (*)(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize)",
  "PFN_vkGetBufferMemoryRequirements": "void (*)(VkDevice, VkBuffer, VkMemoryRequirements *)",
  "PFN_vkGetImageMemoryRequirements": "void (*)(VkDevice, VkImage, VkMemoryRequirements *)",
  "PFN_vkGetImageSparseMemoryRequirements": "void (*)(VkDevice, VkImage, uint32_t *, VkSparseImageMemoryRequirements *)",
  "PFN_vkGetPhysicalDeviceSparseImageFormatProperties": "void (*)(VkPhysicalDevice, VkFormat, VkImageType, VkSampleCountFlagBits, VkImageUsageFlags, VkImageTiling, uint32_t *, VkSparseImageFormatProperties *)",
  "PFN_vkQueueBindSparse": "VkResult (*)(VkQueue, uint32_t, const VkBindSparseInfo *, VkFence)",
  "PFN_vkCreateFence": "VkResult (*)(VkDevice, const VkFenceCreateInfo *, const VkAllocationCallbacks *, VkFence *)",
  "PFN_vkDestroyFence": "void (*)(VkDevice, VkFence, const VkAllocationCallbacks *)",
  "PFN_vkResetFences": "VkResult (*)(VkDevice, uint32_t, const VkFence *)",
  "PFN_vkGetFenceStatus": "VkResult (*)(VkDevice, VkFence)",
  "PFN_vkWaitForFences": "VkResult (*)(VkDevice, uint32_t, const VkFence *, VkBool32, uint64_t)",
  "PFN_vkCreateSemaphore": "VkResult (*)(VkDevice, const VkSemaphoreCreateInfo *, const VkAllocationCallbacks *, VkSemaphore *)",
  "PFN_vkDestroySemaphore": "void (*)(VkDevice, VkSemaphore, const VkAllocationCallbacks *)",
  "PFN_vkCreateEvent": "VkResult (*)(VkDevice, const VkEventCreateInfo *, const VkAllocationCallbacks *, VkEvent *)",
  "PFN_vkDestroyEvent": "void (*)(VkDevice, VkEvent, const VkAllocationCallbacks *)",
  "PFN_vkGetEventStatus": "VkResult (*)(VkDevice, VkEvent)",
  "PFN_vkSetEvent": "VkResult (*)(VkDevice, VkEvent)",
  "PFN_vkResetEvent": "VkResult (*)(VkDevice, VkEvent)",
  "PFN_vkCreateQueryPool": "VkResult (*)(VkDevice, const VkQueryPoolCreateInfo *, const VkAllocationCallbacks *, VkQueryPool *)",
  "PFN_vkDestroyQueryPool": "void (*)(VkDevice, VkQueryPool, const VkAllocationCallbacks *)",
  "PFN_vkGetQueryPoolResults": "VkResult (*)(VkDevice, VkQueryPool, uint32_t, uint32_t, size_t, void *, VkDeviceSize, VkQueryResultFlags)",
  "PFN_vkCreateBuffer": "VkResult (*)(VkDevice, const VkBufferCreateInfo *, const VkAllocationCallbacks *, VkBuffer *)",
  "PFN_vkDestroyBuffer": "void (*)(VkDevice, VkBuffer, const VkAllocationCallbacks *)",
  "PFN_vkCreateBufferView": "VkResult (*)(VkDevice, const VkBufferViewCreateInfo *, const VkAllocationCallbacks *, VkBufferView *)",
  "PFN_vkDestroyBufferView": "void (*)(VkDevice, VkBufferView, const VkAllocationCallbacks *)",
  "PFN_vkCreateImage": "VkResult (*)(VkDevice, const VkImageCreateInfo *, const VkAllocationCallbacks *, VkImage *)",
  "PFN_vkDestroyImage": "void (*)(VkDevice, VkImage, const VkAllocationCallbacks *)",
  "PFN_vkGetImageSubresourceLayout": "void (*)(VkDevice, VkImage, const VkImageSubresource *, VkSubresourceLayout *)",
  "PFN_vkCreateImageView": "VkResult (*)(VkDevice, const VkImageViewCreateInfo *, const VkAllocationCallbacks *, VkImageView *)",
  "PFN_vkDestroyImageView": "void (*)(VkDevice, VkImageView, const VkAllocationCallbacks *)",
  "PFN_vkCreateShaderModule": "VkResult (*)(VkDevice, const VkShaderModuleCreateInfo *, const VkAllocationCallbacks *, VkShaderModule *)",
  "PFN_vkDestroyShaderModule": "void (*)(VkDevice, VkShaderModule, const VkAllocationCallbacks *)",
  "PFN_vkCreatePipelineCache": "VkResult (*)(VkDevice, const VkPipelineCacheCreateInfo *, const VkAllocationCallbacks *, VkPipelineCache *)",
  "PFN_vkDestroyPipelineCache": "void (*)(VkDevice, VkPipelineCache, const VkAllocationCallbacks *)",
  "PFN_vkGetPipelineCacheData": "VkResult (*)(VkDevice, VkPipelineCache, size_t *, void *)",
  "PFN_vkMergePipelineCaches": "VkResult (*)(VkDevice, VkPipelineCache, uint32_t, const VkPipelineCache *)",
  "PFN_vkCreateGraphicsPipelines": "VkResult (*)(VkDevice, VkPipelineCache, uint32_t, const VkGraphicsPipelineCreateInfo *, const VkAllocationCallbacks *, VkPipeline *)",
  "PFN_vkCreateComputePipelines": "VkResult (*)(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo *, const VkAllocationCallbacks *, VkPipeline *)",
  "PFN_vkDestroyPipeline": "void (*)(VkDevice, VkPipeline, const VkAllocationCallbacks *)",
  "PFN_vkCreatePipelineLayout": "VkResult (*)(VkDevice, const VkPipelineLayoutCreateInfo *, const VkAllocationCallbacks *, VkPipelineLayout *)",
  "PFN_vkDestroyPipelineLayout": "void (*)(VkDevice, VkPipelineLayout, const VkAllocationCallbacks *)",
  "PFN_vkCreateSampler": "VkResult (*)(VkDevice, const VkSamplerCreateInfo *, const VkAllocationCallbacks *, VkSampler *)",
  "PFN_vkDestroySampler": "void (*)(VkDevice, VkSampler, const VkAllocationCallbacks *)",
  "PFN_vkCreateDescriptorSetLayout": "VkResult (*)(VkDevice, const VkDescriptorSetLayoutCreateInfo *, const VkAllocationCallbacks *, VkDescriptorSetLayout *)",
  "PFN_vkDestroyDescriptorSetLayout": "void (*)(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks *)",
  "PFN_vkCreateDescriptorPool": "VkResult (*)(VkDevice, const VkDescriptorPoolCreateInfo *, const VkAllocationCallbacks *, VkDescriptorPool *)",
  "PFN_vkDestroyDescriptorPool": "void (*)(VkDevice, VkDescriptorPool, const VkAllocationCallbacks *)",
  "PFN_vkResetDescriptorPool": "VkResult (*)(VkDevice, VkDescriptorPool, VkDescriptorPoolResetFlags)",
  "PFN_vkAllocateDescriptorSets": "VkResult (*)(VkDevice, const VkDescriptorSetAllocateInfo *, VkDescriptorSet *)",
  "PFN_vkFreeDescriptorSets": "VkResult (*)(VkDevice, VkDescriptorPool, uint32_t, const VkDescriptorSet *)",
  "PFN_vkUpdateDescriptorSets": "void (*)(VkDevice, uint32_t, const VkWriteDescriptorSet *, uint32_t, const VkCopyDescriptorSet *)",
  "PFN_vkCreateFramebuffer": "VkResult (*)(VkDevice, const VkFramebufferCreateInfo *, const VkAllocationCallbacks *, VkFramebuffer *)",
  "PFN_vkDestroyFramebuffer": "void (*)(VkDevice, VkFramebuffer, const VkAllocationCallbacks *)",
  "PFN_vkCreateRenderPass": "VkResult (*)(VkDevice, const VkRenderPassCreateInfo *, const VkAllocationCallbacks *, VkRenderPass *)",
  "PFN_vkDestroyRenderPass": "void (*)(VkDevice, VkRenderPass, const VkAllocationCallbacks *)",
  "PFN_vkGetRenderAreaGranularity": "void (*)(VkDevice, VkRenderPass, VkExtent2D *)",
  "PFN_vkCreateCommandPool": "VkResult (*)(VkDevice, const VkCommandPoolCreateInfo *, const VkAllocationCallbacks *, VkCommandPool *)",
  "PFN_vkDestroyCommandPool": "void (*)(VkDevice, VkCommandPool, const VkAllocationCallbacks *)",
  "PFN_vkResetCommandPool": "VkResult (*)(VkDevice, VkCommandPool, VkCommandPoolResetFlags)",
  "PFN_vkAllocateCommandBuffers": "VkResult (*)(VkDevice, const VkCommandBufferAllocateInfo *, VkCommandBuffer *)",
  "PFN_vkFreeCommandBuffers": "void (*)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer *)",
  "PFN_vkBeginCommandBuffer": "VkResult (*)(VkCommandBuffer, const VkCommandBufferBeginInfo *)",
  "PFN_vkEndCommandBuffer": "VkResult (*)(VkCommandBuffer)",
  "PFN_vkResetCommandBuffer": "VkResult (*)(VkCommandBuffer, VkCommandBufferResetFlags)",
  "PFN_vkCmdBindPipeline": "void (*)(VkCommandBuffer, VkPipelineBindPoint, VkPipeline)",
  "PFN_vkCmdSetViewport": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkViewport *)",
  "PFN_vkCmdSetScissor": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkRect2D *)",
  "PFN_vkCmdSetLineWidth": "void (*)(VkCommandBuffer, float)",
  "PFN_vkCmdSetDepthBias": "void (*)(VkCommandBuffer, float, float, float)",
  "PFN_vkCmdSetBlendConstants": "void (*)(VkCommandBuffer, const float *)",
  "PFN_vkCmdSetDepthBounds": "void (*)(VkCommandBuffer, float, float)",
  "PFN_vkCmdSetStencilCompareMask": "void (*)(VkCommandBuffer, VkStencilFaceFlags, uint32_t)",
  "PFN_vkCmdSetStencilWriteMask": "void (*)(VkCommandBuffer, VkStencilFaceFlags, uint32_t)",
  "PFN_vkCmdSetStencilReference": "void (*)(VkCommandBuffer, VkStencilFaceFlags, uint32_t)",
  "PFN_vkCmdBindDescriptorSets": "void (*)(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet *, uint32_t, const uint32_t *)",
  "PFN_vkCmdBindIndexBuffer": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType)",
  "PFN_vkCmdBindVertexBuffers": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer *, const VkDeviceSize *)",
  "PFN_vkCmdDraw": "void (*)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t)",
  "PFN_vkCmdDrawIndexed": "void (*)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)",
  "PFN_vkCmdDrawIndirect": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "PFN_vkCmdDrawIndexedIndirect": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "PFN_vkCmdDispatch": "void (*)(VkCommandBuffer, uint32_t, uint32_t, uint32_t)",
  "PFN_vkCmdDispatchIndirect": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize)",
  "PFN_vkCmdCopyBuffer": "void (*)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy *)",
  "PFN_vkCmdCopyImage": "void (*)(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageCopy *)",
  "PFN_vkCmdBlitImage": "void (*)(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageBlit *, VkFilter)",
  "PFN_vkCmdCopyBufferToImage": "void (*)(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, uint32_t, const VkBufferImageCopy *)",
  "PFN_vkCmdCopyImageToBuffer": "void (*)(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, uint32_t, const VkBufferImageCopy *)",
  "PFN_vkCmdUpdateBuffer": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void *)",
  "PFN_vkCmdFillBuffer": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t)",
  "PFN_vkCmdClearColorImage": "void (*)(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue *, uint32_t, const VkImageSubresourceRange *)",
  "PFN_vkCmdClearDepthStencilImage": "void (*)(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue *, uint32_t, const VkImageSubresourceRange *)",
  "PFN_vkCmdClearAttachments": "void (*)(VkCommandBuffer, uint32_t, const VkClearAttachment *, uint32_t, const VkClearRect *)",
  "PFN_vkCmdResolveImage": "void (*)(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageResolve *)",
  "PFN_vkCmdSetEvent": "void (*)(VkCommandBuffer, VkEvent, VkPipelineStageFlags)",
  "PFN_vkCmdResetEvent": "void (*)(VkCommandBuffer, VkEvent, VkPipelineStageFlags)",
  "PFN_vkCmdWaitEvents": "void (*)(VkCommandBuffer, uint32_t, const VkEvent *, VkPipelineStageFlags, VkPipelineStageFlags, uint32_t, const VkMemoryBarrier *, uint32_t, const VkBufferMemoryBarrier *, uint32_t, const VkImageMemoryBarrier *)",
  "PFN_vkCmdPipelineBarrier": "void (*)(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t, const VkMemoryBarrier *, uint32_t, const VkBufferMemoryBarrier *, uint32_t, const VkImageMemoryBarrier *)",
  "PFN_vkCmdBeginQuery": "void (*)(VkCommandBuffer, VkQueryPool, uint32_t, VkQueryControlFlags)",
  "PFN_vkCmdEndQuery": "void (*)(VkCommandBuffer, VkQueryPool, uint32_t)",
  "PFN_vkCmdResetQueryPool": "void (*)(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t)",
  "PFN_vkCmdWriteTimestamp": "void (*)(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t)",
  "PFN_vkCmdCopyQueryPoolResults": "void (*)(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t, VkBuffer, VkDeviceSize, VkDeviceSize, VkQueryResultFlags)",
  "PFN_vkCmdPushConstants": "void (*)(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, uint32_t, uint32_t, const void *)",
  "PFN_vkCmdBeginRenderPass": "void (*)(VkCommandBuffer, const VkRenderPassBeginInfo *, VkSubpassContents)",
  "PFN_vkCmdNextSubpass": "void (*)(VkCommandBuffer, VkSubpassContents)",
  "PFN_vkCmdEndRenderPass": "void (*)(VkCommandBuffer)",
  "PFN_vkCmdExecuteCommands": "void (*)(VkCommandBuffer, uint32_t, const VkCommandBuffer *)",
  "VkSamplerYcbcrConversion": "struct VkSamplerYcbcrConversion_T *",
  "VkDescriptorUpdateTemplate": "struct VkDescriptorUpdateTemplate_T *",
  "VkPointClippingBehavior": "enum VkPointClippingBehavior",
  "VkTessellationDomainOrigin": "enum VkTessellationDomainOrigin",
  "VkSamplerYcbcrModelConversion": "enum VkSamplerYcbcrModelConversion",
  "VkSamplerYcbcrRange": "enum VkSamplerYcbcrRange",
  "VkChromaLocation": "enum VkChromaLocation",
  "VkDescriptorUpdateTemplateType": "enum VkDescriptorUpdateTemplateType",
  "VkSubgroupFeatureFlagBits": "enum VkSubgroupFeatureFlagBits",
  "VkSubgroupFeatureFlags": "VkFlags",
  "VkPeerMemoryFeatureFlagBits": "enum VkPeerMemoryFeatureFlagBits",
  "VkPeerMemoryFeatureFlags": "VkFlags",
  "VkMemoryAllocateFlagBits": "enum VkMemoryAllocateFlagBits",
  "VkMemoryAllocateFlags": "VkFlags",
  "VkCommandPoolTrimFlags": "VkFlags",
  "VkDescriptorUpdateTemplateCreateFlags": "VkFlags",
  "VkExternalMemoryHandleTypeFlagBits": "enum VkExternalMemoryHandleTypeFlagBits",
  "VkExternalMemoryHandleTypeFlags": "VkFlags",
  "VkExternalMemoryFeatureFlagBits": "enum VkExternalMemoryFeatureFlagBits",
  "VkExternalMemoryFeatureFlags": "VkFlags",
  "VkExternalFenceHandleTypeFlagBits": "enum VkExternalFenceHandleTypeFlagBits",
  "VkExternalFenceHandleTypeFlags": "VkFlags",
  "VkExternalFenceFeatureFlagBits": "enum VkExternalFenceFeatureFlagBits",
  "VkExternalFenceFeatureFlags": "VkFlags",
  "VkFenceImportFlagBits": "enum VkFenceImportFlagBits",
  "VkFenceImportFlags": "VkFlags",
  "VkSemaphoreImportFlagBits": "enum VkSemaphoreImportFlagBits",
  "VkSemaphoreImportFlags": "VkFlags",
  "VkExternalSemaphoreHandleTypeFlagBits": "enum VkExternalSemaphoreHandleTypeFlagBits",
  "VkExternalSemaphoreHandleTypeFlags": "VkFlags",
  "VkExternalSemaphoreFeatureFlagBits": "enum VkExternalSemaphoreFeatureFlagBits",
  "VkExternalSemaphoreFeatureFlags": "VkFlags",
  "VkPhysicalDeviceSubgroupProperties": "struct VkPhysicalDeviceSubgroupProperties",
  "VkBindBufferMemoryInfo": "struct VkBindBufferMemoryInfo",
  "VkBindImageMemoryInfo": "struct VkBindImageMemoryInfo",
  "VkPhysicalDevice16BitStorageFeatures": "struct VkPhysicalDevice16BitStorageFeatures",
  "VkMemoryDedicatedRequirements": "struct VkMemoryDedicatedRequirements",
  "VkMemoryDedicatedAllocateInfo": "struct VkMemoryDedicatedAllocateInfo",
  "VkMemoryAllocateFlagsInfo": "struct VkMemoryAllocateFlagsInfo",
  "VkDeviceGroupRenderPassBeginInfo": "struct VkDeviceGroupRenderPassBeginInfo",
  "VkDeviceGroupCommandBufferBeginInfo": "struct VkDeviceGroupCommandBufferBeginInfo",
  "VkDeviceGroupSubmitInfo": "struct VkDeviceGroupSubmitInfo",
  "VkDeviceGroupBindSparseInfo": "struct VkDeviceGroupBindSparseInfo",
  "VkBindBufferMemoryDeviceGroupInfo": "struct VkBindBufferMemoryDeviceGroupInfo",
  "VkBindImageMemoryDeviceGroupInfo": "struct VkBindImageMemoryDeviceGroupInfo",
  "VkPhysicalDeviceGroupProperties": "struct VkPhysicalDeviceGroupProperties",
  "VkDeviceGroupDeviceCreateInfo": "struct VkDeviceGroupDeviceCreateInfo",
  "VkBufferMemoryRequirementsInfo2": "struct VkBufferMemoryRequirementsInfo2",
  "VkImageMemoryRequirementsInfo2": "struct VkImageMemoryRequirementsInfo2",
  "VkImageSparseMemoryRequirementsInfo2": "struct VkImageSparseMemoryRequirementsInfo2",
  "VkMemoryRequirements2": "struct VkMemoryRequirements2",
  "VkSparseImageMemoryRequirements2": "struct VkSparseImageMemoryRequirements2",
  "VkPhysicalDeviceFeatures2": "struct VkPhysicalDeviceFeatures2",
  "VkPhysicalDeviceProperties2": "struct VkPhysicalDeviceProperties2",
  "VkFormatProperties2": "struct VkFormatProperties2",
  "VkImageFormatProperties2": "struct VkImageFormatProperties2",
  "VkPhysicalDeviceImageFormatInfo2": "struct VkPhysicalDeviceImageFormatInfo2",
  "VkQueueFamilyProperties2": "struct VkQueueFamilyProperties2",
  "VkPhysicalDeviceMemoryProperties2": "struct VkPhysicalDeviceMemoryProperties2",
  "VkSparseImageFormatProperties2": "struct VkSparseImageFormatProperties2",
  "VkPhysicalDeviceSparseImageFormatInfo2": "struct VkPhysicalDeviceSparseImageFormatInfo2",
  "VkPhysicalDevicePointClippingProperties": "struct VkPhysicalDevicePointClippingProperties",
  "VkInputAttachmentAspectReference": "struct VkInputAttachmentAspectReference",
  "VkRenderPassInputAttachmentAspectCreateInfo": "struct VkRenderPassInputAttachmentAspectCreateInfo",
  "VkImageViewUsageCreateInfo": "struct VkImageViewUsageCreateInfo",
  "VkPipelineTessellationDomainOriginStateCreateInfo": "struct VkPipelineTessellationDomainOriginStateCreateInfo",
  "VkRenderPassMultiviewCreateInfo": "struct VkRenderPassMultiviewCreateInfo",
  "VkPhysicalDeviceMultiviewFeatures": "struct VkPhysicalDeviceMultiviewFeatures",
  "VkPhysicalDeviceMultiviewProperties": "struct VkPhysicalDeviceMultiviewProperties",
  "VkPhysicalDeviceVariablePointersFeatures": "struct VkPhysicalDeviceVariablePointersFeatures",
  "VkPhysicalDeviceVariablePointerFeatures": "VkPhysicalDeviceVariablePointersFeatures",
  "VkPhysicalDeviceProtectedMemoryFeatures": "struct VkPhysicalDeviceProtectedMemoryFeatures",
  "VkPhysicalDeviceProtectedMemoryProperties": "struct VkPhysicalDeviceProtectedMemoryProperties",
  "VkDeviceQueueInfo2": "struct VkDeviceQueueInfo2",
  "VkProtectedSubmitInfo": "struct VkProtectedSubmitInfo",
  "VkSamplerYcbcrConversionCreateInfo": "struct VkSamplerYcbcrConversionCreateInfo",
  "VkSamplerYcbcrConversionInfo": "struct VkSamplerYcbcrConversionInfo",
  "VkBindImagePlaneMemoryInfo": "struct VkBindImagePlaneMemoryInfo",
  "VkImagePlaneMemoryRequirementsInfo": "struct VkImagePlaneMemoryRequirementsInfo",
  "VkPhysicalDeviceSamplerYcbcrConversionFeatures": "struct VkPhysicalDeviceSamplerYcbcrConversionFeatures",
  "VkSamplerYcbcrConversionImageFormatProperties": "struct VkSamplerYcbcrConversionImageFormatProperties",
  "VkDescriptorUpdateTemplateEntry": "struct VkDescriptorUpdateTemplateEntry",
  "VkDescriptorUpdateTemplateCreateInfo": "struct VkDescriptorUpdateTemplateCreateInfo",
  "VkExternalMemoryProperties": "struct VkExternalMemoryProperties",
  "VkPhysicalDeviceExternalImageFormatInfo": "struct VkPhysicalDeviceExternalImageFormatInfo",
  "VkExternalImageFormatProperties": "struct VkExternalImageFormatProperties",
  "VkPhysicalDeviceExternalBufferInfo": "struct VkPhysicalDeviceExternalBufferInfo",
  "VkExternalBufferProperties": "struct VkExternalBufferProperties",
  "VkPhysicalDeviceIDProperties": "struct VkPhysicalDeviceIDProperties",
  "VkExternalMemoryImageCreateInfo": "struct VkExternalMemoryImageCreateInfo",
  "VkExternalMemoryBufferCreateInfo": "struct VkExternalMemoryBufferCreateInfo",
  "VkExportMemoryAllocateInfo": "struct VkExportMemoryAllocateInfo",
  "VkPhysicalDeviceExternalFenceInfo": "struct VkPhysicalDeviceExternalFenceInfo",
  "VkExternalFenceProperties": "struct VkExternalFenceProperties",
  "VkExportFenceCreateInfo": "struct VkExportFenceCreateInfo",
  "VkExportSemaphoreCreateInfo": "struct VkExportSemaphoreCreateInfo",
  "VkPhysicalDeviceExternalSemaphoreInfo": "struct VkPhysicalDeviceExternalSemaphoreInfo",
  "VkExternalSemaphoreProperties": "struct VkExternalSemaphoreProperties",
  "VkPhysicalDeviceMaintenance3Properties": "struct VkPhysicalDeviceMaintenance3Properties",
  "VkDescriptorSetLayoutSupport": "struct VkDescriptorSetLayoutSupport",
  "VkPhysicalDeviceShaderDrawParametersFeatures": "struct VkPhysicalDeviceShaderDrawParametersFeatures",
  "VkPhysicalDeviceShaderDrawParameterFeatures": "VkPhysicalDeviceShaderDrawParametersFeatures",
  "PFN_vkEnumerateInstanceVersion": "VkResult (*)(uint32_t *)",
  "PFN_vkBindBufferMemory2": "VkResult (*)(VkDevice, uint32_t, const VkBindBufferMemoryInfo *)",
  "PFN_vkBindImageMemory2": "VkResult (*)(VkDevice, uint32_t, const VkBindImageMemoryInfo *)",
  "PFN_vkGetDeviceGroupPeerMemoryFeatures": "void (*)(VkDevice, uint32_t, uint32_t, uint32_t, VkPeerMemoryFeatureFlags *)",
  "PFN_vkCmdSetDeviceMask": "void (*)(VkCommandBuffer, uint32_t)",
  "PFN_vkCmdDispatchBase": "void (*)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)",
  "PFN_vkEnumeratePhysicalDeviceGroups": "VkResult (*)(VkInstance, uint32_t *, VkPhysicalDeviceGroupProperties *)",
  "PFN_vkGetImageMemoryRequirements2": "void (*)(VkDevice, const VkImageMemoryRequirementsInfo2 *, VkMemoryRequirements2 *)",
  "PFN_vkGetBufferMemoryRequirements2": "void (*)(VkDevice, const VkBufferMemoryRequirementsInfo2 *, VkMemoryRequirements2 *)",
  "PFN_vkGetImageSparseMemoryRequirements2": "void (*)(VkDevice, const VkImageSparseMemoryRequirementsInfo2 *, uint32_t *, VkSparseImageMemoryRequirements2 *)",
  "PFN_vkGetPhysicalDeviceFeatures2": "void (*)(VkPhysicalDevice, VkPhysicalDeviceFeatures2 *)",
  "PFN_vkGetPhysicalDeviceProperties2": "void (*)(VkPhysicalDevice, VkPhysicalDeviceProperties2 *)",
  "PFN_vkGetPhysicalDeviceFormatProperties2": "void (*)(VkPhysicalDevice, VkFormat, VkFormatProperties2 *)",
  "PFN_vkGetPhysicalDeviceImageFormatProperties2": "VkResult (*)(VkPhysicalDevice, const VkPhysicalDeviceImageFormatInfo2 *, VkImageFormatProperties2 *)",
  "PFN_vkGetPhysicalDeviceQueueFamilyProperties2": "void (*)(VkPhysicalDevice, uint32_t *, VkQueueFamilyProperties2 *)",
  "PFN_vkGetPhysicalDeviceMemoryProperties2": "void (*)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2 *)",
  "PFN_vkGetPhysicalDeviceSparseImageFormatProperties2": "void (*)(VkPhysicalDevice, const VkPhysicalDeviceSparseImageFormatInfo2 *, uint32_t *, VkSparseImageFormatProperties2 *)",
  "PFN_vkTrimCommandPool": "void (*)(VkDevice, VkCommandPool, VkCommandPoolTrimFlags)",
  "PFN_vkGetDeviceQueue2": "void (*)(VkDevice, const VkDeviceQueueInfo2 *, VkQueue *)",
  "PFN_vkCreateSamplerYcbcrConversion": "VkResult (*)(VkDevice, const VkSamplerYcbcrConversionCreateInfo *, const VkAllocationCallbacks *, VkSamplerYcbcrConversion *)",
  "PFN_vkDestroySamplerYcbcrConversion": "void (*)(VkDevice, VkSamplerYcbcrConversion, const VkAllocationCallbacks *)",
  "PFN_vkCreateDescriptorUpdateTemplate": "VkResult (*)(VkDevice, const VkDescriptorUpdateTemplateCreateInfo *, const VkAllocationCallbacks *, VkDescriptorUpdateTemplate *)",
  "PFN_vkDestroyDescriptorUpdateTemplate": "void (*)(VkDevice, VkDescriptorUpdateTemplate, const VkAllocationCallbacks *)",
  "PFN_vkUpdateDescriptorSetWithTemplate": "void (*)(VkDevice, VkDescriptorSet, VkDescriptorUpdateTemplate, const void *)",
  "PFN_vkGetPhysicalDeviceExternalBufferProperties": "void (*)(VkPhysicalDevice, const VkPhysicalDeviceExternalBufferInfo *, VkExternalBufferProperties *)",
  "PFN_vkGetPhysicalDeviceExternalFenceProperties": "void (*)(VkPhysicalDevice, const VkPhysicalDeviceExternalFenceInfo *, VkExternalFenceProperties *)",
  "PFN_vkGetPhysicalDeviceExternalSemaphoreProperties": "void (*)(VkPhysicalDevice, const VkPhysicalDeviceExternalSemaphoreInfo *, VkExternalSemaphoreProperties *)",
  "PFN_vkGetDescriptorSetLayoutSupport": "void (*)(VkDevice, const VkDescriptorSetLayoutCreateInfo *, VkDescriptorSetLayoutSupport *)",
  "VkDriverId": "enum VkDriverId",
  "VkShaderFloatControlsIndependence": "enum VkShaderFloatControlsIndependence",
  "VkSamplerReductionMode": "enum VkSamplerReductionMode",
  "VkSemaphoreType": "enum VkSemaphoreType",
  "VkResolveModeFlagBits": "enum VkResolveModeFlagBits",
  "VkResolveModeFlags": "VkFlags",
  "VkDescriptorBindingFlagBits": "enum VkDescriptorBindingFlagBits",
  "VkDescriptorBindingFlags": "VkFlags",
  "VkSemaphoreWaitFlagBits": "enum VkSemaphoreWaitFlagBits",
  "VkSemaphoreWaitFlags": "VkFlags",
  "VkPhysicalDeviceVulkan11Features": "struct VkPhysicalDeviceVulkan11Features",
  "VkPhysicalDeviceVulkan11Properties": "struct VkPhysicalDeviceVulkan11Properties",
  "VkPhysicalDeviceVulkan12Features": "struct VkPhysicalDeviceVulkan12Features",
  "VkConformanceVersion": "struct VkConformanceVersion",
  "VkPhysicalDeviceVulkan12Properties": "struct VkPhysicalDeviceVulkan12Properties",
  "VkImageFormatListCreateInfo": "struct VkImageFormatListCreateInfo",
  "VkAttachmentDescription2": "struct VkAttachmentDescription2",
  "VkAttachmentReference2": "struct VkAttachmentReference2",
  "VkSubpassDescription2": "struct VkSubpassDescription2",
  "VkSubpassDependency2": "struct VkSubpassDependency2",
  "VkRenderPassCreateInfo2": "struct VkRenderPassCreateInfo2",
  "VkSubpassBeginInfo": "struct VkSubpassBeginInfo",
  "VkSubpassEndInfo": "struct VkSubpassEndInfo",
  "VkPhysicalDevice8BitStorageFeatures": "struct VkPhysicalDevice8BitStorageFeatures",
  "VkPhysicalDeviceDriverProperties": "struct VkPhysicalDeviceDriverProperties",
  "VkPhysicalDeviceShaderAtomicInt64Features": "struct VkPhysicalDeviceShaderAtomicInt64Features",
  "VkPhysicalDeviceShaderFloat16Int8Features": "struct VkPhysicalDeviceShaderFloat16Int8Features",
  "VkPhysicalDeviceFloatControlsProperties": "struct VkPhysicalDeviceFloatControlsProperties",
  "VkDescriptorSetLayoutBindingFlagsCreateInfo": "struct VkDescriptorSetLayoutBindingFlagsCreateInfo",
  "VkPhysicalDeviceDescriptorIndexingFeatures": "struct VkPhysicalDeviceDescriptorIndexingFeatures",
  "VkPhysicalDeviceDescriptorIndexingProperties": "struct VkPhysicalDeviceDescriptorIndexingProperties",
  "VkDescriptorSetVariableDescriptorCountAllocateInfo": "struct VkDescriptorSetVariableDescriptorCountAllocateInfo",
  "VkDescriptorSetVariableDescriptorCountLayoutSupport": "struct VkDescriptorSetVariableDescriptorCountLayoutSupport",
  "VkSubpassDescriptionDepthStencilResolve": "struct VkSubpassDescriptionDepthStencilResolve",
  "VkPhysicalDeviceDepthStencilResolveProperties": "struct VkPhysicalDeviceDepthStencilResolveProperties",
  "VkPhysicalDeviceScalarBlockLayoutFeatures": "struct VkPhysicalDeviceScalarBlockLayoutFeatures",
  "VkImageStencilUsageCreateInfo": "struct VkImageStencilUsageCreateInfo",
  "VkSamplerReductionModeCreateInfo": "struct VkSamplerReductionModeCreateInfo",
  "VkPhysicalDeviceSamplerFilterMinmaxProperties": "struct VkPhysicalDeviceSamplerFilterMinmaxProperties",
  "VkPhysicalDeviceVulkanMemoryModelFeatures": "struct VkPhysicalDeviceVulkanMemoryModelFeatures",
  "VkPhysicalDeviceImagelessFramebufferFeatures": "struct VkPhysicalDeviceImagelessFramebufferFeatures",
  "VkFramebufferAttachmentImageInfo": "struct VkFramebufferAttachmentImageInfo",
  "VkFramebufferAttachmentsCreateInfo": "struct VkFramebufferAttachmentsCreateInfo",
  "VkRenderPassAttachmentBeginInfo": "struct VkRenderPassAttachmentBeginInfo",
  "VkPhysicalDeviceUniformBufferStandardLayoutFeatures": "struct VkPhysicalDeviceUniformBufferStandardLayoutFeatures",
  "VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures": "struct VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures",
  "VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures": "struct VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures",
  "VkAttachmentReferenceStencilLayout": "struct VkAttachmentReferenceStencilLayout",
  "VkAttachmentDescriptionStencilLayout": "struct VkAttachmentDescriptionStencilLayout",
  "VkPhysicalDeviceHostQueryResetFeatures": "struct VkPhysicalDeviceHostQueryResetFeatures",
  "VkPhysicalDeviceTimelineSemaphoreFeatures": "struct VkPhysicalDeviceTimelineSemaphoreFeatures",
  "VkPhysicalDeviceTimelineSemaphoreProperties": "struct VkPhysicalDeviceTimelineSemaphoreProperties",
  "VkSemaphoreTypeCreateInfo": "struct VkSemaphoreTypeCreateInfo",
  "VkTimelineSemaphoreSubmitInfo": "struct VkTimelineSemaphoreSubmitInfo",
  "VkSemaphoreWaitInfo": "struct VkSemaphoreWaitInfo",
  "VkSemaphoreSignalInfo": "struct VkSemaphoreSignalInfo",
  "VkPhysicalDeviceBufferDeviceAddressFeatures": "struct VkPhysicalDeviceBufferDeviceAddressFeatures",
  "VkBufferDeviceAddressInfo": "struct VkBufferDeviceAddressInfo",
  "VkBufferOpaqueCaptureAddressCreateInfo": "struct VkBufferOpaqueCaptureAddressCreateInfo",
  "VkMemoryOpaqueCaptureAddressAllocateInfo": "struct VkMemoryOpaqueCaptureAddressAllocateInfo",
  "VkDeviceMemoryOpaqueCaptureAddressInfo": "struct VkDeviceMemoryOpaqueCaptureAddressInfo",
  "PFN_vkCmdDrawIndirectCount": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "PFN_vkCmdDrawIndexedIndirectCount": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "PFN_vkCreateRenderPass2": "VkResult (*)(VkDevice, const VkRenderPassCreateInfo2 *, const VkAllocationCallbacks *, VkRenderPass *)",
  "PFN_vkCmdBeginRenderPass2": "void (*)(VkCommandBuffer, const VkRenderPassBeginInfo *, const VkSubpassBeginInfo *)",
  "PFN_vkCmdNextSubpass2": "void (*)(VkCommandBuffer, const VkSubpassBeginInfo *, const VkSubpassEndInfo *)",
  "PFN_vkCmdEndRenderPass2": "void (*)(VkCommandBuffer, const VkSubpassEndInfo *)",
  "PFN_vkResetQueryPool": "void (*)(VkDevice, VkQueryPool, uint32_t, uint32_t)",
  "PFN_vkGetSemaphoreCounterValue": "VkResult (*)(VkDevice, VkSemaphore, uint64_t *)",
  "PFN_vkWaitSemaphores": "VkResult (*)(VkDevice, const VkSemaphoreWaitInfo *, uint64_t)",
  "PFN_vkSignalSemaphore": "VkResult (*)(VkDevice, const VkSemaphoreSignalInfo *)",
  "PFN_vkGetBufferDeviceAddress": "VkDeviceAddress (*)(VkDevice, const VkBufferDeviceAddressInfo *)",
  "PFN_vkGetBufferOpaqueCaptureAddress": "uint64_t (*)(VkDevice, const VkBufferDeviceAddressInfo *)",
  "PFN_vkGetDeviceMemoryOpaqueCaptureAddress": "uint64_t (*)(VkDevice, const VkDeviceMemoryOpaqueCaptureAddressInfo *)",
  "VkFlags64": "uint64_t",
  "VkPrivateDataSlot": "struct VkPrivateDataSlot_T *",
  "VkPipelineCreationFeedbackFlagBits": "enum VkPipelineCreationFeedbackFlagBits",
  "VkPipelineCreationFeedbackFlags": "VkFlags",
  "VkToolPurposeFlagBits": "enum VkToolPurposeFlagBits",
  "VkToolPurposeFlags": "VkFlags",
  "VkPrivateDataSlotCreateFlags": "VkFlags",
  "VkPipelineStageFlags2": "VkFlags64",
  "VkPipelineStageFlagBits2": "VkFlags64",
  "VkAccessFlags2": "VkFlags64",
  "VkAccessFlagBits2": "VkFlags64",
  "VkSubmitFlagBits": "enum VkSubmitFlagBits",
  "VkSubmitFlags": "VkFlags",
  "VkRenderingFlagBits": "enum VkRenderingFlagBits",
  "VkRenderingFlags": "VkFlags",
  "VkFormatFeatureFlags2": "VkFlags64",
  "VkFormatFeatureFlagBits2": "VkFlags64",
  "VkPhysicalDeviceVulkan13Features": "struct VkPhysicalDeviceVulkan13Features",
  "VkPhysicalDeviceVulkan13Properties": "struct VkPhysicalDeviceVulkan13Properties",
  "VkPipelineCreationFeedback": "struct VkPipelineCreationFeedback",
  "VkPipelineCreationFeedbackCreateInfo": "struct VkPipelineCreationFeedbackCreateInfo",
  "VkPhysicalDeviceShaderTerminateInvocationFeatures": "struct VkPhysicalDeviceShaderTerminateInvocationFeatures",
  "VkPhysicalDeviceToolProperties": "struct VkPhysicalDeviceToolProperties",
  "VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures": "struct VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures",
  "VkPhysicalDevicePrivateDataFeatures": "struct VkPhysicalDevicePrivateDataFeatures",
  "VkDevicePrivateDataCreateInfo": "struct VkDevicePrivateDataCreateInfo",
  "VkPrivateDataSlotCreateInfo": "struct VkPrivateDataSlotCreateInfo",
  "VkPhysicalDevicePipelineCreationCacheControlFeatures": "struct VkPhysicalDevicePipelineCreationCacheControlFeatures",
  "VkMemoryBarrier2": "struct VkMemoryBarrier2",
  "VkBufferMemoryBarrier2": "struct VkBufferMemoryBarrier2",
  "VkImageMemoryBarrier2": "struct VkImageMemoryBarrier2",
  "VkDependencyInfo": "struct VkDependencyInfo",
  "VkSemaphoreSubmitInfo": "struct VkSemaphoreSubmitInfo",
  "VkCommandBufferSubmitInfo": "struct VkCommandBufferSubmitInfo",
  "VkSubmitInfo2": "struct VkSubmitInfo2",
  "VkPhysicalDeviceSynchronization2Features": "struct VkPhysicalDeviceSynchronization2Features",
  "VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures": "struct VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures",
  "VkPhysicalDeviceImageRobustnessFeatures": "struct VkPhysicalDeviceImageRobustnessFeatures",
  "VkBufferCopy2": "struct VkBufferCopy2",
  "VkCopyBufferInfo2": "struct VkCopyBufferInfo2",
  "VkImageCopy2": "struct VkImageCopy2",
  "VkCopyImageInfo2": "struct VkCopyImageInfo2",
  "VkBufferImageCopy2": "struct VkBufferImageCopy2",
  "VkCopyBufferToImageInfo2": "struct VkCopyBufferToImageInfo2",
  "VkCopyImageToBufferInfo2": "struct VkCopyImageToBufferInfo2",
  "VkImageBlit2": "struct VkImageBlit2",
  "VkBlitImageInfo2": "struct VkBlitImageInfo2",
  "VkImageResolve2": "struct VkImageResolve2",
  "VkResolveImageInfo2": "struct VkResolveImageInfo2",
  "VkPhysicalDeviceSubgroupSizeControlFeatures": "struct VkPhysicalDeviceSubgroupSizeControlFeatures",
  "VkPhysicalDeviceSubgroupSizeControlProperties": "struct VkPhysicalDeviceSubgroupSizeControlProperties",
  "VkPipelineShaderStageRequiredSubgroupSizeCreateInfo": "struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo",
  "VkPhysicalDeviceInlineUniformBlockFeatures": "struct VkPhysicalDeviceInlineUniformBlockFeatures",
  "VkPhysicalDeviceInlineUniformBlockProperties": "struct VkPhysicalDeviceInlineUniformBlockProperties",
  "VkWriteDescriptorSetInlineUniformBlock": "struct VkWriteDescriptorSetInlineUniformBlock",
  "VkDescriptorPoolInlineUniformBlockCreateInfo": "struct VkDescriptorPoolInlineUniformBlockCreateInfo",
  "VkPhysicalDeviceTextureCompressionASTCHDRFeatures": "struct VkPhysicalDeviceTextureCompressionASTCHDRFeatures",
  "VkRenderingAttachmentInfo": "struct VkRenderingAttachmentInfo",
  "VkRenderingInfo": "struct VkRenderingInfo",
  "VkPipelineRenderingCreateInfo": "struct VkPipelineRenderingCreateInfo",
  "VkPhysicalDeviceDynamicRenderingFeatures": "struct VkPhysicalDeviceDynamicRenderingFeatures",
  "VkCommandBufferInheritanceRenderingInfo": "struct VkCommandBufferInheritanceRenderingInfo",
  "VkPhysicalDeviceShaderIntegerDotProductFeatures": "struct VkPhysicalDeviceShaderIntegerDotProductFeatures",
  "VkPhysicalDeviceShaderIntegerDotProductProperties": "struct VkPhysicalDeviceShaderIntegerDotProductProperties",
  "VkPhysicalDeviceTexelBufferAlignmentProperties": "struct VkPhysicalDeviceTexelBufferAlignmentProperties",
  "VkFormatProperties3": "struct VkFormatProperties3",
  "VkPhysicalDeviceMaintenance4Features": "struct VkPhysicalDeviceMaintenance4Features",
  "VkPhysicalDeviceMaintenance4Properties": "struct VkPhysicalDeviceMaintenance4Properties",
  "VkDeviceBufferMemoryRequirements": "struct VkDeviceBufferMemoryRequirements",
  "VkDeviceImageMemoryRequirements": "struct VkDeviceImageMemoryRequirements",
  "PFN_vkGetPhysicalDeviceToolProperties": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkPhysicalDeviceToolProperties *)",
  "PFN_vkCreatePrivateDataSlot": "VkResult (*)(VkDevice, const VkPrivateDataSlotCreateInfo *, const VkAllocationCallbacks *, VkPrivateDataSlot *)",
  "PFN_vkDestroyPrivateDataSlot": "void (*)(VkDevice, VkPrivateDataSlot, const VkAllocationCallbacks *)",
  "PFN_vkSetPrivateData": "VkResult (*)(VkDevice, VkObjectType, uint64_t, VkPrivateDataSlot, uint64_t)",
  "PFN_vkGetPrivateData": "void (*)(VkDevice, VkObjectType, uint64_t, VkPrivateDataSlot, uint64_t *)",
  "PFN_vkCmdSetEvent2": "void (*)(VkCommandBuffer, VkEvent, const VkDependencyInfo *)",
  "PFN_vkCmdResetEvent2": "void (*)(VkCommandBuffer, VkEvent, VkPipelineStageFlags2)",
  "PFN_vkCmdWaitEvents2": "void (*)(VkCommandBuffer, uint32_t, const VkEvent *, const VkDependencyInfo *)",
  "PFN_vkCmdPipelineBarrier2": "void (*)(VkCommandBuffer, const VkDependencyInfo *)",
  "PFN_vkCmdWriteTimestamp2": "void (*)(VkCommandBuffer, VkPipelineStageFlags2, VkQueryPool, uint32_t)",
  "PFN_vkQueueSubmit2": "VkResult (*)(VkQueue, uint32_t, const VkSubmitInfo2 *, VkFence)",
  "PFN_vkCmdCopyBuffer2": "void (*)(VkCommandBuffer, const VkCopyBufferInfo2 *)",
  "PFN_vkCmdCopyImage2": "void (*)(VkCommandBuffer, const VkCopyImageInfo2 *)",
  "PFN_vkCmdCopyBufferToImage2": "void (*)(VkCommandBuffer, const VkCopyBufferToImageInfo2 *)",
  "PFN_vkCmdCopyImageToBuffer2": "void (*)(VkCommandBuffer, const VkCopyImageToBufferInfo2 *)",
  "PFN_vkCmdBlitImage2": "void (*)(VkCommandBuffer, const VkBlitImageInfo2 *)",
  "PFN_vkCmdResolveImage2": "void (*)(VkCommandBuffer, const VkResolveImageInfo2 *)",
  "PFN_vkCmdBeginRendering": "void (*)(VkCommandBuffer, const VkRenderingInfo *)",
  "PFN_vkCmdEndRendering": "void (*)(VkCommandBuffer)",
  "PFN_vkCmdSetCullMode": "void (*)(VkCommandBuffer, VkCullModeFlags)",
  "PFN_vkCmdSetFrontFace": "void (*)(VkCommandBuffer, VkFrontFace)",
  "PFN_vkCmdSetPrimitiveTopology": "void (*)(VkCommandBuffer, VkPrimitiveTopology)",
  "PFN_vkCmdSetViewportWithCount": "void (*)(VkCommandBuffer, uint32_t, const VkViewport *)",
  "PFN_vkCmdSetScissorWithCount": "void (*)(VkCommandBuffer, uint32_t, const VkRect2D *)",
  "PFN_vkCmdBindVertexBuffers2": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer *, const VkDeviceSize *, const VkDeviceSize *, const VkDeviceSize *)",
  "PFN_vkCmdSetDepthTestEnable": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetDepthWriteEnable": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetDepthCompareOp": "void (*)(VkCommandBuffer, VkCompareOp)",
  "PFN_vkCmdSetDepthBoundsTestEnable": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetStencilTestEnable": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetStencilOp": "void (*)(VkCommandBuffer, VkStencilFaceFlags, VkStencilOp, VkStencilOp, VkStencilOp, VkCompareOp)",
  "PFN_vkCmdSetRasterizerDiscardEnable": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetDepthBiasEnable": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetPrimitiveRestartEnable": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkGetDeviceBufferMemoryRequirements": "void (*)(VkDevice, const VkDeviceBufferMemoryRequirements *, VkMemoryRequirements2 *)",
  "PFN_vkGetDeviceImageMemoryRequirements": "void (*)(VkDevice, const VkDeviceImageMemoryRequirements *, VkMemoryRequirements2 *)",
  "PFN_vkGetDeviceImageSparseMemoryRequirements": "void (*)(VkDevice, const VkDeviceImageMemoryRequirements *, uint32_t *, VkSparseImageMemoryRequirements2 *)",
  "VkSurfaceKHR": "struct VkSurfaceKHR_T *",
  "VkPresentModeKHR": "enum VkPresentModeKHR",
  "VkColorSpaceKHR": "enum VkColorSpaceKHR",
  "VkSurfaceTransformFlagBitsKHR": "enum VkSurfaceTransformFlagBitsKHR",
  "VkCompositeAlphaFlagBitsKHR": "enum VkCompositeAlphaFlagBitsKHR",
  "VkCompositeAlphaFlagsKHR": "VkFlags",
  "VkSurfaceTransformFlagsKHR": "VkFlags",
  "VkSurfaceCapabilitiesKHR": "struct VkSurfaceCapabilitiesKHR",
  "VkSurfaceFormatKHR": "struct VkSurfaceFormatKHR",
  "PFN_vkDestroySurfaceKHR": "void (*)(VkInstance, VkSurfaceKHR, const VkAllocationCallbacks *)",
  "PFN_vkGetPhysicalDeviceSurfaceSupportKHR": "VkResult (*)(VkPhysicalDevice, uint32_t, VkSurfaceKHR, VkBool32 *)",
  "PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR": "VkResult (*)(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR *)",
  "PFN_vkGetPhysicalDeviceSurfaceFormatsKHR": "VkResult (*)(VkPhysicalDevice, VkSurfaceKHR, uint32_t *, VkSurfaceFormatKHR *)",
  "PFN_vkGetPhysicalDeviceSurfacePresentModesKHR": "VkResult (*)(VkPhysicalDevice, VkSurfaceKHR, uint32_t *, VkPresentModeKHR *)",
  "VkSwapchainKHR": "struct VkSwapchainKHR_T *",
  "VkSwapchainCreateFlagBitsKHR": "enum VkSwapchainCreateFlagBitsKHR",
  "VkSwapchainCreateFlagsKHR": "VkFlags",
  "VkDeviceGroupPresentModeFlagBitsKHR": "enum VkDeviceGroupPresentModeFlagBitsKHR",
  "VkDeviceGroupPresentModeFlagsKHR": "VkFlags",
  "VkSwapchainCreateInfoKHR": "struct VkSwapchainCreateInfoKHR",
  "VkPresentInfoKHR": "struct VkPresentInfoKHR",
  "VkImageSwapchainCreateInfoKHR": "struct VkImageSwapchainCreateInfoKHR",
  "VkBindImageMemorySwapchainInfoKHR": "struct VkBindImageMemorySwapchainInfoKHR",
  "VkAcquireNextImageInfoKHR": "struct VkAcquireNextImageInfoKHR",
  "VkDeviceGroupPresentCapabilitiesKHR": "struct VkDeviceGroupPresentCapabilitiesKHR",
  "VkDeviceGroupPresentInfoKHR": "struct VkDeviceGroupPresentInfoKHR",
  "VkDeviceGroupSwapchainCreateInfoKHR": "struct VkDeviceGroupSwapchainCreateInfoKHR",
  "PFN_vkCreateSwapchainKHR": "VkResult (*)(VkDevice, const VkSwapchainCreateInfoKHR *, const VkAllocationCallbacks *, VkSwapchainKHR *)",
  "PFN_vkDestroySwapchainKHR": "void (*)(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks *)",
  "PFN_vkGetSwapchainImagesKHR": "VkResult (*)(VkDevice, VkSwapchainKHR, uint32_t *, VkImage *)",
  "PFN_vkAcquireNextImageKHR": "VkResult (*)(VkDevice, VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t *)",
  "PFN_vkQueuePresentKHR": "VkResult (*)(VkQueue, const VkPresentInfoKHR *)",
  "PFN_vkGetDeviceGroupPresentCapabilitiesKHR": "VkResult (*)(VkDevice, VkDeviceGroupPresentCapabilitiesKHR *)",
  "PFN_vkGetDeviceGroupSurfacePresentModesKHR": "VkResult (*)(VkDevice, VkSurfaceKHR, VkDeviceGroupPresentModeFlagsKHR *)",
  "PFN_vkGetPhysicalDevicePresentRectanglesKHR": "VkResult (*)(VkPhysicalDevice, VkSurfaceKHR, uint32_t *, VkRect2D *)",
  "PFN_vkAcquireNextImage2KHR": "VkResult (*)(VkDevice, const VkAcquireNextImageInfoKHR *, uint32_t *)",
  "VkDisplayKHR": "struct VkDisplayKHR_T *",
  "VkDisplayModeKHR": "struct VkDisplayModeKHR_T *",
  "VkDisplayModeCreateFlagsKHR": "VkFlags",
  "VkDisplayPlaneAlphaFlagBitsKHR": "enum VkDisplayPlaneAlphaFlagBitsKHR",
  "VkDisplayPlaneAlphaFlagsKHR": "VkFlags",
  "VkDisplaySurfaceCreateFlagsKHR": "VkFlags",
  "VkDisplayModeParametersKHR": "struct VkDisplayModeParametersKHR",
  "VkDisplayModeCreateInfoKHR": "struct VkDisplayModeCreateInfoKHR",
  "VkDisplayModePropertiesKHR": "struct VkDisplayModePropertiesKHR",
  "VkDisplayPlaneCapabilitiesKHR": "struct VkDisplayPlaneCapabilitiesKHR",
  "VkDisplayPlanePropertiesKHR": "struct VkDisplayPlanePropertiesKHR",
  "VkDisplayPropertiesKHR": "struct VkDisplayPropertiesKHR",
  "VkDisplaySurfaceCreateInfoKHR": "struct VkDisplaySurfaceCreateInfoKHR",
  "PFN_vkGetPhysicalDeviceDisplayPropertiesKHR": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkDisplayPropertiesKHR *)",
  "PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkDisplayPlanePropertiesKHR *)",
  "PFN_vkGetDisplayPlaneSupportedDisplaysKHR": "VkResult (*)(VkPhysicalDevice, uint32_t, uint32_t *, VkDisplayKHR *)",
  "PFN_vkGetDisplayModePropertiesKHR": "VkResult (*)(VkPhysicalDevice, VkDisplayKHR, uint32_t *, VkDisplayModePropertiesKHR *)",
  "PFN_vkCreateDisplayModeKHR": "VkResult (*)(VkPhysicalDevice, VkDisplayKHR, const VkDisplayModeCreateInfoKHR *, const VkAllocationCallbacks *, VkDisplayModeKHR *)",
  "PFN_vkGetDisplayPlaneCapabilitiesKHR": "VkResult (*)(VkPhysicalDevice, VkDisplayModeKHR, uint32_t, VkDisplayPlaneCapabilitiesKHR *)",
  "PFN_vkCreateDisplayPlaneSurfaceKHR": "VkResult (*)(VkInstance, const VkDisplaySurfaceCreateInfoKHR *, const VkAllocationCallbacks *, VkSurfaceKHR *)",
  "VkDisplayPresentInfoKHR": "struct VkDisplayPresentInfoKHR",
  "PFN_vkCreateSharedSwapchainsKHR": "VkResult (*)(VkDevice, uint32_t, const VkSwapchainCreateInfoKHR *, const VkAllocationCallbacks *, VkSwapchainKHR *)",
  "VkRenderingFlagsKHR": "VkRenderingFlags",
  "VkRenderingFlagBitsKHR": "VkRenderingFlagBits",
  "VkRenderingInfoKHR": "VkRenderingInfo",
  "VkRenderingAttachmentInfoKHR": "VkRenderingAttachmentInfo",
  "VkPipelineRenderingCreateInfoKHR": "VkPipelineRenderingCreateInfo",
  "VkPhysicalDeviceDynamicRenderingFeaturesKHR": "VkPhysicalDeviceDynamicRenderingFeatures",
  "VkCommandBufferInheritanceRenderingInfoKHR": "VkCommandBufferInheritanceRenderingInfo",
  "VkRenderingFragmentShadingRateAttachmentInfoKHR": "struct VkRenderingFragmentShadingRateAttachmentInfoKHR",
  "VkRenderingFragmentDensityMapAttachmentInfoEXT": "struct VkRenderingFragmentDensityMapAttachmentInfoEXT",
  "VkAttachmentSampleCountInfoAMD": "struct VkAttachmentSampleCountInfoAMD",
  "VkAttachmentSampleCountInfoNV": "VkAttachmentSampleCountInfoAMD",
  "VkMultiviewPerViewAttributesInfoNVX": "struct VkMultiviewPerViewAttributesInfoNVX",
  "PFN_vkCmdBeginRenderingKHR": "void (*)(VkCommandBuffer, const VkRenderingInfo *)",
  "PFN_vkCmdEndRenderingKHR": "void (*)(VkCommandBuffer)",
  "VkRenderPassMultiviewCreateInfoKHR": "VkRenderPassMultiviewCreateInfo",
  "VkPhysicalDeviceMultiviewFeaturesKHR": "VkPhysicalDeviceMultiviewFeatures",
  "VkPhysicalDeviceMultiviewPropertiesKHR": "VkPhysicalDeviceMultiviewProperties",
  "VkPhysicalDeviceFeatures2KHR": "VkPhysicalDeviceFeatures2",
  "VkPhysicalDeviceProperties2KHR": "VkPhysicalDeviceProperties2",
  "VkFormatProperties2KHR": "VkFormatProperties2",
  "VkImageFormatProperties2KHR": "VkImageFormatProperties2",
  "VkPhysicalDeviceImageFormatInfo2KHR": "VkPhysicalDeviceImageFormatInfo2",
  "VkQueueFamilyProperties2KHR": "VkQueueFamilyProperties2",
  "VkPhysicalDeviceMemoryProperties2KHR": "VkPhysicalDeviceMemoryProperties2",
  "VkSparseImageFormatProperties2KHR": "VkSparseImageFormatProperties2",
  "VkPhysicalDeviceSparseImageFormatInfo2KHR": "VkPhysicalDeviceSparseImageFormatInfo2",
  "PFN_vkGetPhysicalDeviceFeatures2KHR": "void (*)(VkPhysicalDevice, VkPhysicalDeviceFeatures2 *)",
  "PFN_vkGetPhysicalDeviceProperties2KHR": "void (*)(VkPhysicalDevice, VkPhysicalDeviceProperties2 *)",
  "PFN_vkGetPhysicalDeviceFormatProperties2KHR": "void (*)(VkPhysicalDevice, VkFormat, VkFormatProperties2 *)",
  "PFN_vkGetPhysicalDeviceImageFormatProperties2KHR": "VkResult (*)(VkPhysicalDevice, const VkPhysicalDeviceImageFormatInfo2 *, VkImageFormatProperties2 *)",
  "PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR": "void (*)(VkPhysicalDevice, uint32_t *, VkQueueFamilyProperties2 *)",
  "PFN_vkGetPhysicalDeviceMemoryProperties2KHR": "void (*)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2 *)",
  "PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR": "void (*)(VkPhysicalDevice, const VkPhysicalDeviceSparseImageFormatInfo2 *, uint32_t *, VkSparseImageFormatProperties2 *)",
  "VkPeerMemoryFeatureFlagsKHR": "VkPeerMemoryFeatureFlags",
  "VkPeerMemoryFeatureFlagBitsKHR": "VkPeerMemoryFeatureFlagBits",
  "VkMemoryAllocateFlagsKHR": "VkMemoryAllocateFlags",
  "VkMemoryAllocateFlagBitsKHR": "VkMemoryAllocateFlagBits",
  "VkMemoryAllocateFlagsInfoKHR": "VkMemoryAllocateFlagsInfo",
  "VkDeviceGroupRenderPassBeginInfoKHR": "VkDeviceGroupRenderPassBeginInfo",
  "VkDeviceGroupCommandBufferBeginInfoKHR": "VkDeviceGroupCommandBufferBeginInfo",
  "VkDeviceGroupSubmitInfoKHR": "VkDeviceGroupSubmitInfo",
  "VkDeviceGroupBindSparseInfoKHR": "VkDeviceGroupBindSparseInfo",
  "VkBindBufferMemoryDeviceGroupInfoKHR": "VkBindBufferMemoryDeviceGroupInfo",
  "VkBindImageMemoryDeviceGroupInfoKHR": "VkBindImageMemoryDeviceGroupInfo",
  "PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR": "void (*)(VkDevice, uint32_t, uint32_t, uint32_t, VkPeerMemoryFeatureFlags *)",
  "PFN_vkCmdSetDeviceMaskKHR": "void (*)(VkCommandBuffer, uint32_t)",
  "PFN_vkCmdDispatchBaseKHR": "void (*)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)",
  "VkCommandPoolTrimFlagsKHR": "VkCommandPoolTrimFlags",
  "PFN_vkTrimCommandPoolKHR": "void (*)(VkDevice, VkCommandPool, VkCommandPoolTrimFlags)",
  "VkPhysicalDeviceGroupPropertiesKHR": "VkPhysicalDeviceGroupProperties",
  "VkDeviceGroupDeviceCreateInfoKHR": "VkDeviceGroupDeviceCreateInfo",
  "PFN_vkEnumeratePhysicalDeviceGroupsKHR": "VkResult (*)(VkInstance, uint32_t *, VkPhysicalDeviceGroupProperties *)",
  "VkExternalMemoryHandleTypeFlagsKHR": "VkExternalMemoryHandleTypeFlags",
  "VkExternalMemoryHandleTypeFlagBitsKHR": "VkExternalMemoryHandleTypeFlagBits",
  "VkExternalMemoryFeatureFlagsKHR": "VkExternalMemoryFeatureFlags",
  "VkExternalMemoryFeatureFlagBitsKHR": "VkExternalMemoryFeatureFlagBits",
  "VkExternalMemoryPropertiesKHR": "VkExternalMemoryProperties",
  "VkPhysicalDeviceExternalImageFormatInfoKHR": "VkPhysicalDeviceExternalImageFormatInfo",
  "VkExternalImageFormatPropertiesKHR": "VkExternalImageFormatProperties",
  "VkPhysicalDeviceExternalBufferInfoKHR": "VkPhysicalDeviceExternalBufferInfo",
  "VkExternalBufferPropertiesKHR": "VkExternalBufferProperties",
  "VkPhysicalDeviceIDPropertiesKHR": "VkPhysicalDeviceIDProperties",
  "PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR": "void (*)(VkPhysicalDevice, const VkPhysicalDeviceExternalBufferInfo *, VkExternalBufferProperties *)",
  "VkExternalMemoryImageCreateInfoKHR": "VkExternalMemoryImageCreateInfo",
  "VkExternalMemoryBufferCreateInfoKHR": "VkExternalMemoryBufferCreateInfo",
  "VkExportMemoryAllocateInfoKHR": "VkExportMemoryAllocateInfo",
  "VkImportMemoryFdInfoKHR": "struct VkImportMemoryFdInfoKHR",
  "VkMemoryFdPropertiesKHR": "struct VkMemoryFdPropertiesKHR",
  "VkMemoryGetFdInfoKHR": "struct VkMemoryGetFdInfoKHR",
  "PFN_vkGetMemoryFdKHR": "VkResult (*)(VkDevice, const VkMemoryGetFdInfoKHR *, int *)",
  "PFN_vkGetMemoryFdPropertiesKHR": "VkResult (*)(VkDevice, VkExternalMemoryHandleTypeFlagBits, int, VkMemoryFdPropertiesKHR *)",
  "VkExternalSemaphoreHandleTypeFlagsKHR": "VkExternalSemaphoreHandleTypeFlags",
  "VkExternalSemaphoreHandleTypeFlagBitsKHR": "VkExternalSemaphoreHandleTypeFlagBits",
  "VkExternalSemaphoreFeatureFlagsKHR": "VkExternalSemaphoreFeatureFlags",
  "VkExternalSemaphoreFeatureFlagBitsKHR": "VkExternalSemaphoreFeatureFlagBits",
  "VkPhysicalDeviceExternalSemaphoreInfoKHR": "VkPhysicalDeviceExternalSemaphoreInfo",
  "VkExternalSemaphorePropertiesKHR": "VkExternalSemaphoreProperties",
  "PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR": "void (*)(VkPhysicalDevice, const VkPhysicalDeviceExternalSemaphoreInfo *, VkExternalSemaphoreProperties *)",
  "VkSemaphoreImportFlagsKHR": "VkSemaphoreImportFlags",
  "VkSemaphoreImportFlagBitsKHR": "VkSemaphoreImportFlagBits",
  "VkExportSemaphoreCreateInfoKHR": "VkExportSemaphoreCreateInfo",
  "VkImportSemaphoreFdInfoKHR": "struct VkImportSemaphoreFdInfoKHR",
  "VkSemaphoreGetFdInfoKHR": "struct VkSemaphoreGetFdInfoKHR",
  "PFN_vkImportSemaphoreFdKHR": "VkResult (*)(VkDevice, const VkImportSemaphoreFdInfoKHR *)",
  "PFN_vkGetSemaphoreFdKHR": "VkResult (*)(VkDevice, const VkSemaphoreGetFdInfoKHR *, int *)",
  "VkPhysicalDevicePushDescriptorPropertiesKHR": "struct VkPhysicalDevicePushDescriptorPropertiesKHR",
  "PFN_vkCmdPushDescriptorSetKHR": "void (*)(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkWriteDescriptorSet *)",
  "PFN_vkCmdPushDescriptorSetWithTemplateKHR": "void (*)(VkCommandBuffer, VkDescriptorUpdateTemplate, VkPipelineLayout, uint32_t, const void *)",
  "VkPhysicalDeviceShaderFloat16Int8FeaturesKHR": "VkPhysicalDeviceShaderFloat16Int8Features",
  "VkPhysicalDeviceFloat16Int8FeaturesKHR": "VkPhysicalDeviceShaderFloat16Int8Features",
  "VkPhysicalDevice16BitStorageFeaturesKHR": "VkPhysicalDevice16BitStorageFeatures",
  "VkRectLayerKHR": "struct VkRectLayerKHR",
  "VkPresentRegionKHR": "struct VkPresentRegionKHR",
  "VkPresentRegionsKHR": "struct VkPresentRegionsKHR",
  "VkDescriptorUpdateTemplateKHR": "VkDescriptorUpdateTemplate",
  "VkDescriptorUpdateTemplateTypeKHR": "VkDescriptorUpdateTemplateType",
  "VkDescriptorUpdateTemplateCreateFlagsKHR": "VkDescriptorUpdateTemplateCreateFlags",
  "VkDescriptorUpdateTemplateEntryKHR": "VkDescriptorUpdateTemplateEntry",
  "VkDescriptorUpdateTemplateCreateInfoKHR": "VkDescriptorUpdateTemplateCreateInfo",
  "PFN_vkCreateDescriptorUpdateTemplateKHR": "VkResult (*)(VkDevice, const VkDescriptorUpdateTemplateCreateInfo *, const VkAllocationCallbacks *, VkDescriptorUpdateTemplate *)",
  "PFN_vkDestroyDescriptorUpdateTemplateKHR": "void (*)(VkDevice, VkDescriptorUpdateTemplate, const VkAllocationCallbacks *)",
  "PFN_vkUpdateDescriptorSetWithTemplateKHR": "void (*)(VkDevice, VkDescriptorSet, VkDescriptorUpdateTemplate, const void *)",
  "VkPhysicalDeviceImagelessFramebufferFeaturesKHR": "VkPhysicalDeviceImagelessFramebufferFeatures",
  "VkFramebufferAttachmentsCreateInfoKHR": "VkFramebufferAttachmentsCreateInfo",
  "VkFramebufferAttachmentImageInfoKHR": "VkFramebufferAttachmentImageInfo",
  "VkRenderPassAttachmentBeginInfoKHR": "VkRenderPassAttachmentBeginInfo",
  "VkRenderPassCreateInfo2KHR": "VkRenderPassCreateInfo2",
  "VkAttachmentDescription2KHR": "VkAttachmentDescription2",
  "VkAttachmentReference2KHR": "VkAttachmentReference2",
  "VkSubpassDescription2KHR": "VkSubpassDescription2",
  "VkSubpassDependency2KHR": "VkSubpassDependency2",
  "VkSubpassBeginInfoKHR": "VkSubpassBeginInfo",
  "VkSubpassEndInfoKHR": "VkSubpassEndInfo",
  "PFN_vkCreateRenderPass2KHR": "VkResult (*)(VkDevice, const VkRenderPassCreateInfo2 *, const VkAllocationCallbacks *, VkRenderPass *)",
  "PFN_vkCmdBeginRenderPass2KHR": "void (*)(VkCommandBuffer, const VkRenderPassBeginInfo *, const VkSubpassBeginInfo *)",
  "PFN_vkCmdNextSubpass2KHR": "void (*)(VkCommandBuffer, const VkSubpassBeginInfo *, const VkSubpassEndInfo *)",
  "PFN_vkCmdEndRenderPass2KHR": "void (*)(VkCommandBuffer, const VkSubpassEndInfo *)",
  "VkSharedPresentSurfaceCapabilitiesKHR": "struct VkSharedPresentSurfaceCapabilitiesKHR",
  "PFN_vkGetSwapchainStatusKHR": "VkResult (*)(VkDevice, VkSwapchainKHR)",
  "VkExternalFenceHandleTypeFlagsKHR": "VkExternalFenceHandleTypeFlags",
  "VkExternalFenceHandleTypeFlagBitsKHR": "VkExternalFenceHandleTypeFlagBits",
  "VkExternalFenceFeatureFlagsKHR": "VkExternalFenceFeatureFlags",
  "VkExternalFenceFeatureFlagBitsKHR": "VkExternalFenceFeatureFlagBits",
  "VkPhysicalDeviceExternalFenceInfoKHR": "VkPhysicalDeviceExternalFenceInfo",
  "VkExternalFencePropertiesKHR": "VkExternalFenceProperties",
  "PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR": "void (*)(VkPhysicalDevice, const VkPhysicalDeviceExternalFenceInfo *, VkExternalFenceProperties *)",
  "VkFenceImportFlagsKHR": "VkFenceImportFlags",
  "VkFenceImportFlagBitsKHR": "VkFenceImportFlagBits",
  "VkExportFenceCreateInfoKHR": "VkExportFenceCreateInfo",
  "VkImportFenceFdInfoKHR": "struct VkImportFenceFdInfoKHR",
  "VkFenceGetFdInfoKHR": "struct VkFenceGetFdInfoKHR",
  "PFN_vkImportFenceFdKHR": "VkResult (*)(VkDevice, const VkImportFenceFdInfoKHR *)",
  "PFN_vkGetFenceFdKHR": "VkResult (*)(VkDevice, const VkFenceGetFdInfoKHR *, int *)",
  "VkPerformanceCounterUnitKHR": "enum VkPerformanceCounterUnitKHR",
  "VkPerformanceCounterScopeKHR": "enum VkPerformanceCounterScopeKHR",
  "VkPerformanceCounterStorageKHR": "enum VkPerformanceCounterStorageKHR",
  "VkPerformanceCounterDescriptionFlagBitsKHR": "enum VkPerformanceCounterDescriptionFlagBitsKHR",
  "VkPerformanceCounterDescriptionFlagsKHR": "VkFlags",
  "VkAcquireProfilingLockFlagBitsKHR": "enum VkAcquireProfilingLockFlagBitsKHR",
  "VkAcquireProfilingLockFlagsKHR": "VkFlags",
  "VkPhysicalDevicePerformanceQueryFeaturesKHR": "struct VkPhysicalDevicePerformanceQueryFeaturesKHR",
  "VkPhysicalDevicePerformanceQueryPropertiesKHR": "struct VkPhysicalDevicePerformanceQueryPropertiesKHR",
  "VkPerformanceCounterKHR": "struct VkPerformanceCounterKHR",
  "VkPerformanceCounterDescriptionKHR": "struct VkPerformanceCounterDescriptionKHR",
  "VkQueryPoolPerformanceCreateInfoKHR": "struct VkQueryPoolPerformanceCreateInfoKHR",
  "VkPerformanceCounterResultKHR": "union VkPerformanceCounterResultKHR",
  "VkAcquireProfilingLockInfoKHR": "struct VkAcquireProfilingLockInfoKHR",
  "VkPerformanceQuerySubmitInfoKHR": "struct VkPerformanceQuerySubmitInfoKHR",
  "PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR": "VkResult (*)(VkPhysicalDevice, uint32_t, uint32_t *, VkPerformanceCounterKHR *, VkPerformanceCounterDescriptionKHR *)",
  "PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR": "void (*)(VkPhysicalDevice, const VkQueryPoolPerformanceCreateInfoKHR *, uint32_t *)",
  "PFN_vkAcquireProfilingLockKHR": "VkResult (*)(VkDevice, const VkAcquireProfilingLockInfoKHR *)",
  "PFN_vkReleaseProfilingLockKHR": "void (*)(VkDevice)",
  "VkPointClippingBehaviorKHR": "VkPointClippingBehavior",
  "VkTessellationDomainOriginKHR": "VkTessellationDomainOrigin",
  "VkPhysicalDevicePointClippingPropertiesKHR": "VkPhysicalDevicePointClippingProperties",
  "VkRenderPassInputAttachmentAspectCreateInfoKHR": "VkRenderPassInputAttachmentAspectCreateInfo",
  "VkInputAttachmentAspectReferenceKHR": "VkInputAttachmentAspectReference",
  "VkImageViewUsageCreateInfoKHR": "VkImageViewUsageCreateInfo",
  "VkPipelineTessellationDomainOriginStateCreateInfoKHR": "VkPipelineTessellationDomainOriginStateCreateInfo",
  "VkPhysicalDeviceSurfaceInfo2KHR": "struct VkPhysicalDeviceSurfaceInfo2KHR",
  "VkSurfaceCapabilities2KHR": "struct VkSurfaceCapabilities2KHR",
  "VkSurfaceFormat2KHR": "struct VkSurfaceFormat2KHR",
  "PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR": "VkResult (*)(VkPhysicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR *, VkSurfaceCapabilities2KHR *)",
  "PFN_vkGetPhysicalDeviceSurfaceFormats2KHR": "VkResult (*)(VkPhysicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR *, uint32_t *, VkSurfaceFormat2KHR *)",
  "VkPhysicalDeviceVariablePointerFeaturesKHR": "VkPhysicalDeviceVariablePointersFeatures",
  "VkPhysicalDeviceVariablePointersFeaturesKHR": "VkPhysicalDeviceVariablePointersFeatures",
  "VkDisplayProperties2KHR": "struct VkDisplayProperties2KHR",
  "VkDisplayPlaneProperties2KHR": "struct VkDisplayPlaneProperties2KHR",
  "VkDisplayModeProperties2KHR": "struct VkDisplayModeProperties2KHR",
  "VkDisplayPlaneInfo2KHR": "struct VkDisplayPlaneInfo2KHR",
  "VkDisplayPlaneCapabilities2KHR": "struct VkDisplayPlaneCapabilities2KHR",
  "PFN_vkGetPhysicalDeviceDisplayProperties2KHR": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkDisplayProperties2KHR *)",
  "PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkDisplayPlaneProperties2KHR *)",
  "PFN_vkGetDisplayModeProperties2KHR": "VkResult (*)(VkPhysicalDevice, VkDisplayKHR, uint32_t *, VkDisplayModeProperties2KHR *)",
  "PFN_vkGetDisplayPlaneCapabilities2KHR": "VkResult (*)(VkPhysicalDevice, const VkDisplayPlaneInfo2KHR *, VkDisplayPlaneCapabilities2KHR *)",
  "VkMemoryDedicatedRequirementsKHR": "VkMemoryDedicatedRequirements",
  "VkMemoryDedicatedAllocateInfoKHR": "VkMemoryDedicatedAllocateInfo",
  "VkBufferMemoryRequirementsInfo2KHR": "VkBufferMemoryRequirementsInfo2",
  "VkImageMemoryRequirementsInfo2KHR": "VkImageMemoryRequirementsInfo2",
  "VkImageSparseMemoryRequirementsInfo2KHR": "VkImageSparseMemoryRequirementsInfo2",
  "VkMemoryRequirements2KHR": "VkMemoryRequirements2",
  "VkSparseImageMemoryRequirements2KHR": "VkSparseImageMemoryRequirements2",
  "PFN_vkGetImageMemoryRequirements2KHR": "void (*)(VkDevice, const VkImageMemoryRequirementsInfo2 *, VkMemoryRequirements2 *)",
  "PFN_vkGetBufferMemoryRequirements2KHR": "void (*)(VkDevice, const VkBufferMemoryRequirementsInfo2 *, VkMemoryRequirements2 *)",
  "PFN_vkGetImageSparseMemoryRequirements2KHR": "void (*)(VkDevice, const VkImageSparseMemoryRequirementsInfo2 *, uint32_t *, VkSparseImageMemoryRequirements2 *)",
  "VkImageFormatListCreateInfoKHR": "VkImageFormatListCreateInfo",
  "VkSamplerYcbcrConversionKHR": "VkSamplerYcbcrConversion",
  "VkSamplerYcbcrModelConversionKHR": "VkSamplerYcbcrModelConversion",
  "VkSamplerYcbcrRangeKHR": "VkSamplerYcbcrRange",
  "VkChromaLocationKHR": "VkChromaLocation",
  "VkSamplerYcbcrConversionCreateInfoKHR": "VkSamplerYcbcrConversionCreateInfo",
  "VkSamplerYcbcrConversionInfoKHR": "VkSamplerYcbcrConversionInfo",
  "VkBindImagePlaneMemoryInfoKHR": "VkBindImagePlaneMemoryInfo",
  "VkImagePlaneMemoryRequirementsInfoKHR": "VkImagePlaneMemoryRequirementsInfo",
  "VkPhysicalDeviceSamplerYcbcrConversionFeaturesKHR": "VkPhysicalDeviceSamplerYcbcrConversionFeatures",
  "VkSamplerYcbcrConversionImageFormatPropertiesKHR": "VkSamplerYcbcrConversionImageFormatProperties",
  "PFN_vkCreateSamplerYcbcrConversionKHR": "VkResult (*)(VkDevice, const VkSamplerYcbcrConversionCreateInfo *, const VkAllocationCallbacks *, VkSamplerYcbcrConversion *)",
  "PFN_vkDestroySamplerYcbcrConversionKHR": "void (*)(VkDevice, VkSamplerYcbcrConversion, const VkAllocationCallbacks *)",
  "VkBindBufferMemoryInfoKHR": "VkBindBufferMemoryInfo",
  "VkBindImageMemoryInfoKHR": "VkBindImageMemoryInfo",
  "PFN_vkBindBufferMemory2KHR": "VkResult (*)(VkDevice, uint32_t, const VkBindBufferMemoryInfo *)",
  "PFN_vkBindImageMemory2KHR": "VkResult (*)(VkDevice, uint32_t, const VkBindImageMemoryInfo *)",
  "VkPhysicalDeviceMaintenance3PropertiesKHR": "VkPhysicalDeviceMaintenance3Properties",
  "VkDescriptorSetLayoutSupportKHR": "VkDescriptorSetLayoutSupport",
  "PFN_vkGetDescriptorSetLayoutSupportKHR": "void (*)(VkDevice, const VkDescriptorSetLayoutCreateInfo *, VkDescriptorSetLayoutSupport *)",
  "PFN_vkCmdDrawIndirectCountKHR": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "PFN_vkCmdDrawIndexedIndirectCountKHR": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR": "VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures",
  "VkPhysicalDevice8BitStorageFeaturesKHR": "VkPhysicalDevice8BitStorageFeatures",
  "VkPhysicalDeviceShaderAtomicInt64FeaturesKHR": "VkPhysicalDeviceShaderAtomicInt64Features",
  "VkPhysicalDeviceShaderClockFeaturesKHR": "struct VkPhysicalDeviceShaderClockFeaturesKHR",
  "VkQueueGlobalPriorityKHR": "enum VkQueueGlobalPriorityKHR",
  "VkDeviceQueueGlobalPriorityCreateInfoKHR": "struct VkDeviceQueueGlobalPriorityCreateInfoKHR",
  "VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR": "struct VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR",
  "VkQueueFamilyGlobalPriorityPropertiesKHR": "struct VkQueueFamilyGlobalPriorityPropertiesKHR",
  "VkDriverIdKHR": "VkDriverId",
  "VkConformanceVersionKHR": "VkConformanceVersion",
  "VkPhysicalDeviceDriverPropertiesKHR": "VkPhysicalDeviceDriverProperties",
  "VkShaderFloatControlsIndependenceKHR": "VkShaderFloatControlsIndependence",
  "VkPhysicalDeviceFloatControlsPropertiesKHR": "VkPhysicalDeviceFloatControlsProperties",
  "VkResolveModeFlagBitsKHR": "VkResolveModeFlagBits",
  "VkResolveModeFlagsKHR": "VkResolveModeFlags",
  "VkSubpassDescriptionDepthStencilResolveKHR": "VkSubpassDescriptionDepthStencilResolve",
  "VkPhysicalDeviceDepthStencilResolvePropertiesKHR": "VkPhysicalDeviceDepthStencilResolveProperties",
  "VkSemaphoreTypeKHR": "VkSemaphoreType",
  "VkSemaphoreWaitFlagBitsKHR": "VkSemaphoreWaitFlagBits",
  "VkSemaphoreWaitFlagsKHR": "VkSemaphoreWaitFlags",
  "VkPhysicalDeviceTimelineSemaphoreFeaturesKHR": "VkPhysicalDeviceTimelineSemaphoreFeatures",
  "VkPhysicalDeviceTimelineSemaphorePropertiesKHR": "VkPhysicalDeviceTimelineSemaphoreProperties",
  "VkSemaphoreTypeCreateInfoKHR": "VkSemaphoreTypeCreateInfo",
  "VkTimelineSemaphoreSubmitInfoKHR": "VkTimelineSemaphoreSubmitInfo",
  "VkSemaphoreWaitInfoKHR": "VkSemaphoreWaitInfo",
  "VkSemaphoreSignalInfoKHR": "VkSemaphoreSignalInfo",
  "PFN_vkGetSemaphoreCounterValueKHR": "VkResult (*)(VkDevice, VkSemaphore, uint64_t *)",
  "PFN_vkWaitSemaphoresKHR": "VkResult (*)(VkDevice, const VkSemaphoreWaitInfo *, uint64_t)",
  "PFN_vkSignalSemaphoreKHR": "VkResult (*)(VkDevice, const VkSemaphoreSignalInfo *)",
  "VkPhysicalDeviceVulkanMemoryModelFeaturesKHR": "VkPhysicalDeviceVulkanMemoryModelFeatures",
  "VkPhysicalDeviceShaderTerminateInvocationFeaturesKHR": "VkPhysicalDeviceShaderTerminateInvocationFeatures",
  "VkFragmentShadingRateCombinerOpKHR": "enum VkFragmentShadingRateCombinerOpKHR",
  "VkFragmentShadingRateAttachmentInfoKHR": "struct VkFragmentShadingRateAttachmentInfoKHR",
  "VkPipelineFragmentShadingRateStateCreateInfoKHR": "struct VkPipelineFragmentShadingRateStateCreateInfoKHR",
  "VkPhysicalDeviceFragmentShadingRateFeaturesKHR": "struct VkPhysicalDeviceFragmentShadingRateFeaturesKHR",
  "VkPhysicalDeviceFragmentShadingRatePropertiesKHR": "struct VkPhysicalDeviceFragmentShadingRatePropertiesKHR",
  "VkPhysicalDeviceFragmentShadingRateKHR": "struct VkPhysicalDeviceFragmentShadingRateKHR",
  "PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkPhysicalDeviceFragmentShadingRateKHR *)",
  "PFN_vkCmdSetFragmentShadingRateKHR": "void (*)(VkCommandBuffer, const VkExtent2D *, const VkFragmentShadingRateCombinerOpKHR *)",
  "VkSurfaceProtectedCapabilitiesKHR": "struct VkSurfaceProtectedCapabilitiesKHR",
  "VkPhysicalDeviceSeparateDepthStencilLayoutsFeaturesKHR": "VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures",
  "VkAttachmentReferenceStencilLayoutKHR": "VkAttachmentReferenceStencilLayout",
  "VkAttachmentDescriptionStencilLayoutKHR": "VkAttachmentDescriptionStencilLayout",
  "VkPhysicalDevicePresentWaitFeaturesKHR": "struct VkPhysicalDevicePresentWaitFeaturesKHR",
  "PFN_vkWaitForPresentKHR": "VkResult (*)(VkDevice, VkSwapchainKHR, uint64_t, uint64_t)",
  "VkPhysicalDeviceUniformBufferStandardLayoutFeaturesKHR": "VkPhysicalDeviceUniformBufferStandardLayoutFeatures",
  "VkPhysicalDeviceBufferDeviceAddressFeaturesKHR": "VkPhysicalDeviceBufferDeviceAddressFeatures",
  "VkBufferDeviceAddressInfoKHR": "VkBufferDeviceAddressInfo",
  "VkBufferOpaqueCaptureAddressCreateInfoKHR": "VkBufferOpaqueCaptureAddressCreateInfo",
  "VkMemoryOpaqueCaptureAddressAllocateInfoKHR": "VkMemoryOpaqueCaptureAddressAllocateInfo",
  "VkDeviceMemoryOpaqueCaptureAddressInfoKHR": "VkDeviceMemoryOpaqueCaptureAddressInfo",
  "PFN_vkGetBufferDeviceAddressKHR": "VkDeviceAddress (*)(VkDevice, const VkBufferDeviceAddressInfo *)",
  "PFN_vkGetBufferOpaqueCaptureAddressKHR": "uint64_t (*)(VkDevice, const VkBufferDeviceAddressInfo *)",
  "PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR": "uint64_t (*)(VkDevice, const VkDeviceMemoryOpaqueCaptureAddressInfo *)",
  "VkDeferredOperationKHR": "struct VkDeferredOperationKHR_T *",
  "PFN_vkCreateDeferredOperationKHR": "VkResult (*)(VkDevice, const VkAllocationCallbacks *, VkDeferredOperationKHR *)",
  "PFN_vkDestroyDeferredOperationKHR": "void (*)(VkDevice, VkDeferredOperationKHR, const VkAllocationCallbacks *)",
  "PFN_vkGetDeferredOperationMaxConcurrencyKHR": "uint32_t (*)(VkDevice, VkDeferredOperationKHR)",
  "PFN_vkGetDeferredOperationResultKHR": "VkResult (*)(VkDevice, VkDeferredOperationKHR)",
  "PFN_vkDeferredOperationJoinKHR": "VkResult (*)(VkDevice, VkDeferredOperationKHR)",
  "VkPipelineExecutableStatisticFormatKHR": "enum VkPipelineExecutableStatisticFormatKHR",
  "VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR": "struct VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR",
  "VkPipelineInfoKHR": "struct VkPipelineInfoKHR",
  "VkPipelineExecutablePropertiesKHR": "struct VkPipelineExecutablePropertiesKHR",
  "VkPipelineExecutableInfoKHR": "struct VkPipelineExecutableInfoKHR",
  "VkPipelineExecutableStatisticValueKHR": "union VkPipelineExecutableStatisticValueKHR",
  "VkPipelineExecutableStatisticKHR": "struct VkPipelineExecutableStatisticKHR",
  "VkPipelineExecutableInternalRepresentationKHR": "struct VkPipelineExecutableInternalRepresentationKHR",
  "PFN_vkGetPipelineExecutablePropertiesKHR": "VkResult (*)(VkDevice, const VkPipelineInfoKHR *, uint32_t *, VkPipelineExecutablePropertiesKHR *)",
  "PFN_vkGetPipelineExecutableStatisticsKHR": "VkResult (*)(VkDevice, const VkPipelineExecutableInfoKHR *, uint32_t *, VkPipelineExecutableStatisticKHR *)",
  "PFN_vkGetPipelineExecutableInternalRepresentationsKHR": "VkResult (*)(VkDevice, const VkPipelineExecutableInfoKHR *, uint32_t *, VkPipelineExecutableInternalRepresentationKHR *)",
  "VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR": "VkPhysicalDeviceShaderIntegerDotProductFeatures",
  "VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR": "VkPhysicalDeviceShaderIntegerDotProductProperties",
  "VkPipelineLibraryCreateInfoKHR": "struct VkPipelineLibraryCreateInfoKHR",
  "VkPresentIdKHR": "struct VkPresentIdKHR",
  "VkPhysicalDevicePresentIdFeaturesKHR": "struct VkPhysicalDevicePresentIdFeaturesKHR",
  "VkPipelineStageFlags2KHR": "VkPipelineStageFlags2",
  "VkPipelineStageFlagBits2KHR": "VkPipelineStageFlagBits2",
  "VkAccessFlags2KHR": "VkAccessFlags2",
  "VkAccessFlagBits2KHR": "VkAccessFlagBits2",
  "VkSubmitFlagBitsKHR": "VkSubmitFlagBits",
  "VkSubmitFlagsKHR": "VkSubmitFlags",
  "VkMemoryBarrier2KHR": "VkMemoryBarrier2",
  "VkBufferMemoryBarrier2KHR": "VkBufferMemoryBarrier2",
  "VkImageMemoryBarrier2KHR": "VkImageMemoryBarrier2",
  "VkDependencyInfoKHR": "VkDependencyInfo",
  "VkSubmitInfo2KHR": "VkSubmitInfo2",
  "VkSemaphoreSubmitInfoKHR": "VkSemaphoreSubmitInfo",
  "VkCommandBufferSubmitInfoKHR": "VkCommandBufferSubmitInfo",
  "VkPhysicalDeviceSynchronization2FeaturesKHR": "VkPhysicalDeviceSynchronization2Features",
  "VkQueueFamilyCheckpointProperties2NV": "struct VkQueueFamilyCheckpointProperties2NV",
  "VkCheckpointData2NV": "struct VkCheckpointData2NV",
  "PFN_vkCmdSetEvent2KHR": "void (*)(VkCommandBuffer, VkEvent, const VkDependencyInfo *)",
  "PFN_vkCmdResetEvent2KHR": "void (*)(VkCommandBuffer, VkEvent, VkPipelineStageFlags2)",
  "PFN_vkCmdWaitEvents2KHR": "void (*)(VkCommandBuffer, uint32_t, const VkEvent *, const VkDependencyInfo *)",
  "PFN_vkCmdPipelineBarrier2KHR": "void (*)(VkCommandBuffer, const VkDependencyInfo *)",
  "PFN_vkCmdWriteTimestamp2KHR": "void (*)(VkCommandBuffer, VkPipelineStageFlags2, VkQueryPool, uint32_t)",
  "PFN_vkQueueSubmit2KHR": "VkResult (*)(VkQueue, uint32_t, const VkSubmitInfo2 *, VkFence)",
  "PFN_vkCmdWriteBufferMarker2AMD": "void (*)(VkCommandBuffer, VkPipelineStageFlags2, VkBuffer, VkDeviceSize, uint32_t)",
  "PFN_vkGetQueueCheckpointData2NV": "void (*)(VkQueue, uint32_t *, VkCheckpointData2NV *)",
  "VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR": "struct VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR",
  "VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR": "VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures",
  "VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR": "struct VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR",
  "VkCopyBufferInfo2KHR": "VkCopyBufferInfo2",
  "VkCopyImageInfo2KHR": "VkCopyImageInfo2",
  "VkCopyBufferToImageInfo2KHR": "VkCopyBufferToImageInfo2",
  "VkCopyImageToBufferInfo2KHR": "VkCopyImageToBufferInfo2",
  "VkBlitImageInfo2KHR": "VkBlitImageInfo2",
  "VkResolveImageInfo2KHR": "VkResolveImageInfo2",
  "VkBufferCopy2KHR": "VkBufferCopy2",
  "VkImageCopy2KHR": "VkImageCopy2",
  "VkImageBlit2KHR": "VkImageBlit2",
  "VkBufferImageCopy2KHR": "VkBufferImageCopy2",
  "VkImageResolve2KHR": "VkImageResolve2",
  "PFN_vkCmdCopyBuffer2KHR": "void (*)(VkCommandBuffer, const VkCopyBufferInfo2 *)",
  "PFN_vkCmdCopyImage2KHR": "void (*)(VkCommandBuffer, const VkCopyImageInfo2 *)",
  "PFN_vkCmdCopyBufferToImage2KHR": "void (*)(VkCommandBuffer, const VkCopyBufferToImageInfo2 *)",
  "PFN_vkCmdCopyImageToBuffer2KHR": "void (*)(VkCommandBuffer, const VkCopyImageToBufferInfo2 *)",
  "PFN_vkCmdBlitImage2KHR": "void (*)(VkCommandBuffer, const VkBlitImageInfo2 *)",
  "PFN_vkCmdResolveImage2KHR": "void (*)(VkCommandBuffer, const VkResolveImageInfo2 *)",
  "VkFormatFeatureFlags2KHR": "VkFormatFeatureFlags2",
  "VkFormatFeatureFlagBits2KHR": "VkFormatFeatureFlagBits2",
  "VkFormatProperties3KHR": "VkFormatProperties3",
  "VkPhysicalDeviceMaintenance4FeaturesKHR": "VkPhysicalDeviceMaintenance4Features",
  "VkPhysicalDeviceMaintenance4PropertiesKHR": "VkPhysicalDeviceMaintenance4Properties",
  "VkDeviceBufferMemoryRequirementsKHR": "VkDeviceBufferMemoryRequirements",
  "VkDeviceImageMemoryRequirementsKHR": "VkDeviceImageMemoryRequirements",
  "PFN_vkGetDeviceBufferMemoryRequirementsKHR": "void (*)(VkDevice, const VkDeviceBufferMemoryRequirements *, VkMemoryRequirements2 *)",
  "PFN_vkGetDeviceImageMemoryRequirementsKHR": "void (*)(VkDevice, const VkDeviceImageMemoryRequirements *, VkMemoryRequirements2 *)",
  "PFN_vkGetDeviceImageSparseMemoryRequirementsKHR": "void (*)(VkDevice, const VkDeviceImageMemoryRequirements *, uint32_t *, VkSparseImageMemoryRequirements2 *)",
  "VkDebugReportCallbackEXT": "struct VkDebugReportCallbackEXT_T *",
  "VkDebugReportObjectTypeEXT": "enum VkDebugReportObjectTypeEXT",
  "VkDebugReportFlagBitsEXT": "enum VkDebugReportFlagBitsEXT",
  "VkDebugReportFlagsEXT": "VkFlags",
  "PFN_vkDebugReportCallbackEXT": "VkBool32 (*)(VkDebugReportFlagsEXT, VkDebugReportObjectTypeEXT, uint64_t, size_t, int32_t, const char *, const char *, void *)",
  "VkDebugReportCallbackCreateInfoEXT": "struct VkDebugReportCallbackCreateInfoEXT",
  "PFN_vkCreateDebugReportCallbackEXT": "VkResult (*)(VkInstance, const VkDebugReportCallbackCreateInfoEXT *, const VkAllocationCallbacks *, VkDebugReportCallbackEXT *)",
  "PFN_vkDestroyDebugReportCallbackEXT": "void (*)(VkInstance, VkDebugReportCallbackEXT, const VkAllocationCallbacks *)",
  "PFN_vkDebugReportMessageEXT": "void (*)(VkInstance, VkDebugReportFlagsEXT, VkDebugReportObjectTypeEXT, uint64_t, size_t, int32_t, const char *, const char *)",
  "VkRasterizationOrderAMD": "enum VkRasterizationOrderAMD",
  "VkPipelineRasterizationStateRasterizationOrderAMD": "struct VkPipelineRasterizationStateRasterizationOrderAMD",
  "VkDebugMarkerObjectNameInfoEXT": "struct VkDebugMarkerObjectNameInfoEXT",
  "VkDebugMarkerObjectTagInfoEXT": "struct VkDebugMarkerObjectTagInfoEXT",
  "VkDebugMarkerMarkerInfoEXT": "struct VkDebugMarkerMarkerInfoEXT",
  "PFN_vkDebugMarkerSetObjectTagEXT": "VkResult (*)(VkDevice, const VkDebugMarkerObjectTagInfoEXT *)",
  "PFN_vkDebugMarkerSetObjectNameEXT": "VkResult (*)(VkDevice, const VkDebugMarkerObjectNameInfoEXT *)",
  "PFN_vkCmdDebugMarkerBeginEXT": "void (*)(VkCommandBuffer, const VkDebugMarkerMarkerInfoEXT *)",
  "PFN_vkCmdDebugMarkerEndEXT": "void (*)(VkCommandBuffer)",
  "PFN_vkCmdDebugMarkerInsertEXT": "void (*)(VkCommandBuffer, const VkDebugMarkerMarkerInfoEXT *)",
  "VkDedicatedAllocationImageCreateInfoNV": "struct VkDedicatedAllocationImageCreateInfoNV",
  "VkDedicatedAllocationBufferCreateInfoNV": "struct VkDedicatedAllocationBufferCreateInfoNV",
  "VkDedicatedAllocationMemoryAllocateInfoNV": "struct VkDedicatedAllocationMemoryAllocateInfoNV",
  "VkPipelineRasterizationStateStreamCreateFlagsEXT": "VkFlags",
  "VkPhysicalDeviceTransformFeedbackFeaturesEXT": "struct VkPhysicalDeviceTransformFeedbackFeaturesEXT",
  "VkPhysicalDeviceTransformFeedbackPropertiesEXT": "struct VkPhysicalDeviceTransformFeedbackPropertiesEXT",
  "VkPipelineRasterizationStateStreamCreateInfoEXT": "struct VkPipelineRasterizationStateStreamCreateInfoEXT",
  "PFN_vkCmdBindTransformFeedbackBuffersEXT": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer *, const VkDeviceSize *, const VkDeviceSize *)",
  "PFN_vkCmdBeginTransformFeedbackEXT": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer *, const VkDeviceSize *)",
  "PFN_vkCmdEndTransformFeedbackEXT": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer *, const VkDeviceSize *)",
  "PFN_vkCmdBeginQueryIndexedEXT": "void (*)(VkCommandBuffer, VkQueryPool, uint32_t, VkQueryControlFlags, uint32_t)",
  "PFN_vkCmdEndQueryIndexedEXT": "void (*)(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t)",
  "PFN_vkCmdDrawIndirectByteCountEXT": "void (*)(VkCommandBuffer, uint32_t, uint32_t, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "VkCuModuleNVX": "struct VkCuModuleNVX_T *",
  "VkCuFunctionNVX": "struct VkCuFunctionNVX_T *",
  "VkCuModuleCreateInfoNVX": "struct VkCuModuleCreateInfoNVX",
  "VkCuFunctionCreateInfoNVX": "struct VkCuFunctionCreateInfoNVX",
  "VkCuLaunchInfoNVX": "struct VkCuLaunchInfoNVX",
  "PFN_vkCreateCuModuleNVX": "VkResult (*)(VkDevice, const VkCuModuleCreateInfoNVX *, const VkAllocationCallbacks *, VkCuModuleNVX *)",
  "PFN_vkCreateCuFunctionNVX": "VkResult (*)(VkDevice, const VkCuFunctionCreateInfoNVX *, const VkAllocationCallbacks *, VkCuFunctionNVX *)",
  "PFN_vkDestroyCuModuleNVX": "void (*)(VkDevice, VkCuModuleNVX, const VkAllocationCallbacks *)",
  "PFN_vkDestroyCuFunctionNVX": "void (*)(VkDevice, VkCuFunctionNVX, const VkAllocationCallbacks *)",
  "PFN_vkCmdCuLaunchKernelNVX": "void (*)(VkCommandBuffer, const VkCuLaunchInfoNVX *)",
  "VkImageViewHandleInfoNVX": "struct VkImageViewHandleInfoNVX",
  "VkImageViewAddressPropertiesNVX": "struct VkImageViewAddressPropertiesNVX",
  "PFN_vkGetImageViewHandleNVX": "uint32_t (*)(VkDevice, const VkImageViewHandleInfoNVX *)",
  "PFN_vkGetImageViewAddressNVX": "VkResult (*)(VkDevice, VkImageView, VkImageViewAddressPropertiesNVX *)",
  "PFN_vkCmdDrawIndirectCountAMD": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "PFN_vkCmdDrawIndexedIndirectCountAMD": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "VkTextureLODGatherFormatPropertiesAMD": "struct VkTextureLODGatherFormatPropertiesAMD",
  "VkShaderInfoTypeAMD": "enum VkShaderInfoTypeAMD",
  "VkShaderResourceUsageAMD": "struct VkShaderResourceUsageAMD",
  "VkShaderStatisticsInfoAMD": "struct VkShaderStatisticsInfoAMD",
  "PFN_vkGetShaderInfoAMD": "VkResult (*)(VkDevice, VkPipeline, VkShaderStageFlagBits, VkShaderInfoTypeAMD, size_t *, void *)",
  "VkPhysicalDeviceCornerSampledImageFeaturesNV": "struct VkPhysicalDeviceCornerSampledImageFeaturesNV",
  "VkExternalMemoryHandleTypeFlagBitsNV": "enum VkExternalMemoryHandleTypeFlagBitsNV",
  "VkExternalMemoryHandleTypeFlagsNV": "VkFlags",
  "VkExternalMemoryFeatureFlagBitsNV": "enum VkExternalMemoryFeatureFlagBitsNV",
  "VkExternalMemoryFeatureFlagsNV": "VkFlags",
  "VkExternalImageFormatPropertiesNV": "struct VkExternalImageFormatPropertiesNV",
  "PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV": "VkResult (*)(VkPhysicalDevice, VkFormat, VkImageType, VkImageTiling, VkImageUsageFlags, VkImageCreateFlags, VkExternalMemoryHandleTypeFlagsNV, VkExternalImageFormatPropertiesNV *)",
  "VkExternalMemoryImageCreateInfoNV": "struct VkExternalMemoryImageCreateInfoNV",
  "VkExportMemoryAllocateInfoNV": "struct VkExportMemoryAllocateInfoNV",
  "VkValidationCheckEXT": "enum VkValidationCheckEXT",
  "VkValidationFlagsEXT": "struct VkValidationFlagsEXT",
  "VkPhysicalDeviceTextureCompressionASTCHDRFeaturesEXT": "VkPhysicalDeviceTextureCompressionASTCHDRFeatures",
  "VkImageViewASTCDecodeModeEXT": "struct VkImageViewASTCDecodeModeEXT",
  "VkPhysicalDeviceASTCDecodeFeaturesEXT": "struct VkPhysicalDeviceASTCDecodeFeaturesEXT",
  "VkConditionalRenderingFlagBitsEXT": "enum VkConditionalRenderingFlagBitsEXT",
  "VkConditionalRenderingFlagsEXT": "VkFlags",
  "VkConditionalRenderingBeginInfoEXT": "struct VkConditionalRenderingBeginInfoEXT",
  "VkPhysicalDeviceConditionalRenderingFeaturesEXT": "struct VkPhysicalDeviceConditionalRenderingFeaturesEXT",
  "VkCommandBufferInheritanceConditionalRenderingInfoEXT": "struct VkCommandBufferInheritanceConditionalRenderingInfoEXT",
  "PFN_vkCmdBeginConditionalRenderingEXT": "void (*)(VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT *)",
  "PFN_vkCmdEndConditionalRenderingEXT": "void (*)(VkCommandBuffer)",
  "VkViewportWScalingNV": "struct VkViewportWScalingNV",
  "VkPipelineViewportWScalingStateCreateInfoNV": "struct VkPipelineViewportWScalingStateCreateInfoNV",
  "PFN_vkCmdSetViewportWScalingNV": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkViewportWScalingNV *)",
  "PFN_vkReleaseDisplayEXT": "VkResult (*)(VkPhysicalDevice, VkDisplayKHR)",
  "VkSurfaceCounterFlagBitsEXT": "enum VkSurfaceCounterFlagBitsEXT",
  "VkSurfaceCounterFlagsEXT": "VkFlags",
  "VkSurfaceCapabilities2EXT": "struct VkSurfaceCapabilities2EXT",
  "PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT": "VkResult (*)(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilities2EXT *)",
  "VkDisplayPowerStateEXT": "enum VkDisplayPowerStateEXT",
  "VkDeviceEventTypeEXT": "enum VkDeviceEventTypeEXT",
  "VkDisplayEventTypeEXT": "enum VkDisplayEventTypeEXT",
  "VkDisplayPowerInfoEXT": "struct VkDisplayPowerInfoEXT",
  "VkDeviceEventInfoEXT": "struct VkDeviceEventInfoEXT",
  "VkDisplayEventInfoEXT": "struct VkDisplayEventInfoEXT",
  "VkSwapchainCounterCreateInfoEXT": "struct VkSwapchainCounterCreateInfoEXT",
  "PFN_vkDisplayPowerControlEXT": "VkResult (*)(VkDevice, VkDisplayKHR, const VkDisplayPowerInfoEXT *)",
  "PFN_vkRegisterDeviceEventEXT": "VkResult (*)(VkDevice, const VkDeviceEventInfoEXT *, const VkAllocationCallbacks *, VkFence *)",
  "PFN_vkRegisterDisplayEventEXT": "VkResult (*)(VkDevice, VkDisplayKHR, const VkDisplayEventInfoEXT *, const VkAllocationCallbacks *, VkFence *)",
  "PFN_vkGetSwapchainCounterEXT": "VkResult (*)(VkDevice, VkSwapchainKHR, VkSurfaceCounterFlagBitsEXT, uint64_t *)",
  "VkRefreshCycleDurationGOOGLE": "struct VkRefreshCycleDurationGOOGLE",
  "VkPastPresentationTimingGOOGLE": "struct VkPastPresentationTimingGOOGLE",
  "VkPresentTimeGOOGLE": "struct VkPresentTimeGOOGLE",
  "VkPresentTimesInfoGOOGLE": "struct VkPresentTimesInfoGOOGLE",
  "PFN_vkGetRefreshCycleDurationGOOGLE": "VkResult (*)(VkDevice, VkSwapchainKHR, VkRefreshCycleDurationGOOGLE *)",
  "PFN_vkGetPastPresentationTimingGOOGLE": "VkResult (*)(VkDevice, VkSwapchainKHR, uint32_t *, VkPastPresentationTimingGOOGLE *)",
  "VkPhysicalDeviceMultiviewPerViewAttributesPropertiesNVX": "struct VkPhysicalDeviceMultiviewPerViewAttributesPropertiesNVX",
  "VkViewportCoordinateSwizzleNV": "enum VkViewportCoordinateSwizzleNV",
  "VkPipelineViewportSwizzleStateCreateFlagsNV": "VkFlags",
  "VkViewportSwizzleNV": "struct VkViewportSwizzleNV",
  "VkPipelineViewportSwizzleStateCreateInfoNV": "struct VkPipelineViewportSwizzleStateCreateInfoNV",
  "VkDiscardRectangleModeEXT": "enum VkDiscardRectangleModeEXT",
  "VkPipelineDiscardRectangleStateCreateFlagsEXT": "VkFlags",
  "VkPhysicalDeviceDiscardRectanglePropertiesEXT": "struct VkPhysicalDeviceDiscardRectanglePropertiesEXT",
  "VkPipelineDiscardRectangleStateCreateInfoEXT": "struct VkPipelineDiscardRectangleStateCreateInfoEXT",
  "PFN_vkCmdSetDiscardRectangleEXT": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkRect2D *)",
  "VkConservativeRasterizationModeEXT": "enum VkConservativeRasterizationModeEXT",
  "VkPipelineRasterizationConservativeStateCreateFlagsEXT": "VkFlags",
  "VkPhysicalDeviceConservativeRasterizationPropertiesEXT": "struct VkPhysicalDeviceConservativeRasterizationPropertiesEXT",
  "VkPipelineRasterizationConservativeStateCreateInfoEXT": "struct VkPipelineRasterizationConservativeStateCreateInfoEXT",
  "VkPipelineRasterizationDepthClipStateCreateFlagsEXT": "VkFlags",
  "VkPhysicalDeviceDepthClipEnableFeaturesEXT": "struct VkPhysicalDeviceDepthClipEnableFeaturesEXT",
  "VkPipelineRasterizationDepthClipStateCreateInfoEXT": "struct VkPipelineRasterizationDepthClipStateCreateInfoEXT",
  "VkXYColorEXT": "struct VkXYColorEXT",
  "VkHdrMetadataEXT": "struct VkHdrMetadataEXT",
  "PFN_vkSetHdrMetadataEXT": "void (*)(VkDevice, uint32_t, const VkSwapchainKHR *, const VkHdrMetadataEXT *)",
  "VkDebugUtilsMessengerEXT": "struct VkDebugUtilsMessengerEXT_T *",
  "VkDebugUtilsMessengerCallbackDataFlagsEXT": "VkFlags",
  "VkDebugUtilsMessageSeverityFlagBitsEXT": "enum VkDebugUtilsMessageSeverityFlagBitsEXT",
  "VkDebugUtilsMessageTypeFlagBitsEXT": "enum VkDebugUtilsMessageTypeFlagBitsEXT",
  "VkDebugUtilsMessageTypeFlagsEXT": "VkFlags",
  "VkDebugUtilsMessageSeverityFlagsEXT": "VkFlags",
  "VkDebugUtilsMessengerCreateFlagsEXT": "VkFlags",
  "VkDebugUtilsLabelEXT": "struct VkDebugUtilsLabelEXT",
  "VkDebugUtilsObjectNameInfoEXT": "struct VkDebugUtilsObjectNameInfoEXT",
  "VkDebugUtilsMessengerCallbackDataEXT": "struct VkDebugUtilsMessengerCallbackDataEXT",
  "PFN_vkDebugUtilsMessengerCallbackEXT": "VkBool32 (*)(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT *, void *)",
  "VkDebugUtilsMessengerCreateInfoEXT": "struct VkDebugUtilsMessengerCreateInfoEXT",
  "VkDebugUtilsObjectTagInfoEXT": "struct VkDebugUtilsObjectTagInfoEXT",
  "PFN_vkSetDebugUtilsObjectNameEXT": "VkResult (*)(VkDevice, const VkDebugUtilsObjectNameInfoEXT *)",
  "PFN_vkSetDebugUtilsObjectTagEXT": "VkResult (*)(VkDevice, const VkDebugUtilsObjectTagInfoEXT *)",
  "PFN_vkQueueBeginDebugUtilsLabelEXT": "void (*)(VkQueue, const VkDebugUtilsLabelEXT *)",
  "PFN_vkQueueEndDebugUtilsLabelEXT": "void (*)(VkQueue)",
  "PFN_vkQueueInsertDebugUtilsLabelEXT": "void (*)(VkQueue, const VkDebugUtilsLabelEXT *)",
  "PFN_vkCmdBeginDebugUtilsLabelEXT": "void (*)(VkCommandBuffer, const VkDebugUtilsLabelEXT *)",
  "PFN_vkCmdEndDebugUtilsLabelEXT": "void (*)(VkCommandBuffer)",
  "PFN_vkCmdInsertDebugUtilsLabelEXT": "void (*)(VkCommandBuffer, const VkDebugUtilsLabelEXT *)",
  "PFN_vkCreateDebugUtilsMessengerEXT": "VkResult (*)(VkInstance, const VkDebugUtilsMessengerCreateInfoEXT *, const VkAllocationCallbacks *, VkDebugUtilsMessengerEXT *)",
  "PFN_vkDestroyDebugUtilsMessengerEXT": "void (*)(VkInstance, VkDebugUtilsMessengerEXT, const VkAllocationCallbacks *)",
  "PFN_vkSubmitDebugUtilsMessageEXT": "void (*)(VkInstance, VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT *)",
  "VkSamplerReductionModeEXT": "VkSamplerReductionMode",
  "VkSamplerReductionModeCreateInfoEXT": "VkSamplerReductionModeCreateInfo",
  "VkPhysicalDeviceSamplerFilterMinmaxPropertiesEXT": "VkPhysicalDeviceSamplerFilterMinmaxProperties",
  "VkPhysicalDeviceInlineUniformBlockFeaturesEXT": "VkPhysicalDeviceInlineUniformBlockFeatures",
  "VkPhysicalDeviceInlineUniformBlockPropertiesEXT": "VkPhysicalDeviceInlineUniformBlockProperties",
  "VkWriteDescriptorSetInlineUniformBlockEXT": "VkWriteDescriptorSetInlineUniformBlock",
  "VkDescriptorPoolInlineUniformBlockCreateInfoEXT": "VkDescriptorPoolInlineUniformBlockCreateInfo",
  "VkSampleLocationEXT": "struct VkSampleLocationEXT",
  "VkSampleLocationsInfoEXT": "struct VkSampleLocationsInfoEXT",
  "VkAttachmentSampleLocationsEXT": "struct VkAttachmentSampleLocationsEXT",
  "VkSubpassSampleLocationsEXT": "struct VkSubpassSampleLocationsEXT",
  "VkRenderPassSampleLocationsBeginInfoEXT": "struct VkRenderPassSampleLocationsBeginInfoEXT",
  "VkPipelineSampleLocationsStateCreateInfoEXT": "struct VkPipelineSampleLocationsStateCreateInfoEXT",
  "VkPhysicalDeviceSampleLocationsPropertiesEXT": "struct VkPhysicalDeviceSampleLocationsPropertiesEXT",
  "VkMultisamplePropertiesEXT": "struct VkMultisamplePropertiesEXT",
  "PFN_vkCmdSetSampleLocationsEXT": "void (*)(VkCommandBuffer, const VkSampleLocationsInfoEXT *)",
  "PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT": "void (*)(VkPhysicalDevice, VkSampleCountFlagBits, VkMultisamplePropertiesEXT *)",
  "VkBlendOverlapEXT": "enum VkBlendOverlapEXT",
  "VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT": "struct VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT",
  "VkPhysicalDeviceBlendOperationAdvancedPropertiesEXT": "struct VkPhysicalDeviceBlendOperationAdvancedPropertiesEXT",
  "VkPipelineColorBlendAdvancedStateCreateInfoEXT": "struct VkPipelineColorBlendAdvancedStateCreateInfoEXT",
  "VkPipelineCoverageToColorStateCreateFlagsNV": "VkFlags",
  "VkPipelineCoverageToColorStateCreateInfoNV": "struct VkPipelineCoverageToColorStateCreateInfoNV",
  "VkCoverageModulationModeNV": "enum VkCoverageModulationModeNV",
  "VkPipelineCoverageModulationStateCreateFlagsNV": "VkFlags",
  "VkPipelineCoverageModulationStateCreateInfoNV": "struct VkPipelineCoverageModulationStateCreateInfoNV",
  "VkPhysicalDeviceShaderSMBuiltinsPropertiesNV": "struct VkPhysicalDeviceShaderSMBuiltinsPropertiesNV",
  "VkPhysicalDeviceShaderSMBuiltinsFeaturesNV": "struct VkPhysicalDeviceShaderSMBuiltinsFeaturesNV",
  "VkDrmFormatModifierPropertiesEXT": "struct VkDrmFormatModifierPropertiesEXT",
  "VkDrmFormatModifierPropertiesListEXT": "struct VkDrmFormatModifierPropertiesListEXT",
  "VkPhysicalDeviceImageDrmFormatModifierInfoEXT": "struct VkPhysicalDeviceImageDrmFormatModifierInfoEXT",
  "VkImageDrmFormatModifierListCreateInfoEXT": "struct VkImageDrmFormatModifierListCreateInfoEXT",
  "VkImageDrmFormatModifierExplicitCreateInfoEXT": "struct VkImageDrmFormatModifierExplicitCreateInfoEXT",
  "VkImageDrmFormatModifierPropertiesEXT": "struct VkImageDrmFormatModifierPropertiesEXT",
  "VkDrmFormatModifierProperties2EXT": "struct VkDrmFormatModifierProperties2EXT",
  "VkDrmFormatModifierPropertiesList2EXT": "struct VkDrmFormatModifierPropertiesList2EXT",
  "PFN_vkGetImageDrmFormatModifierPropertiesEXT": "VkResult (*)(VkDevice, VkImage, VkImageDrmFormatModifierPropertiesEXT *)",
  "VkValidationCacheEXT": "struct VkValidationCacheEXT_T *",
  "VkValidationCacheHeaderVersionEXT": "enum VkValidationCacheHeaderVersionEXT",
  "VkValidationCacheCreateFlagsEXT": "VkFlags",
  "VkValidationCacheCreateInfoEXT": "struct VkValidationCacheCreateInfoEXT",
  "VkShaderModuleValidationCacheCreateInfoEXT": "struct VkShaderModuleValidationCacheCreateInfoEXT",
  "PFN_vkCreateValidationCacheEXT": "VkResult (*)(VkDevice, const VkValidationCacheCreateInfoEXT *, const VkAllocationCallbacks *, VkValidationCacheEXT *)",
  "PFN_vkDestroyValidationCacheEXT": "void (*)(VkDevice, VkValidationCacheEXT, const VkAllocationCallbacks *)",
  "PFN_vkMergeValidationCachesEXT": "VkResult (*)(VkDevice, VkValidationCacheEXT, uint32_t, const VkValidationCacheEXT *)",
  "PFN_vkGetValidationCacheDataEXT": "VkResult (*)(VkDevice, VkValidationCacheEXT, size_t *, void *)",
  "VkDescriptorBindingFlagBitsEXT": "VkDescriptorBindingFlagBits",
  "VkDescriptorBindingFlagsEXT": "VkDescriptorBindingFlags",
  "VkDescriptorSetLayoutBindingFlagsCreateInfoEXT": "VkDescriptorSetLayoutBindingFlagsCreateInfo",
  "VkPhysicalDeviceDescriptorIndexingFeaturesEXT": "VkPhysicalDeviceDescriptorIndexingFeatures",
  "VkPhysicalDeviceDescriptorIndexingPropertiesEXT": "VkPhysicalDeviceDescriptorIndexingProperties",
  "VkDescriptorSetVariableDescriptorCountAllocateInfoEXT": "VkDescriptorSetVariableDescriptorCountAllocateInfo",
  "VkDescriptorSetVariableDescriptorCountLayoutSupportEXT": "VkDescriptorSetVariableDescriptorCountLayoutSupport",
  "VkShadingRatePaletteEntryNV": "enum VkShadingRatePaletteEntryNV",
  "VkCoarseSampleOrderTypeNV": "enum VkCoarseSampleOrderTypeNV",
  "VkShadingRatePaletteNV": "struct VkShadingRatePaletteNV",
  "VkPipelineViewportShadingRateImageStateCreateInfoNV": "struct VkPipelineViewportShadingRateImageStateCreateInfoNV",
  "VkPhysicalDeviceShadingRateImageFeaturesNV": "struct VkPhysicalDeviceShadingRateImageFeaturesNV",
  "VkPhysicalDeviceShadingRateImagePropertiesNV": "struct VkPhysicalDeviceShadingRateImagePropertiesNV",
  "VkCoarseSampleLocationNV": "struct VkCoarseSampleLocationNV",
  "VkCoarseSampleOrderCustomNV": "struct VkCoarseSampleOrderCustomNV",
  "VkPipelineViewportCoarseSampleOrderStateCreateInfoNV": "struct VkPipelineViewportCoarseSampleOrderStateCreateInfoNV",
  "PFN_vkCmdBindShadingRateImageNV": "void (*)(VkCommandBuffer, VkImageView, VkImageLayout)",
  "PFN_vkCmdSetViewportShadingRatePaletteNV": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkShadingRatePaletteNV *)",
  "PFN_vkCmdSetCoarseSampleOrderNV": "void (*)(VkCommandBuffer, VkCoarseSampleOrderTypeNV, uint32_t, const VkCoarseSampleOrderCustomNV *)",
  "VkAccelerationStructureNV": "struct VkAccelerationStructureNV_T *",
  "VkRayTracingShaderGroupTypeKHR": "enum VkRayTracingShaderGroupTypeKHR",
  "VkRayTracingShaderGroupTypeNV": "VkRayTracingShaderGroupTypeKHR",
  "VkGeometryTypeKHR": "enum VkGeometryTypeKHR",
  "VkGeometryTypeNV": "VkGeometryTypeKHR",
  "VkAccelerationStructureTypeKHR": "enum VkAccelerationStructureTypeKHR",
  "VkAccelerationStructureTypeNV": "VkAccelerationStructureTypeKHR",
  "VkCopyAccelerationStructureModeKHR": "enum VkCopyAccelerationStructureModeKHR",
  "VkCopyAccelerationStructureModeNV": "VkCopyAccelerationStructureModeKHR",
  "VkAccelerationStructureMemoryRequirementsTypeNV": "enum VkAccelerationStructureMemoryRequirementsTypeNV",
  "VkGeometryFlagBitsKHR": "enum VkGeometryFlagBitsKHR",
  "VkGeometryFlagsKHR": "VkFlags",
  "VkGeometryFlagsNV": "VkGeometryFlagsKHR",
  "VkGeometryFlagBitsNV": "VkGeometryFlagBitsKHR",
  "VkGeometryInstanceFlagBitsKHR": "enum VkGeometryInstanceFlagBitsKHR",
  "VkGeometryInstanceFlagsKHR": "VkFlags",
  "VkGeometryInstanceFlagsNV": "VkGeometryInstanceFlagsKHR",
  "VkGeometryInstanceFlagBitsNV": "VkGeometryInstanceFlagBitsKHR",
  "VkBuildAccelerationStructureFlagBitsKHR": "enum VkBuildAccelerationStructureFlagBitsKHR",
  "VkBuildAccelerationStructureFlagsKHR": "VkFlags",
  "VkBuildAccelerationStructureFlagsNV": "VkBuildAccelerationStructureFlagsKHR",
  "VkBuildAccelerationStructureFlagBitsNV": "VkBuildAccelerationStructureFlagBitsKHR",
  "VkRayTracingShaderGroupCreateInfoNV": "struct VkRayTracingShaderGroupCreateInfoNV",
  "VkRayTracingPipelineCreateInfoNV": "struct VkRayTracingPipelineCreateInfoNV",
  "VkGeometryTrianglesNV": "struct VkGeometryTrianglesNV",
  "VkGeometryAABBNV": "struct VkGeometryAABBNV",
  "VkGeometryDataNV": "struct VkGeometryDataNV",
  "VkGeometryNV": "struct VkGeometryNV",
  "VkAccelerationStructureInfoNV": "struct VkAccelerationStructureInfoNV",
  "VkAccelerationStructureCreateInfoNV": "struct VkAccelerationStructureCreateInfoNV",
  "VkBindAccelerationStructureMemoryInfoNV": "struct VkBindAccelerationStructureMemoryInfoNV",
  "VkWriteDescriptorSetAccelerationStructureNV": "struct VkWriteDescriptorSetAccelerationStructureNV",
  "VkAccelerationStructureMemoryRequirementsInfoNV": "struct VkAccelerationStructureMemoryRequirementsInfoNV",
  "VkPhysicalDeviceRayTracingPropertiesNV": "struct VkPhysicalDeviceRayTracingPropertiesNV",
  "VkTransformMatrixKHR": "struct VkTransformMatrixKHR",
  "VkTransformMatrixNV": "VkTransformMatrixKHR",
  "VkAabbPositionsKHR": "struct VkAabbPositionsKHR",
  "VkAabbPositionsNV": "VkAabbPositionsKHR",
  "VkAccelerationStructureInstanceKHR": "struct VkAccelerationStructureInstanceKHR",
  "VkAccelerationStructureInstanceNV": "VkAccelerationStructureInstanceKHR",
  "PFN_vkCreateAccelerationStructureNV": "VkResult (*)(VkDevice, const VkAccelerationStructureCreateInfoNV *, const VkAllocationCallbacks *, VkAccelerationStructureNV *)",
  "PFN_vkDestroyAccelerationStructureNV": "void (*)(VkDevice, VkAccelerationStructureNV, const VkAllocationCallbacks *)",
  "PFN_vkGetAccelerationStructureMemoryRequirementsNV": "void (*)(VkDevice, const VkAccelerationStructureMemoryRequirementsInfoNV *, VkMemoryRequirements2KHR *)",
  "PFN_vkBindAccelerationStructureMemoryNV": "VkResult (*)(VkDevice, uint32_t, const VkBindAccelerationStructureMemoryInfoNV *)",
  "PFN_vkCmdBuildAccelerationStructureNV": "void (*)(VkCommandBuffer, const VkAccelerationStructureInfoNV *, VkBuffer, VkDeviceSize, VkBool32, VkAccelerationStructureNV, VkAccelerationStructureNV, VkBuffer, VkDeviceSize)",
  "PFN_vkCmdCopyAccelerationStructureNV": "void (*)(VkCommandBuffer, VkAccelerationStructureNV, VkAccelerationStructureNV, VkCopyAccelerationStructureModeKHR)",
  "PFN_vkCmdTraceRaysNV": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, VkDeviceSize, VkBuffer, VkDeviceSize, VkDeviceSize, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t, uint32_t, uint32_t)",
  "PFN_vkCreateRayTracingPipelinesNV": "VkResult (*)(VkDevice, VkPipelineCache, uint32_t, const VkRayTracingPipelineCreateInfoNV *, const VkAllocationCallbacks *, VkPipeline *)",
  "PFN_vkGetRayTracingShaderGroupHandlesKHR": "VkResult (*)(VkDevice, VkPipeline, uint32_t, uint32_t, size_t, void *)",
  "PFN_vkGetRayTracingShaderGroupHandlesNV": "VkResult (*)(VkDevice, VkPipeline, uint32_t, uint32_t, size_t, void *)",
  "PFN_vkGetAccelerationStructureHandleNV": "VkResult (*)(VkDevice, VkAccelerationStructureNV, size_t, void *)",
  "PFN_vkCmdWriteAccelerationStructuresPropertiesNV": "void (*)(VkCommandBuffer, uint32_t, const VkAccelerationStructureNV *, VkQueryType, VkQueryPool, uint32_t)",
  "PFN_vkCompileDeferredNV": "VkResult (*)(VkDevice, VkPipeline, uint32_t)",
  "VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV": "struct VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV",
  "VkPipelineRepresentativeFragmentTestStateCreateInfoNV": "struct VkPipelineRepresentativeFragmentTestStateCreateInfoNV",
  "VkPhysicalDeviceImageViewImageFormatInfoEXT": "struct VkPhysicalDeviceImageViewImageFormatInfoEXT",
  "VkFilterCubicImageViewImageFormatPropertiesEXT": "struct VkFilterCubicImageViewImageFormatPropertiesEXT",
  "VkQueueGlobalPriorityEXT": "VkQueueGlobalPriorityKHR",
  "VkDeviceQueueGlobalPriorityCreateInfoEXT": "VkDeviceQueueGlobalPriorityCreateInfoKHR",
  "VkImportMemoryHostPointerInfoEXT": "struct VkImportMemoryHostPointerInfoEXT",
  "VkMemoryHostPointerPropertiesEXT": "struct VkMemoryHostPointerPropertiesEXT",
  "VkPhysicalDeviceExternalMemoryHostPropertiesEXT": "struct VkPhysicalDeviceExternalMemoryHostPropertiesEXT",
  "PFN_vkGetMemoryHostPointerPropertiesEXT": "VkResult (*)(VkDevice, VkExternalMemoryHandleTypeFlagBits, const void *, VkMemoryHostPointerPropertiesEXT *)",
  "PFN_vkCmdWriteBufferMarkerAMD": "void (*)(VkCommandBuffer, VkPipelineStageFlagBits, VkBuffer, VkDeviceSize, uint32_t)",
  "VkPipelineCompilerControlFlagBitsAMD": "enum VkPipelineCompilerControlFlagBitsAMD",
  "VkPipelineCompilerControlFlagsAMD": "VkFlags",
  "VkPipelineCompilerControlCreateInfoAMD": "struct VkPipelineCompilerControlCreateInfoAMD",
  "VkTimeDomainEXT": "enum VkTimeDomainEXT",
  "VkCalibratedTimestampInfoEXT": "struct VkCalibratedTimestampInfoEXT",
  "PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkTimeDomainEXT *)",
  "PFN_vkGetCalibratedTimestampsEXT": "VkResult (*)(VkDevice, uint32_t, const VkCalibratedTimestampInfoEXT *, uint64_t *, uint64_t *)",
  "VkPhysicalDeviceShaderCorePropertiesAMD": "struct VkPhysicalDeviceShaderCorePropertiesAMD",
  "VkMemoryOverallocationBehaviorAMD": "enum VkMemoryOverallocationBehaviorAMD",
  "VkDeviceMemoryOverallocationCreateInfoAMD": "struct VkDeviceMemoryOverallocationCreateInfoAMD",
  "VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT": "struct VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT",
  "VkVertexInputBindingDivisorDescriptionEXT": "struct VkVertexInputBindingDivisorDescriptionEXT",
  "VkPipelineVertexInputDivisorStateCreateInfoEXT": "struct VkPipelineVertexInputDivisorStateCreateInfoEXT",
  "VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT": "struct VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT",
  "VkPipelineCreationFeedbackFlagBitsEXT": "VkPipelineCreationFeedbackFlagBits",
  "VkPipelineCreationFeedbackFlagsEXT": "VkPipelineCreationFeedbackFlags",
  "VkPipelineCreationFeedbackCreateInfoEXT": "VkPipelineCreationFeedbackCreateInfo",
  "VkPipelineCreationFeedbackEXT": "VkPipelineCreationFeedback",
  "VkPhysicalDeviceComputeShaderDerivativesFeaturesNV": "struct VkPhysicalDeviceComputeShaderDerivativesFeaturesNV",
  "VkPhysicalDeviceMeshShaderFeaturesNV": "struct VkPhysicalDeviceMeshShaderFeaturesNV",
  "VkPhysicalDeviceMeshShaderPropertiesNV": "struct VkPhysicalDeviceMeshShaderPropertiesNV",
  "VkDrawMeshTasksIndirectCommandNV": "struct VkDrawMeshTasksIndirectCommandNV",
  "PFN_vkCmdDrawMeshTasksNV": "void (*)(VkCommandBuffer, uint32_t, uint32_t)",
  "PFN_vkCmdDrawMeshTasksIndirectNV": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "PFN_vkCmdDrawMeshTasksIndirectCountNV": "void (*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t)",
  "VkPhysicalDeviceFragmentShaderBarycentricFeaturesNV": "struct VkPhysicalDeviceFragmentShaderBarycentricFeaturesNV",
  "VkPhysicalDeviceShaderImageFootprintFeaturesNV": "struct VkPhysicalDeviceShaderImageFootprintFeaturesNV",
  "VkPipelineViewportExclusiveScissorStateCreateInfoNV": "struct VkPipelineViewportExclusiveScissorStateCreateInfoNV",
  "VkPhysicalDeviceExclusiveScissorFeaturesNV": "struct VkPhysicalDeviceExclusiveScissorFeaturesNV",
  "PFN_vkCmdSetExclusiveScissorNV": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkRect2D *)",
  "VkQueueFamilyCheckpointPropertiesNV": "struct VkQueueFamilyCheckpointPropertiesNV",
  "VkCheckpointDataNV": "struct VkCheckpointDataNV",
  "PFN_vkCmdSetCheckpointNV": "void (*)(VkCommandBuffer, const void *)",
  "PFN_vkGetQueueCheckpointDataNV": "void (*)(VkQueue, uint32_t *, VkCheckpointDataNV *)",
  "VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL": "struct VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL",
  "VkPerformanceConfigurationINTEL": "struct VkPerformanceConfigurationINTEL_T *",
  "VkPerformanceConfigurationTypeINTEL": "enum VkPerformanceConfigurationTypeINTEL",
  "VkQueryPoolSamplingModeINTEL": "enum VkQueryPoolSamplingModeINTEL",
  "VkPerformanceOverrideTypeINTEL": "enum VkPerformanceOverrideTypeINTEL",
  "VkPerformanceParameterTypeINTEL": "enum VkPerformanceParameterTypeINTEL",
  "VkPerformanceValueTypeINTEL": "enum VkPerformanceValueTypeINTEL",
  "VkPerformanceValueDataINTEL": "union VkPerformanceValueDataINTEL",
  "VkPerformanceValueINTEL": "struct VkPerformanceValueINTEL",
  "VkInitializePerformanceApiInfoINTEL": "struct VkInitializePerformanceApiInfoINTEL",
  "VkQueryPoolPerformanceQueryCreateInfoINTEL": "struct VkQueryPoolPerformanceQueryCreateInfoINTEL",
  "VkQueryPoolCreateInfoINTEL": "VkQueryPoolPerformanceQueryCreateInfoINTEL",
  "VkPerformanceMarkerInfoINTEL": "struct VkPerformanceMarkerInfoINTEL",
  "VkPerformanceStreamMarkerInfoINTEL": "struct VkPerformanceStreamMarkerInfoINTEL",
  "VkPerformanceOverrideInfoINTEL": "struct VkPerformanceOverrideInfoINTEL",
  "VkPerformanceConfigurationAcquireInfoINTEL": "struct VkPerformanceConfigurationAcquireInfoINTEL",
  "PFN_vkInitializePerformanceApiINTEL": "VkResult (*)(VkDevice, const VkInitializePerformanceApiInfoINTEL *)",
  "PFN_vkUninitializePerformanceApiINTEL": "void (*)(VkDevice)",
  "PFN_vkCmdSetPerformanceMarkerINTEL": "VkResult (*)(VkCommandBuffer, const VkPerformanceMarkerInfoINTEL *)",
  "PFN_vkCmdSetPerformanceStreamMarkerINTEL": "VkResult (*)(VkCommandBuffer, const VkPerformanceStreamMarkerInfoINTEL *)",
  "PFN_vkCmdSetPerformanceOverrideINTEL": "VkResult (*)(VkCommandBuffer, const VkPerformanceOverrideInfoINTEL *)",
  "PFN_vkAcquirePerformanceConfigurationINTEL": "VkResult (*)(VkDevice, const VkPerformanceConfigurationAcquireInfoINTEL *, VkPerformanceConfigurationINTEL *)",
  "PFN_vkReleasePerformanceConfigurationINTEL": "VkResult (*)(VkDevice, VkPerformanceConfigurationINTEL)",
  "PFN_vkQueueSetPerformanceConfigurationINTEL": "VkResult (*)(VkQueue, VkPerformanceConfigurationINTEL)",
  "PFN_vkGetPerformanceParameterINTEL": "VkResult (*)(VkDevice, VkPerformanceParameterTypeINTEL, VkPerformanceValueINTEL *)",
  "VkPhysicalDevicePCIBusInfoPropertiesEXT": "struct VkPhysicalDevicePCIBusInfoPropertiesEXT",
  "VkDisplayNativeHdrSurfaceCapabilitiesAMD": "struct VkDisplayNativeHdrSurfaceCapabilitiesAMD",
  "VkSwapchainDisplayNativeHdrCreateInfoAMD": "struct VkSwapchainDisplayNativeHdrCreateInfoAMD",
  "PFN_vkSetLocalDimmingAMD": "void (*)(VkDevice, VkSwapchainKHR, VkBool32)",
  "VkPhysicalDeviceFragmentDensityMapFeaturesEXT": "struct VkPhysicalDeviceFragmentDensityMapFeaturesEXT",
  "VkPhysicalDeviceFragmentDensityMapPropertiesEXT": "struct VkPhysicalDeviceFragmentDensityMapPropertiesEXT",
  "VkRenderPassFragmentDensityMapCreateInfoEXT": "struct VkRenderPassFragmentDensityMapCreateInfoEXT",
  "VkPhysicalDeviceScalarBlockLayoutFeaturesEXT": "VkPhysicalDeviceScalarBlockLayoutFeatures",
  "VkPhysicalDeviceSubgroupSizeControlFeaturesEXT": "VkPhysicalDeviceSubgroupSizeControlFeatures",
  "VkPhysicalDeviceSubgroupSizeControlPropertiesEXT": "VkPhysicalDeviceSubgroupSizeControlProperties",
  "VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT": "VkPipelineShaderStageRequiredSubgroupSizeCreateInfo",
  "VkShaderCorePropertiesFlagBitsAMD": "enum VkShaderCorePropertiesFlagBitsAMD",
  "VkShaderCorePropertiesFlagsAMD": "VkFlags",
  "VkPhysicalDeviceShaderCoreProperties2AMD": "struct VkPhysicalDeviceShaderCoreProperties2AMD",
  "VkPhysicalDeviceCoherentMemoryFeaturesAMD": "struct VkPhysicalDeviceCoherentMemoryFeaturesAMD",
  "VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT": "struct VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT",
  "VkPhysicalDeviceMemoryBudgetPropertiesEXT": "struct VkPhysicalDeviceMemoryBudgetPropertiesEXT",
  "VkPhysicalDeviceMemoryPriorityFeaturesEXT": "struct VkPhysicalDeviceMemoryPriorityFeaturesEXT",
  "VkMemoryPriorityAllocateInfoEXT": "struct VkMemoryPriorityAllocateInfoEXT",
  "VkPhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV": "struct VkPhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV",
  "VkPhysicalDeviceBufferDeviceAddressFeaturesEXT": "struct VkPhysicalDeviceBufferDeviceAddressFeaturesEXT",
  "VkPhysicalDeviceBufferAddressFeaturesEXT": "VkPhysicalDeviceBufferDeviceAddressFeaturesEXT",
  "VkBufferDeviceAddressInfoEXT": "VkBufferDeviceAddressInfo",
  "VkBufferDeviceAddressCreateInfoEXT": "struct VkBufferDeviceAddressCreateInfoEXT",
  "PFN_vkGetBufferDeviceAddressEXT": "VkDeviceAddress (*)(VkDevice, const VkBufferDeviceAddressInfo *)",
  "VkToolPurposeFlagBitsEXT": "VkToolPurposeFlagBits",
  "VkToolPurposeFlagsEXT": "VkToolPurposeFlags",
  "VkPhysicalDeviceToolPropertiesEXT": "VkPhysicalDeviceToolProperties",
  "PFN_vkGetPhysicalDeviceToolPropertiesEXT": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkPhysicalDeviceToolProperties *)",
  "VkImageStencilUsageCreateInfoEXT": "VkImageStencilUsageCreateInfo",
  "VkValidationFeatureEnableEXT": "enum VkValidationFeatureEnableEXT",
  "VkValidationFeatureDisableEXT": "enum VkValidationFeatureDisableEXT",
  "VkValidationFeaturesEXT": "struct VkValidationFeaturesEXT",
  "VkComponentTypeNV": "enum VkComponentTypeNV",
  "VkScopeNV": "enum VkScopeNV",
  "VkCooperativeMatrixPropertiesNV": "struct VkCooperativeMatrixPropertiesNV",
  "VkPhysicalDeviceCooperativeMatrixFeaturesNV": "struct VkPhysicalDeviceCooperativeMatrixFeaturesNV",
  "VkPhysicalDeviceCooperativeMatrixPropertiesNV": "struct VkPhysicalDeviceCooperativeMatrixPropertiesNV",
  "PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkCooperativeMatrixPropertiesNV *)",
  "VkCoverageReductionModeNV": "enum VkCoverageReductionModeNV",
  "VkPipelineCoverageReductionStateCreateFlagsNV": "VkFlags",
  "VkPhysicalDeviceCoverageReductionModeFeaturesNV": "struct VkPhysicalDeviceCoverageReductionModeFeaturesNV",
  "VkPipelineCoverageReductionStateCreateInfoNV": "struct VkPipelineCoverageReductionStateCreateInfoNV",
  "VkFramebufferMixedSamplesCombinationNV": "struct VkFramebufferMixedSamplesCombinationNV",
  "PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV": "VkResult (*)(VkPhysicalDevice, uint32_t *, VkFramebufferMixedSamplesCombinationNV *)",
  "VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT": "struct VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT",
  "VkPhysicalDeviceYcbcrImageArraysFeaturesEXT": "struct VkPhysicalDeviceYcbcrImageArraysFeaturesEXT",
  "VkProvokingVertexModeEXT": "enum VkProvokingVertexModeEXT",
  "VkPhysicalDeviceProvokingVertexFeaturesEXT": "struct VkPhysicalDeviceProvokingVertexFeaturesEXT",
  "VkPhysicalDeviceProvokingVertexPropertiesEXT": "struct VkPhysicalDeviceProvokingVertexPropertiesEXT",
  "VkPipelineRasterizationProvokingVertexStateCreateInfoEXT": "struct VkPipelineRasterizationProvokingVertexStateCreateInfoEXT",
  "VkHeadlessSurfaceCreateFlagsEXT": "VkFlags",
  "VkHeadlessSurfaceCreateInfoEXT": "struct VkHeadlessSurfaceCreateInfoEXT",
  "PFN_vkCreateHeadlessSurfaceEXT": "VkResult (*)(VkInstance, const VkHeadlessSurfaceCreateInfoEXT *, const VkAllocationCallbacks *, VkSurfaceKHR *)",
  "VkLineRasterizationModeEXT": "enum VkLineRasterizationModeEXT",
  "VkPhysicalDeviceLineRasterizationFeaturesEXT": "struct VkPhysicalDeviceLineRasterizationFeaturesEXT",
  "VkPhysicalDeviceLineRasterizationPropertiesEXT": "struct VkPhysicalDeviceLineRasterizationPropertiesEXT",
  "VkPipelineRasterizationLineStateCreateInfoEXT": "struct VkPipelineRasterizationLineStateCreateInfoEXT",
  "PFN_vkCmdSetLineStippleEXT": "void (*)(VkCommandBuffer, uint32_t, uint16_t)",
  "VkPhysicalDeviceShaderAtomicFloatFeaturesEXT": "struct VkPhysicalDeviceShaderAtomicFloatFeaturesEXT",
  "VkPhysicalDeviceHostQueryResetFeaturesEXT": "VkPhysicalDeviceHostQueryResetFeatures",
  "PFN_vkResetQueryPoolEXT": "void (*)(VkDevice, VkQueryPool, uint32_t, uint32_t)",
  "VkPhysicalDeviceIndexTypeUint8FeaturesEXT": "struct VkPhysicalDeviceIndexTypeUint8FeaturesEXT",
  "VkPhysicalDeviceExtendedDynamicStateFeaturesEXT": "struct VkPhysicalDeviceExtendedDynamicStateFeaturesEXT",
  "PFN_vkCmdSetCullModeEXT": "void (*)(VkCommandBuffer, VkCullModeFlags)",
  "PFN_vkCmdSetFrontFaceEXT": "void (*)(VkCommandBuffer, VkFrontFace)",
  "PFN_vkCmdSetPrimitiveTopologyEXT": "void (*)(VkCommandBuffer, VkPrimitiveTopology)",
  "PFN_vkCmdSetViewportWithCountEXT": "void (*)(VkCommandBuffer, uint32_t, const VkViewport *)",
  "PFN_vkCmdSetScissorWithCountEXT": "void (*)(VkCommandBuffer, uint32_t, const VkRect2D *)",
  "PFN_vkCmdBindVertexBuffers2EXT": "void (*)(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer *, const VkDeviceSize *, const VkDeviceSize *, const VkDeviceSize *)",
  "PFN_vkCmdSetDepthTestEnableEXT": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetDepthWriteEnableEXT": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetDepthCompareOpEXT": "void (*)(VkCommandBuffer, VkCompareOp)",
  "PFN_vkCmdSetDepthBoundsTestEnableEXT": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetStencilTestEnableEXT": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetStencilOpEXT": "void (*)(VkCommandBuffer, VkStencilFaceFlags, VkStencilOp, VkStencilOp, VkStencilOp, VkCompareOp)",
  "VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT": "struct VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT",
  "VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT": "VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures",
  "VkIndirectCommandsLayoutNV": "struct VkIndirectCommandsLayoutNV_T *",
  "VkIndirectCommandsTokenTypeNV": "enum VkIndirectCommandsTokenTypeNV",
  "VkIndirectStateFlagBitsNV": "enum VkIndirectStateFlagBitsNV",
  "VkIndirectStateFlagsNV": "VkFlags",
  "VkIndirectCommandsLayoutUsageFlagBitsNV": "enum VkIndirectCommandsLayoutUsageFlagBitsNV",
  "VkIndirectCommandsLayoutUsageFlagsNV": "VkFlags",
  "VkPhysicalDeviceDeviceGeneratedCommandsPropertiesNV": "struct VkPhysicalDeviceDeviceGeneratedCommandsPropertiesNV",
  "VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV": "struct VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV",
  "VkGraphicsShaderGroupCreateInfoNV": "struct VkGraphicsShaderGroupCreateInfoNV",
  "VkGraphicsPipelineShaderGroupsCreateInfoNV": "struct VkGraphicsPipelineShaderGroupsCreateInfoNV",
  "VkBindShaderGroupIndirectCommandNV": "struct VkBindShaderGroupIndirectCommandNV",
  "VkBindIndexBufferIndirectCommandNV": "struct VkBindIndexBufferIndirectCommandNV",
  "VkBindVertexBufferIndirectCommandNV": "struct VkBindVertexBufferIndirectCommandNV",
  "VkSetStateFlagsIndirectCommandNV": "struct VkSetStateFlagsIndirectCommandNV",
  "VkIndirectCommandsStreamNV": "struct VkIndirectCommandsStreamNV",
  "VkIndirectCommandsLayoutTokenNV": "struct VkIndirectCommandsLayoutTokenNV",
  "VkIndirectCommandsLayoutCreateInfoNV": "struct VkIndirectCommandsLayoutCreateInfoNV",
  "VkGeneratedCommandsInfoNV": "struct VkGeneratedCommandsInfoNV",
  "VkGeneratedCommandsMemoryRequirementsInfoNV": "struct VkGeneratedCommandsMemoryRequirementsInfoNV",
  "PFN_vkGetGeneratedCommandsMemoryRequirementsNV": "void (*)(VkDevice, const VkGeneratedCommandsMemoryRequirementsInfoNV *, VkMemoryRequirements2 *)",
  "PFN_vkCmdPreprocessGeneratedCommandsNV": "void (*)(VkCommandBuffer, const VkGeneratedCommandsInfoNV *)",
  "PFN_vkCmdExecuteGeneratedCommandsNV": "void (*)(VkCommandBuffer, VkBool32, const VkGeneratedCommandsInfoNV *)",
  "PFN_vkCmdBindPipelineShaderGroupNV": "void (*)(VkCommandBuffer, VkPipelineBindPoint, VkPipeline, uint32_t)",
  "PFN_vkCreateIndirectCommandsLayoutNV": "VkResult (*)(VkDevice, const VkIndirectCommandsLayoutCreateInfoNV *, const VkAllocationCallbacks *, VkIndirectCommandsLayoutNV *)",
  "PFN_vkDestroyIndirectCommandsLayoutNV": "void (*)(VkDevice, VkIndirectCommandsLayoutNV, const VkAllocationCallbacks *)",
  "VkPhysicalDeviceInheritedViewportScissorFeaturesNV": "struct VkPhysicalDeviceInheritedViewportScissorFeaturesNV",
  "VkCommandBufferInheritanceViewportScissorInfoNV": "struct VkCommandBufferInheritanceViewportScissorInfoNV",
  "VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT": "struct VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT",
  "VkPhysicalDeviceTexelBufferAlignmentPropertiesEXT": "VkPhysicalDeviceTexelBufferAlignmentProperties",
  "VkRenderPassTransformBeginInfoQCOM": "struct VkRenderPassTransformBeginInfoQCOM",
  "VkCommandBufferInheritanceRenderPassTransformInfoQCOM": "struct VkCommandBufferInheritanceRenderPassTransformInfoQCOM",
  "VkDeviceMemoryReportEventTypeEXT": "enum VkDeviceMemoryReportEventTypeEXT",
  "VkDeviceMemoryReportFlagsEXT": "VkFlags",
  "VkPhysicalDeviceDeviceMemoryReportFeaturesEXT": "struct VkPhysicalDeviceDeviceMemoryReportFeaturesEXT",
  "VkDeviceMemoryReportCallbackDataEXT": "struct VkDeviceMemoryReportCallbackDataEXT",
  "PFN_vkDeviceMemoryReportCallbackEXT": "void (*)(const VkDeviceMemoryReportCallbackDataEXT *, void *)",
  "VkDeviceDeviceMemoryReportCreateInfoEXT": "struct VkDeviceDeviceMemoryReportCreateInfoEXT",
  "PFN_vkAcquireDrmDisplayEXT": "VkResult (*)(VkPhysicalDevice, int32_t, VkDisplayKHR)",
  "PFN_vkGetDrmDisplayEXT": "VkResult (*)(VkPhysicalDevice, int32_t, uint32_t, VkDisplayKHR *)",
  "VkPhysicalDeviceRobustness2FeaturesEXT": "struct VkPhysicalDeviceRobustness2FeaturesEXT",
  "VkPhysicalDeviceRobustness2PropertiesEXT": "struct VkPhysicalDeviceRobustness2PropertiesEXT",
  "VkSamplerCustomBorderColorCreateInfoEXT": "struct VkSamplerCustomBorderColorCreateInfoEXT",
  "VkPhysicalDeviceCustomBorderColorPropertiesEXT": "struct VkPhysicalDeviceCustomBorderColorPropertiesEXT",
  "VkPhysicalDeviceCustomBorderColorFeaturesEXT": "struct VkPhysicalDeviceCustomBorderColorFeaturesEXT",
  "VkPrivateDataSlotEXT": "VkPrivateDataSlot",
  "VkPrivateDataSlotCreateFlagsEXT": "VkPrivateDataSlotCreateFlags",
  "VkPhysicalDevicePrivateDataFeaturesEXT": "VkPhysicalDevicePrivateDataFeatures",
  "VkDevicePrivateDataCreateInfoEXT": "VkDevicePrivateDataCreateInfo",
  "VkPrivateDataSlotCreateInfoEXT": "VkPrivateDataSlotCreateInfo",
  "PFN_vkCreatePrivateDataSlotEXT": "VkResult (*)(VkDevice, const VkPrivateDataSlotCreateInfo *, const VkAllocationCallbacks *, VkPrivateDataSlot *)",
  "PFN_vkDestroyPrivateDataSlotEXT": "void (*)(VkDevice, VkPrivateDataSlot, const VkAllocationCallbacks *)",
  "PFN_vkSetPrivateDataEXT": "VkResult (*)(VkDevice, VkObjectType, uint64_t, VkPrivateDataSlot, uint64_t)",
  "PFN_vkGetPrivateDataEXT": "void (*)(VkDevice, VkObjectType, uint64_t, VkPrivateDataSlot, uint64_t *)",
  "VkPhysicalDevicePipelineCreationCacheControlFeaturesEXT": "VkPhysicalDevicePipelineCreationCacheControlFeatures",
  "VkDeviceDiagnosticsConfigFlagBitsNV": "enum VkDeviceDiagnosticsConfigFlagBitsNV",
  "VkDeviceDiagnosticsConfigFlagsNV": "VkFlags",
  "VkPhysicalDeviceDiagnosticsConfigFeaturesNV": "struct VkPhysicalDeviceDiagnosticsConfigFeaturesNV",
  "VkDeviceDiagnosticsConfigCreateInfoNV": "struct VkDeviceDiagnosticsConfigCreateInfoNV",
  "VkGraphicsPipelineLibraryFlagBitsEXT": "enum VkGraphicsPipelineLibraryFlagBitsEXT",
  "VkGraphicsPipelineLibraryFlagsEXT": "VkFlags",
  "VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT": "struct VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT",
  "VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT": "struct VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT",
  "VkGraphicsPipelineLibraryCreateInfoEXT": "struct VkGraphicsPipelineLibraryCreateInfoEXT",
  "VkFragmentShadingRateTypeNV": "enum VkFragmentShadingRateTypeNV",
  "VkFragmentShadingRateNV": "enum VkFragmentShadingRateNV",
  "VkPhysicalDeviceFragmentShadingRateEnumsFeaturesNV": "struct VkPhysicalDeviceFragmentShadingRateEnumsFeaturesNV",
  "VkPhysicalDeviceFragmentShadingRateEnumsPropertiesNV": "struct VkPhysicalDeviceFragmentShadingRateEnumsPropertiesNV",
  "VkPipelineFragmentShadingRateEnumStateCreateInfoNV": "struct VkPipelineFragmentShadingRateEnumStateCreateInfoNV",
  "PFN_vkCmdSetFragmentShadingRateEnumNV": "void (*)(VkCommandBuffer, VkFragmentShadingRateNV, const VkFragmentShadingRateCombinerOpKHR *)",
  "VkAccelerationStructureMotionInstanceTypeNV": "enum VkAccelerationStructureMotionInstanceTypeNV",
  "VkAccelerationStructureMotionInfoFlagsNV": "VkFlags",
  "VkAccelerationStructureMotionInstanceFlagsNV": "VkFlags",
  "VkDeviceOrHostAddressConstKHR": "union VkDeviceOrHostAddressConstKHR",
  "VkAccelerationStructureGeometryMotionTrianglesDataNV": "struct VkAccelerationStructureGeometryMotionTrianglesDataNV",
  "VkAccelerationStructureMotionInfoNV": "struct VkAccelerationStructureMotionInfoNV",
  "VkAccelerationStructureMatrixMotionInstanceNV": "struct VkAccelerationStructureMatrixMotionInstanceNV",
  "VkSRTDataNV": "struct VkSRTDataNV",
  "VkAccelerationStructureSRTMotionInstanceNV": "struct VkAccelerationStructureSRTMotionInstanceNV",
  "VkAccelerationStructureMotionInstanceDataNV": "union VkAccelerationStructureMotionInstanceDataNV",
  "VkAccelerationStructureMotionInstanceNV": "struct VkAccelerationStructureMotionInstanceNV",
  "VkPhysicalDeviceRayTracingMotionBlurFeaturesNV": "struct VkPhysicalDeviceRayTracingMotionBlurFeaturesNV",
  "VkPhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT": "struct VkPhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT",
  "VkPhysicalDeviceFragmentDensityMap2FeaturesEXT": "struct VkPhysicalDeviceFragmentDensityMap2FeaturesEXT",
  "VkPhysicalDeviceFragmentDensityMap2PropertiesEXT": "struct VkPhysicalDeviceFragmentDensityMap2PropertiesEXT",
  "VkCopyCommandTransformInfoQCOM": "struct VkCopyCommandTransformInfoQCOM",
  "VkPhysicalDeviceImageRobustnessFeaturesEXT": "VkPhysicalDeviceImageRobustnessFeatures",
  "VkPhysicalDevice4444FormatsFeaturesEXT": "struct VkPhysicalDevice4444FormatsFeaturesEXT",
  "VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM": "struct VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM",
  "VkPhysicalDeviceRGBA10X6FormatsFeaturesEXT": "struct VkPhysicalDeviceRGBA10X6FormatsFeaturesEXT",
  "PFN_vkAcquireWinrtDisplayNV": "VkResult (*)(VkPhysicalDevice, VkDisplayKHR)",
  "PFN_vkGetWinrtDisplayNV": "VkResult (*)(VkPhysicalDevice, uint32_t, VkDisplayKHR *)",
  "VkPhysicalDeviceMutableDescriptorTypeFeaturesVALVE": "struct VkPhysicalDeviceMutableDescriptorTypeFeaturesVALVE",
  "VkMutableDescriptorTypeListVALVE": "struct VkMutableDescriptorTypeListVALVE",
  "VkMutableDescriptorTypeCreateInfoVALVE": "struct VkMutableDescriptorTypeCreateInfoVALVE",
  "VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT": "struct VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT",
  "VkVertexInputBindingDescription2EXT": "struct VkVertexInputBindingDescription2EXT",
  "VkVertexInputAttributeDescription2EXT": "struct VkVertexInputAttributeDescription2EXT",
  "PFN_vkCmdSetVertexInputEXT": "void (*)(VkCommandBuffer, uint32_t, const VkVertexInputBindingDescription2EXT *, uint32_t, const VkVertexInputAttributeDescription2EXT *)",
  "VkPhysicalDeviceDrmPropertiesEXT": "struct VkPhysicalDeviceDrmPropertiesEXT",
  "VkPhysicalDeviceDepthClipControlFeaturesEXT": "struct VkPhysicalDeviceDepthClipControlFeaturesEXT",
  "VkPipelineViewportDepthClipControlCreateInfoEXT": "struct VkPipelineViewportDepthClipControlCreateInfoEXT",
  "VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT": "struct VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT",
  "VkSubpassShadingPipelineCreateInfoHUAWEI": "struct VkSubpassShadingPipelineCreateInfoHUAWEI",
  "VkPhysicalDeviceSubpassShadingFeaturesHUAWEI": "struct VkPhysicalDeviceSubpassShadingFeaturesHUAWEI",
  "VkPhysicalDeviceSubpassShadingPropertiesHUAWEI": "struct VkPhysicalDeviceSubpassShadingPropertiesHUAWEI",
  "PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI": "VkResult (*)(VkDevice, VkRenderPass, VkExtent2D *)",
  "PFN_vkCmdSubpassShadingHUAWEI": "void (*)(VkCommandBuffer)",
  "VkPhysicalDeviceInvocationMaskFeaturesHUAWEI": "struct VkPhysicalDeviceInvocationMaskFeaturesHUAWEI",
  "PFN_vkCmdBindInvocationMaskHUAWEI": "void (*)(VkCommandBuffer, VkImageView, VkImageLayout)",
  "VkRemoteAddressNV": "void *",
  "VkMemoryGetRemoteAddressInfoNV": "struct VkMemoryGetRemoteAddressInfoNV",
  "VkPhysicalDeviceExternalMemoryRDMAFeaturesNV": "struct VkPhysicalDeviceExternalMemoryRDMAFeaturesNV",
  "PFN_vkGetMemoryRemoteAddressNV": "VkResult (*)(VkDevice, const VkMemoryGetRemoteAddressInfoNV *, VkRemoteAddressNV *)",
  "VkPhysicalDeviceExtendedDynamicState2FeaturesEXT": "struct VkPhysicalDeviceExtendedDynamicState2FeaturesEXT",
  "PFN_vkCmdSetPatchControlPointsEXT": "void (*)(VkCommandBuffer, uint32_t)",
  "PFN_vkCmdSetRasterizerDiscardEnableEXT": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetDepthBiasEnableEXT": "void (*)(VkCommandBuffer, VkBool32)",
  "PFN_vkCmdSetLogicOpEXT": "void (*)(VkCommandBuffer, VkLogicOp)",
  "PFN_vkCmdSetPrimitiveRestartEnableEXT": "void (*)(VkCommandBuffer, VkBool32)",
  "VkPhysicalDeviceColorWriteEnableFeaturesEXT": "struct VkPhysicalDeviceColorWriteEnableFeaturesEXT",
  "VkPipelineColorWriteCreateInfoEXT": "struct VkPipelineColorWriteCreateInfoEXT",
  "PFN_vkCmdSetColorWriteEnableEXT": "void (*)(VkCommandBuffer, uint32_t, const VkBool32 *)",
  "VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT": "struct VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT",
  "VkPhysicalDeviceGlobalPriorityQueryFeaturesEXT": "VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR",
  "VkQueueFamilyGlobalPriorityPropertiesEXT": "VkQueueFamilyGlobalPriorityPropertiesKHR",
  "VkPhysicalDeviceImageViewMinLodFeaturesEXT": "struct VkPhysicalDeviceImageViewMinLodFeaturesEXT",
  "VkImageViewMinLodCreateInfoEXT": "struct VkImageViewMinLodCreateInfoEXT",
  "VkPhysicalDeviceMultiDrawFeaturesEXT": "struct VkPhysicalDeviceMultiDrawFeaturesEXT",
  "VkPhysicalDeviceMultiDrawPropertiesEXT": "struct VkPhysicalDeviceMultiDrawPropertiesEXT",
  "VkMultiDrawInfoEXT": "struct VkMultiDrawInfoEXT",
  "VkMultiDrawIndexedInfoEXT": "struct VkMultiDrawIndexedInfoEXT",
  "PFN_vkCmdDrawMultiEXT": "void (*)(VkCommandBuffer, uint32_t, const VkMultiDrawInfoEXT *, uint32_t, uint32_t, uint32_t)",
  "PFN_vkCmdDrawMultiIndexedEXT": "void (*)(VkCommandBuffer, uint32_t, const VkMultiDrawIndexedInfoEXT *, uint32_t, uint32_t, uint32_t, const int32_t *)",
  "VkPhysicalDeviceImage2DViewOf3DFeaturesEXT": "struct VkPhysicalDeviceImage2DViewOf3DFeaturesEXT",
  "VkPhysicalDeviceBorderColorSwizzleFeaturesEXT": "struct VkPhysicalDeviceBorderColorSwizzleFeaturesEXT",
  "VkSamplerBorderColorComponentMappingCreateInfoEXT": "struct VkSamplerBorderColorComponentMappingCreateInfoEXT",
  "VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT": "struct VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT",
  "PFN_vkSetDeviceMemoryPriorityEXT": "void (*)(VkDevice, VkDeviceMemory, float)",
  "VkPhysicalDeviceDescriptorSetHostMappingFeaturesVALVE": "struct VkPhysicalDeviceDescriptorSetHostMappingFeaturesVALVE",
  "VkDescriptorSetBindingReferenceVALVE": "struct VkDescriptorSetBindingReferenceVALVE",
  "VkDescriptorSetLayoutHostMappingInfoVALVE": "struct VkDescriptorSetLayoutHostMappingInfoVALVE",
  "PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE": "void (*)(VkDevice, const VkDescriptorSetBindingReferenceVALVE *, VkDescriptorSetLayoutHostMappingInfoVALVE *)",
  "PFN_vkGetDescriptorSetHostMappingVALVE": "void (*)(VkDevice, VkDescriptorSet, void **)",
  "VkPhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM": "struct VkPhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM",
  "VkPhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM": "struct VkPhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM",
  "VkSubpassFragmentDensityMapOffsetEndInfoQCOM": "struct VkSubpassFragmentDensityMapOffsetEndInfoQCOM",
  "VkPhysicalDeviceLinearColorAttachmentFeaturesNV": "struct VkPhysicalDeviceLinearColorAttachmentFeaturesNV",
  "VkAccelerationStructureKHR": "struct VkAccelerationStructureKHR_T *",
  "VkBuildAccelerationStructureModeKHR": "enum VkBuildAccelerationStructureModeKHR",
  "VkAccelerationStructureBuildTypeKHR": "enum VkAccelerationStructureBuildTypeKHR",
  "VkAccelerationStructureCompatibilityKHR": "enum VkAccelerationStructureCompatibilityKHR",
  "VkAccelerationStructureCreateFlagBitsKHR": "enum VkAccelerationStructureCreateFlagBitsKHR",
  "VkAccelerationStructureCreateFlagsKHR": "VkFlags",
  "VkDeviceOrHostAddressKHR": "union VkDeviceOrHostAddressKHR",
  "VkAccelerationStructureBuildRangeInfoKHR": "struct VkAccelerationStructureBuildRangeInfoKHR",
  "VkAccelerationStructureGeometryTrianglesDataKHR": "struct VkAccelerationStructureGeometryTrianglesDataKHR",
  "VkAccelerationStructureGeometryAabbsDataKHR": "struct VkAccelerationStructureGeometryAabbsDataKHR",
  "VkAccelerationStructureGeometryInstancesDataKHR": "struct VkAccelerationStructureGeometryInstancesDataKHR",
  "VkAccelerationStructureGeometryDataKHR": "union VkAccelerationStructureGeometryDataKHR",
  "VkAccelerationStructureGeometryKHR": "struct VkAccelerationStructureGeometryKHR",
  "VkAccelerationStructureBuildGeometryInfoKHR": "struct VkAccelerationStructureBuildGeometryInfoKHR",
  "VkAccelerationStructureCreateInfoKHR": "struct VkAccelerationStructureCreateInfoKHR",
  "VkWriteDescriptorSetAccelerationStructureKHR": "struct VkWriteDescriptorSetAccelerationStructureKHR",
  "VkPhysicalDeviceAccelerationStructureFeaturesKHR": "struct VkPhysicalDeviceAccelerationStructureFeaturesKHR",
  "VkPhysicalDeviceAccelerationStructurePropertiesKHR": "struct VkPhysicalDeviceAccelerationStructurePropertiesKHR",
  "VkAccelerationStructureDeviceAddressInfoKHR": "struct VkAccelerationStructureDeviceAddressInfoKHR",
  "VkAccelerationStructureVersionInfoKHR": "struct VkAccelerationStructureVersionInfoKHR",
  "VkCopyAccelerationStructureToMemoryInfoKHR": "struct VkCopyAccelerationStructureToMemoryInfoKHR",
  "VkCopyMemoryToAccelerationStructureInfoKHR": "struct VkCopyMemoryToAccelerationStructureInfoKHR",
  "VkCopyAccelerationStructureInfoKHR": "struct VkCopyAccelerationStructureInfoKHR",
  "VkAccelerationStructureBuildSizesInfoKHR": "struct VkAccelerationStructureBuildSizesInfoKHR",
  "PFN_vkCreateAccelerationStructureKHR": "VkResult (*)(VkDevice, const VkAccelerationStructureCreateInfoKHR *, const VkAllocationCallbacks *, VkAccelerationStructureKHR *)",
  "PFN_vkDestroyAccelerationStructureKHR": "void (*)(VkDevice, VkAccelerationStructureKHR, const VkAllocationCallbacks *)",
  "PFN_vkCmdBuildAccelerationStructuresKHR": "void (*)(VkCommandBuffer, uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR *, const VkAccelerationStructureBuildRangeInfoKHR *const *)",
  "PFN_vkCmdBuildAccelerationStructuresIndirectKHR": "void (*)(VkCommandBuffer, uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR *, const VkDeviceAddress *, const uint32_t *, const uint32_t *const *)",
  "PFN_vkBuildAccelerationStructuresKHR": "VkResult (*)(VkDevice, VkDeferredOperationKHR, uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR *, const VkAccelerationStructureBuildRangeInfoKHR *const *)",
  "PFN_vkCopyAccelerationStructureKHR": "VkResult (*)(VkDevice, VkDeferredOperationKHR, const VkCopyAccelerationStructureInfoKHR *)",
  "PFN_vkCopyAccelerationStructureToMemoryKHR": "VkResult (*)(VkDevice, VkDeferredOperationKHR, const VkCopyAccelerationStructureToMemoryInfoKHR *)",
  "PFN_vkCopyMemoryToAccelerationStructureKHR": "VkResult (*)(VkDevice, VkDeferredOperationKHR, const VkCopyMemoryToAccelerationStructureInfoKHR *)",
  "PFN_vkWriteAccelerationStructuresPropertiesKHR": "VkResult (*)(VkDevice, uint32_t, const VkAccelerationStructureKHR *, VkQueryType, size_t, void *, size_t)",
  "PFN_vkCmdCopyAccelerationStructureKHR": "void (*)(VkCommandBuffer, const VkCopyAccelerationStructureInfoKHR *)",
  "PFN_vkCmdCopyAccelerationStructureToMemoryKHR": "void (*)(VkCommandBuffer, const VkCopyAccelerationStructureToMemoryInfoKHR *)",
  "PFN_vkCmdCopyMemoryToAccelerationStructureKHR": "void (*)(VkCommandBuffer, const VkCopyMemoryToAccelerationStructureInfoKHR *)",
  "PFN_vkGetAccelerationStructureDeviceAddressKHR": "VkDeviceAddress (*)(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR *)",
  "PFN_vkCmdWriteAccelerationStructuresPropertiesKHR": "void (*)(VkCommandBuffer, uint32_t, const VkAccelerationStructureKHR *, VkQueryType, VkQueryPool, uint32_t)",
  "PFN_vkGetDeviceAccelerationStructureCompatibilityKHR": "void (*)(VkDevice, const VkAccelerationStructureVersionInfoKHR *, VkAccelerationStructureCompatibilityKHR *)",
  "PFN_vkGetAccelerationStructureBuildSizesKHR": "void (*)(VkDevice, VkAccelerationStructureBuildTypeKHR, const VkAccelerationStructureBuildGeometryInfoKHR *, const uint32_t *, VkAccelerationStructureBuildSizesInfoKHR *)",
  "VkShaderGroupShaderKHR": "enum VkShaderGroupShaderKHR",
  "VkRayTracingShaderGroupCreateInfoKHR": "struct VkRayTracingShaderGroupCreateInfoKHR",
  "VkRayTracingPipelineInterfaceCreateInfoKHR": "struct VkRayTracingPipelineInterfaceCreateInfoKHR",
  "VkRayTracingPipelineCreateInfoKHR": "struct VkRayTracingPipelineCreateInfoKHR",
  "VkPhysicalDeviceRayTracingPipelineFeaturesKHR": "struct VkPhysicalDeviceRayTracingPipelineFeaturesKHR",
  "VkPhysicalDeviceRayTracingPipelinePropertiesKHR": "struct VkPhysicalDeviceRayTracingPipelinePropertiesKHR",
  "VkStridedDeviceAddressRegionKHR": "struct VkStridedDeviceAddressRegionKHR",
  "VkTraceRaysIndirectCommandKHR": "struct VkTraceRaysIndirectCommandKHR",
  "PFN_vkCmdTraceRaysKHR": "void (*)(VkCommandBuffer, const VkStridedDeviceAddressRegionKHR *, const VkStridedDeviceAddressRegionKHR *, const VkStridedDeviceAddressRegionKHR *, const VkStridedDeviceAddressRegionKHR *, uint32_t, uint32_t, uint32_t)",
  "PFN_vkCreateRayTracingPipelinesKHR": "VkResult (*)(VkDevice, VkDeferredOperationKHR, VkPipelineCache, uint32_t, const VkRayTracingPipelineCreateInfoKHR *, const VkAllocationCallbacks *, VkPipeline *)",
  "PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR": "VkResult (*)(VkDevice, VkPipeline, uint32_t, uint32_t, size_t, void *)",
  "PFN_vkCmdTraceRaysIndirectKHR": "void (*)(VkCommandBuffer, const VkStridedDeviceAddressRegionKHR *, const VkStridedDeviceAddressRegionKHR *, const VkStridedDeviceAddressRegionKHR *, const VkStridedDeviceAddressRegionKHR *, VkDeviceAddress)",
  "PFN_vkGetRayTracingShaderGroupStackSizeKHR": "VkDeviceSize (*)(VkDevice, VkPipeline, uint32_t, VkShaderGroupShaderKHR)",
  "PFN_vkCmdSetRayTracingPipelineStackSizeKHR": "void (*)(VkCommandBuffer, uint32_t)",
  "VkPhysicalDeviceRayQueryFeaturesKHR": "struct VkPhysicalDeviceRayQueryFeaturesKHR"
}
VK_PIPELINE_CACHE_HEADER_VERSION_ONE = 1
VK_PIPELINE_CACHE_HEADER_VERSION_MAX_ENUM = 2147483647
VK_VENDOR_ID_VIV = 65537
VK_VENDOR_ID_VSI = 65538
VK_VENDOR_ID_KAZAN = 65539
VK_VENDOR_ID_CODEPLAY = 65540
VK_VENDOR_ID_MESA = 65541
VK_VENDOR_ID_POCL = 65542
VK_VENDOR_ID_MAX_ENUM = 2147483647
VK_SYSTEM_ALLOCATION_SCOPE_COMMAND = 0
VK_SYSTEM_ALLOCATION_SCOPE_OBJECT = 1
VK_SYSTEM_ALLOCATION_SCOPE_CACHE = 2
VK_SYSTEM_ALLOCATION_SCOPE_DEVICE = 3
VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE = 4
VK_SYSTEM_ALLOCATION_SCOPE_MAX_ENUM = 2147483647
VK_INTERNAL_ALLOCATION_TYPE_EXECUTABLE = 0
VK_INTERNAL_ALLOCATION_TYPE_MAX_ENUM = 2147483647
VK_IMAGE_TILING_OPTIMAL = 0
VK_IMAGE_TILING_LINEAR = 1
VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT = 1000158000
VK_IMAGE_TILING_MAX_ENUM = 2147483647
VK_IMAGE_TYPE_1D = 0
VK_IMAGE_TYPE_2D = 1
VK_IMAGE_TYPE_3D = 2
VK_IMAGE_TYPE_MAX_ENUM = 2147483647
VK_PHYSICAL_DEVICE_TYPE_OTHER = 0
VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3
VK_PHYSICAL_DEVICE_TYPE_CPU = 4
VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM = 2147483647
VK_QUERY_TYPE_OCCLUSION = 0
VK_QUERY_TYPE_PIPELINE_STATISTICS = 1
VK_QUERY_TYPE_TIMESTAMP = 2
VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT = 1000028004
VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR = 1000116000
VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR = 1000150000
VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR = 1000150001
VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV = 1000165000
VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL = 1000210000
VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT = 1000382000
VK_QUERY_TYPE_MAX_ENUM = 2147483647
VK_SHARING_MODE_EXCLUSIVE = 0
VK_SHARING_MODE_CONCURRENT = 1
VK_SHARING_MODE_MAX_ENUM = 2147483647
VK_COMPONENT_SWIZZLE_IDENTITY = 0
VK_COMPONENT_SWIZZLE_ZERO = 1
VK_COMPONENT_SWIZZLE_ONE = 2
VK_COMPONENT_SWIZZLE_R = 3
VK_COMPONENT_SWIZZLE_G = 4
VK_COMPONENT_SWIZZLE_B = 5
VK_COMPONENT_SWIZZLE_A = 6
VK_COMPONENT_SWIZZLE_MAX_ENUM = 2147483647
VK_IMAGE_VIEW_TYPE_1D = 0
VK_IMAGE_VIEW_TYPE_2D = 1
VK_IMAGE_VIEW_TYPE_3D = 2
VK_IMAGE_VIEW_TYPE_CUBE = 3
VK_IMAGE_VIEW_TYPE_1D_ARRAY = 4
VK_IMAGE_VIEW_TYPE_2D_ARRAY = 5
VK_IMAGE_VIEW_TYPE_CUBE_ARRAY = 6
VK_IMAGE_VIEW_TYPE_MAX_ENUM = 2147483647
VK_BLEND_FACTOR_ZERO = 0
VK_BLEND_FACTOR_ONE = 1
VK_BLEND_FACTOR_SRC_COLOR = 2
VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR = 3
VK_BLEND_FACTOR_DST_COLOR = 4
VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR = 5
VK_BLEND_FACTOR_SRC_ALPHA = 6
VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA = 7
VK_BLEND_FACTOR_DST_ALPHA = 8
VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA = 9
VK_BLEND_FACTOR_CONSTANT_COLOR = 10
VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR = 11
VK_BLEND_FACTOR_CONSTANT_ALPHA = 12
VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA = 13
VK_BLEND_FACTOR_SRC_ALPHA_SATURATE = 14
VK_BLEND_FACTOR_SRC1_COLOR = 15
VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR = 16
VK_BLEND_FACTOR_SRC1_ALPHA = 17
VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA = 18
VK_BLEND_FACTOR_MAX_ENUM = 2147483647
VK_BLEND_OP_ADD = 0
VK_BLEND_OP_SUBTRACT = 1
VK_BLEND_OP_REVERSE_SUBTRACT = 2
VK_BLEND_OP_MIN = 3
VK_BLEND_OP_MAX = 4
VK_BLEND_OP_ZERO_EXT = 1000148000
VK_BLEND_OP_SRC_EXT = 1000148001
VK_BLEND_OP_DST_EXT = 1000148002
VK_BLEND_OP_SRC_OVER_EXT = 1000148003
VK_BLEND_OP_DST_OVER_EXT = 1000148004
VK_BLEND_OP_SRC_IN_EXT = 1000148005
VK_BLEND_OP_DST_IN_EXT = 1000148006
VK_BLEND_OP_SRC_OUT_EXT = 1000148007
VK_BLEND_OP_DST_OUT_EXT = 1000148008
VK_BLEND_OP_SRC_ATOP_EXT = 1000148009
VK_BLEND_OP_DST_ATOP_EXT = 1000148010
VK_BLEND_OP_XOR_EXT = 1000148011
VK_BLEND_OP_MULTIPLY_EXT = 1000148012
VK_BLEND_OP_SCREEN_EXT = 1000148013
VK_BLEND_OP_OVERLAY_EXT = 1000148014
VK_BLEND_OP_DARKEN_EXT = 1000148015
VK_BLEND_OP_LIGHTEN_EXT = 1000148016
VK_BLEND_OP_COLORDODGE_EXT = 1000148017
VK_BLEND_OP_COLORBURN_EXT = 1000148018
VK_BLEND_OP_HARDLIGHT_EXT = 1000148019
VK_BLEND_OP_SOFTLIGHT_EXT = 1000148020
VK_BLEND_OP_DIFFERENCE_EXT = 1000148021
VK_BLEND_OP_EXCLUSION_EXT = 1000148022
VK_BLEND_OP_INVERT_EXT = 1000148023
VK_BLEND_OP_INVERT_RGB_EXT = 1000148024
VK_BLEND_OP_LINEARDODGE_EXT = 1000148025
VK_BLEND_OP_LINEARBURN_EXT = 1000148026
VK_BLEND_OP_VIVIDLIGHT_EXT = 1000148027
VK_BLEND_OP_LINEARLIGHT_EXT = 1000148028
VK_BLEND_OP_PINLIGHT_EXT = 1000148029
VK_BLEND_OP_HARDMIX_EXT = 1000148030
VK_BLEND_OP_HSL_HUE_EXT = 1000148031
VK_BLEND_OP_HSL_SATURATION_EXT = 1000148032
VK_BLEND_OP_HSL_COLOR_EXT = 1000148033
VK_BLEND_OP_HSL_LUMINOSITY_EXT = 1000148034
VK_BLEND_OP_PLUS_EXT = 1000148035
VK_BLEND_OP_PLUS_CLAMPED_EXT = 1000148036
VK_BLEND_OP_PLUS_CLAMPED_ALPHA_EXT = 1000148037
VK_BLEND_OP_PLUS_DARKER_EXT = 1000148038
VK_BLEND_OP_MINUS_EXT = 1000148039
VK_BLEND_OP_MINUS_CLAMPED_EXT = 1000148040
VK_BLEND_OP_CONTRAST_EXT = 1000148041
VK_BLEND_OP_INVERT_OVG_EXT = 1000148042
VK_BLEND_OP_RED_EXT = 1000148043
VK_BLEND_OP_GREEN_EXT = 1000148044
VK_BLEND_OP_BLUE_EXT = 1000148045
VK_BLEND_OP_MAX_ENUM = 2147483647
VK_COMPARE_OP_NEVER = 0
VK_COMPARE_OP_LESS = 1
VK_COMPARE_OP_EQUAL = 2
VK_COMPARE_OP_LESS_OR_EQUAL = 3
VK_COMPARE_OP_GREATER = 4
VK_COMPARE_OP_NOT_EQUAL = 5
VK_COMPARE_OP_GREATER_OR_EQUAL = 6
VK_COMPARE_OP_ALWAYS = 7
VK_COMPARE_OP_MAX_ENUM = 2147483647
VK_FRONT_FACE_COUNTER_CLOCKWISE = 0
VK_FRONT_FACE_CLOCKWISE = 1
VK_FRONT_FACE_MAX_ENUM = 2147483647
VK_VERTEX_INPUT_RATE_VERTEX = 0
VK_VERTEX_INPUT_RATE_INSTANCE = 1
VK_VERTEX_INPUT_RATE_MAX_ENUM = 2147483647
VK_PRIMITIVE_TOPOLOGY_POINT_LIST = 0
VK_PRIMITIVE_TOPOLOGY_LINE_LIST = 1
VK_PRIMITIVE_TOPOLOGY_LINE_STRIP = 2
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 4
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN = 5
VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY = 6
VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY = 7
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY = 8
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY = 9
VK_PRIMITIVE_TOPOLOGY_PATCH_LIST = 10
VK_PRIMITIVE_TOPOLOGY_MAX_ENUM = 2147483647
VK_POLYGON_MODE_FILL = 0
VK_POLYGON_MODE_LINE = 1
VK_POLYGON_MODE_POINT = 2
VK_POLYGON_MODE_FILL_RECTANGLE_NV = 1000153000
VK_POLYGON_MODE_MAX_ENUM = 2147483647
VK_STENCIL_OP_KEEP = 0
VK_STENCIL_OP_ZERO = 1
VK_STENCIL_OP_REPLACE = 2
VK_STENCIL_OP_INCREMENT_AND_CLAMP = 3
VK_STENCIL_OP_DECREMENT_AND_CLAMP = 4
VK_STENCIL_OP_INVERT = 5
VK_STENCIL_OP_INCREMENT_AND_WRAP = 6
VK_STENCIL_OP_DECREMENT_AND_WRAP = 7
VK_STENCIL_OP_MAX_ENUM = 2147483647
VK_LOGIC_OP_CLEAR = 0
VK_LOGIC_OP_AND = 1
VK_LOGIC_OP_AND_REVERSE = 2
VK_LOGIC_OP_COPY = 3
VK_LOGIC_OP_AND_INVERTED = 4
VK_LOGIC_OP_NO_OP = 5
VK_LOGIC_OP_XOR = 6
VK_LOGIC_OP_OR = 7
VK_LOGIC_OP_NOR = 8
VK_LOGIC_OP_EQUIVALENT = 9
VK_LOGIC_OP_INVERT = 10
VK_LOGIC_OP_OR_REVERSE = 11
VK_LOGIC_OP_COPY_INVERTED = 12
VK_LOGIC_OP_OR_INVERTED = 13
VK_LOGIC_OP_NAND = 14
VK_LOGIC_OP_SET = 15
VK_LOGIC_OP_MAX_ENUM = 2147483647
VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK = 0
VK_BORDER_COLOR_INT_TRANSPARENT_BLACK = 1
VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK = 2
VK_BORDER_COLOR_INT_OPAQUE_BLACK = 3
VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE = 4
VK_BORDER_COLOR_INT_OPAQUE_WHITE = 5
VK_BORDER_COLOR_FLOAT_CUSTOM_EXT = 1000287003
VK_BORDER_COLOR_INT_CUSTOM_EXT = 1000287004
VK_BORDER_COLOR_MAX_ENUM = 2147483647
VK_SAMPLER_MIPMAP_MODE_NEAREST = 0
VK_SAMPLER_MIPMAP_MODE_LINEAR = 1
VK_SAMPLER_MIPMAP_MODE_MAX_ENUM = 2147483647
VK_ATTACHMENT_LOAD_OP_LOAD = 0
VK_ATTACHMENT_LOAD_OP_CLEAR = 1
VK_ATTACHMENT_LOAD_OP_DONT_CARE = 2
VK_ATTACHMENT_LOAD_OP_NONE_EXT = 1000400000
VK_ATTACHMENT_LOAD_OP_MAX_ENUM = 2147483647
VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
VK_COMMAND_BUFFER_LEVEL_SECONDARY = 1
VK_COMMAND_BUFFER_LEVEL_MAX_ENUM = 2147483647
VK_SUBPASS_CONTENTS_INLINE = 0
VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS = 1
VK_SUBPASS_CONTENTS_MAX_ENUM = 2147483647
VK_SAMPLE_COUNT_1_BIT = 1
VK_SAMPLE_COUNT_2_BIT = 2
VK_SAMPLE_COUNT_4_BIT = 4
VK_SAMPLE_COUNT_8_BIT = 8
VK_SAMPLE_COUNT_16_BIT = 16
VK_SAMPLE_COUNT_32_BIT = 32
VK_SAMPLE_COUNT_64_BIT = 64
VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM = 2147483647
VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = 1
VK_INSTANCE_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 2
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 4
VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 8
VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT = 16
VK_MEMORY_PROPERTY_PROTECTED_BIT = 32
VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD = 64
VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD = 128
VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV = 256
VK_MEMORY_PROPERTY_FLAG_BITS_MAX_ENUM = 2147483647
VK_QUEUE_GRAPHICS_BIT = 1
VK_QUEUE_COMPUTE_BIT = 2
VK_QUEUE_TRANSFER_BIT = 4
VK_QUEUE_SPARSE_BINDING_BIT = 8
VK_QUEUE_PROTECTED_BIT = 16
VK_QUEUE_FLAG_BITS_MAX_ENUM = 2147483647
VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT = 1
VK_DEVICE_QUEUE_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_SPARSE_MEMORY_BIND_METADATA_BIT = 1
VK_SPARSE_MEMORY_BIND_FLAG_BITS_MAX_ENUM = 2147483647
VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT = 1
VK_SPARSE_IMAGE_FORMAT_ALIGNED_MIP_SIZE_BIT = 2
VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT = 4
VK_SPARSE_IMAGE_FORMAT_FLAG_BITS_MAX_ENUM = 2147483647
VK_FENCE_CREATE_SIGNALED_BIT = 1
VK_FENCE_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT = 1
VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT = 2
VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT = 4
VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT = 8
VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT = 16
VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT = 32
VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT = 64
VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT = 128
VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT = 256
VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT = 512
VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT = 1024
VK_QUERY_PIPELINE_STATISTIC_FLAG_BITS_MAX_ENUM = 2147483647
VK_QUERY_RESULT_64_BIT = 1
VK_QUERY_RESULT_WAIT_BIT = 2
VK_QUERY_RESULT_WITH_AVAILABILITY_BIT = 4
VK_QUERY_RESULT_PARTIAL_BIT = 8
VK_QUERY_RESULT_FLAG_BITS_MAX_ENUM = 2147483647
VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DYNAMIC_BIT_EXT = 1
VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DEFERRED_BIT_EXT = 2
VK_IMAGE_VIEW_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_COLOR_COMPONENT_R_BIT = 1
VK_COLOR_COMPONENT_G_BIT = 2
VK_COLOR_COMPONENT_B_BIT = 4
VK_COLOR_COMPONENT_A_BIT = 8
VK_COLOR_COMPONENT_FLAG_BITS_MAX_ENUM = 2147483647
VK_CULL_MODE_NONE = 0
VK_CULL_MODE_FRONT_BIT = 1
VK_CULL_MODE_BACK_BIT = 2
VK_CULL_MODE_FRONT_AND_BACK = 3
VK_CULL_MODE_FLAG_BITS_MAX_ENUM = 2147483647
VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_ARM = 1
VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_ARM = 2
VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_ARM = 1
VK_PIPELINE_COLOR_BLEND_STATE_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT = 2
VK_PIPELINE_LAYOUT_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_SAMPLER_CREATE_SUBSAMPLED_BIT_EXT = 1
VK_SAMPLER_CREATE_SUBSAMPLED_COARSE_RECONSTRUCTION_BIT_EXT = 2
VK_SAMPLER_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT = 1
VK_ATTACHMENT_DESCRIPTION_FLAG_BITS_MAX_ENUM = 2147483647
VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM = 2
VK_RENDER_PASS_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_SUBPASS_DESCRIPTION_PER_VIEW_ATTRIBUTES_BIT_NVX = 1
VK_SUBPASS_DESCRIPTION_PER_VIEW_POSITION_X_ONLY_BIT_NVX = 2
VK_SUBPASS_DESCRIPTION_FRAGMENT_REGION_BIT_QCOM = 4
VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM = 8
VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT_ARM = 16
VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_ARM = 32
VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_ARM = 64
VK_SUBPASS_DESCRIPTION_FLAG_BITS_MAX_ENUM = 2147483647
VK_COMMAND_POOL_CREATE_TRANSIENT_BIT = 1
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 2
VK_COMMAND_POOL_CREATE_PROTECTED_BIT = 4
VK_COMMAND_POOL_CREATE_FLAG_BITS_MAX_ENUM = 2147483647
VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT = 1
VK_COMMAND_POOL_RESET_FLAG_BITS_MAX_ENUM = 2147483647
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 1
VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT = 2
VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT = 4
VK_COMMAND_BUFFER_USAGE_FLAG_BITS_MAX_ENUM = 2147483647
VK_QUERY_CONTROL_PRECISE_BIT = 1
VK_QUERY_CONTROL_FLAG_BITS_MAX_ENUM = 2147483647
VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT = 1
VK_COMMAND_BUFFER_RESET_FLAG_BITS_MAX_ENUM = 2147483647
VK_SUBGROUP_FEATURE_BASIC_BIT = 1
VK_SUBGROUP_FEATURE_VOTE_BIT = 2
VK_SUBGROUP_FEATURE_ARITHMETIC_BIT = 4
VK_SUBGROUP_FEATURE_BALLOT_BIT = 8
VK_SUBGROUP_FEATURE_SHUFFLE_BIT = 16
VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT = 32
VK_SUBGROUP_FEATURE_CLUSTERED_BIT = 64
VK_SUBGROUP_FEATURE_QUAD_BIT = 128
VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV = 256
VK_SUBGROUP_FEATURE_FLAG_BITS_MAX_ENUM = 2147483647
VK_PRESENT_MODE_IMMEDIATE_KHR = 0
VK_PRESENT_MODE_MAILBOX_KHR = 1
VK_PRESENT_MODE_FIFO_KHR = 2
VK_PRESENT_MODE_FIFO_RELAXED_KHR = 3
VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR = 1000111000
VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR = 1000111001
VK_PRESENT_MODE_MAX_ENUM_KHR = 2147483647
VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR = 1
VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR = 2
VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR = 4
VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR = 8
VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR = 16
VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR = 32
VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR = 64
VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR = 128
VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR = 256
VK_SURFACE_TRANSFORM_FLAG_BITS_MAX_ENUM_KHR = 2147483647
VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR = 1
VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR = 2
VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR = 4
VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR = 8
VK_COMPOSITE_ALPHA_FLAG_BITS_MAX_ENUM_KHR = 2147483647
VK_SWAPCHAIN_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR = 1
VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR = 2
VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR = 4
VK_SWAPCHAIN_CREATE_FLAG_BITS_MAX_ENUM_KHR = 2147483647
VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR = 1
VK_DEVICE_GROUP_PRESENT_MODE_REMOTE_BIT_KHR = 2
VK_DEVICE_GROUP_PRESENT_MODE_SUM_BIT_KHR = 4
VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_MULTI_DEVICE_BIT_KHR = 8
VK_DEVICE_GROUP_PRESENT_MODE_FLAG_BITS_MAX_ENUM_KHR = 2147483647
VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR = 1
VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR = 2
VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR = 4
VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR = 8
VK_DISPLAY_PLANE_ALPHA_FLAG_BITS_MAX_ENUM_KHR = 2147483647
VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR = 0
VK_PERFORMANCE_COUNTER_UNIT_PERCENTAGE_KHR = 1
VK_PERFORMANCE_COUNTER_UNIT_NANOSECONDS_KHR = 2
VK_PERFORMANCE_COUNTER_UNIT_BYTES_KHR = 3
VK_PERFORMANCE_COUNTER_UNIT_BYTES_PER_SECOND_KHR = 4
VK_PERFORMANCE_COUNTER_UNIT_KELVIN_KHR = 5
VK_PERFORMANCE_COUNTER_UNIT_WATTS_KHR = 6
VK_PERFORMANCE_COUNTER_UNIT_VOLTS_KHR = 7
VK_PERFORMANCE_COUNTER_UNIT_AMPS_KHR = 8
VK_PERFORMANCE_COUNTER_UNIT_HERTZ_KHR = 9
VK_PERFORMANCE_COUNTER_UNIT_CYCLES_KHR = 10
VK_PERFORMANCE_COUNTER_UNIT_MAX_ENUM_KHR = 2147483647
VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR = 0
VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR = 1
VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR = 2
VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR = 3
VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR = 4
VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR = 5
VK_PERFORMANCE_COUNTER_STORAGE_MAX_ENUM_KHR = 2147483647
VK_ACQUIRE_PROFILING_LOCK_FLAG_BITS_MAX_ENUM_KHR = 2147483647
VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR = 0
VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR = 1
VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MIN_KHR = 2
VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_KHR = 3
VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MUL_KHR = 4
VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_ENUM_KHR = 2147483647
VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR = 0
VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR = 1
VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR = 2
VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR = 3
VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_MAX_ENUM_KHR = 2147483647
VK_DEBUG_REPORT_INFORMATION_BIT_EXT = 1
VK_DEBUG_REPORT_WARNING_BIT_EXT = 2
VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT = 4
VK_DEBUG_REPORT_ERROR_BIT_EXT = 8
VK_DEBUG_REPORT_DEBUG_BIT_EXT = 16
VK_DEBUG_REPORT_FLAG_BITS_MAX_ENUM_EXT = 2147483647
VK_RASTERIZATION_ORDER_STRICT_AMD = 0
VK_RASTERIZATION_ORDER_RELAXED_AMD = 1
VK_RASTERIZATION_ORDER_MAX_ENUM_AMD = 2147483647
VK_SHADER_INFO_TYPE_STATISTICS_AMD = 0
VK_SHADER_INFO_TYPE_BINARY_AMD = 1
VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD = 2
VK_SHADER_INFO_TYPE_MAX_ENUM_AMD = 2147483647
VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_NV = 1
VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_NV = 2
VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_IMAGE_BIT_NV = 4
VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_IMAGE_KMT_BIT_NV = 8
VK_EXTERNAL_MEMORY_HANDLE_TYPE_FLAG_BITS_MAX_ENUM_NV = 2147483647
VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT_NV = 1
VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT_NV = 2
VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT_NV = 4
VK_EXTERNAL_MEMORY_FEATURE_FLAG_BITS_MAX_ENUM_NV = 2147483647
VK_VALIDATION_CHECK_ALL_EXT = 0
VK_VALIDATION_CHECK_SHADERS_EXT = 1
VK_VALIDATION_CHECK_MAX_ENUM_EXT = 2147483647
VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT = 1
VK_CONDITIONAL_RENDERING_FLAG_BITS_MAX_ENUM_EXT = 2147483647
VK_DISPLAY_POWER_STATE_OFF_EXT = 0
VK_DISPLAY_POWER_STATE_SUSPEND_EXT = 1
VK_DISPLAY_POWER_STATE_ON_EXT = 2
VK_DISPLAY_POWER_STATE_MAX_ENUM_EXT = 2147483647
VK_DEVICE_EVENT_TYPE_DISPLAY_HOTPLUG_EXT = 0
VK_DEVICE_EVENT_TYPE_MAX_ENUM_EXT = 2147483647
VK_DISPLAY_EVENT_TYPE_FIRST_PIXEL_OUT_EXT = 0
VK_DISPLAY_EVENT_TYPE_MAX_ENUM_EXT = 2147483647
VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV = 0
VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_X_NV = 1
VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV = 2
VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV = 3
VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV = 4
VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV = 5
VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV = 6
VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_W_NV = 7
VK_VIEWPORT_COORDINATE_SWIZZLE_MAX_ENUM_NV = 2147483647
VK_DISCARD_RECTANGLE_MODE_INCLUSIVE_EXT = 0
VK_DISCARD_RECTANGLE_MODE_EXCLUSIVE_EXT = 1
VK_DISCARD_RECTANGLE_MODE_MAX_ENUM_EXT = 2147483647
VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT = 0
VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT = 1
VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT = 2
VK_CONSERVATIVE_RASTERIZATION_MODE_MAX_ENUM_EXT = 2147483647
VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT = 1
VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT = 16
VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT = 256
VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT = 4096
VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT = 2147483647
VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT = 1
VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT = 2
VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT = 4
VK_DEBUG_UTILS_MESSAGE_TYPE_FLAG_BITS_MAX_ENUM_EXT = 2147483647
VK_BLEND_OVERLAP_UNCORRELATED_EXT = 0
VK_BLEND_OVERLAP_DISJOINT_EXT = 1
VK_BLEND_OVERLAP_CONJOINT_EXT = 2
VK_BLEND_OVERLAP_MAX_ENUM_EXT = 2147483647
VK_COVERAGE_MODULATION_MODE_NONE_NV = 0
VK_COVERAGE_MODULATION_MODE_RGB_NV = 1
VK_COVERAGE_MODULATION_MODE_ALPHA_NV = 2
VK_COVERAGE_MODULATION_MODE_RGBA_NV = 3
VK_COVERAGE_MODULATION_MODE_MAX_ENUM_NV = 2147483647
VK_VALIDATION_CACHE_HEADER_VERSION_ONE_EXT = 1
VK_VALIDATION_CACHE_HEADER_VERSION_MAX_ENUM_EXT = 2147483647
VK_SHADING_RATE_PALETTE_ENTRY_NO_INVOCATIONS_NV = 0
VK_SHADING_RATE_PALETTE_ENTRY_16_INVOCATIONS_PER_PIXEL_NV = 1
VK_SHADING_RATE_PALETTE_ENTRY_8_INVOCATIONS_PER_PIXEL_NV = 2
VK_SHADING_RATE_PALETTE_ENTRY_4_INVOCATIONS_PER_PIXEL_NV = 3
VK_SHADING_RATE_PALETTE_ENTRY_2_INVOCATIONS_PER_PIXEL_NV = 4
VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_PIXEL_NV = 5
VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X1_PIXELS_NV = 6
VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_1X2_PIXELS_NV = 7
VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X2_PIXELS_NV = 8
VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_4X2_PIXELS_NV = 9
VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X4_PIXELS_NV = 10
VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_4X4_PIXELS_NV = 11
VK_SHADING_RATE_PALETTE_ENTRY_MAX_ENUM_NV = 2147483647
VK_COARSE_SAMPLE_ORDER_TYPE_DEFAULT_NV = 0
VK_COARSE_SAMPLE_ORDER_TYPE_CUSTOM_NV = 1
VK_COARSE_SAMPLE_ORDER_TYPE_PIXEL_MAJOR_NV = 2
VK_COARSE_SAMPLE_ORDER_TYPE_SAMPLE_MAJOR_NV = 3
VK_COARSE_SAMPLE_ORDER_TYPE_MAX_ENUM_NV = 2147483647
VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV = 0
VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV = 1
VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV = 2
VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_MAX_ENUM_NV = 2147483647
VK_PIPELINE_COMPILER_CONTROL_FLAG_BITS_MAX_ENUM_AMD = 2147483647
VK_TIME_DOMAIN_DEVICE_EXT = 0
VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT = 1
VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT = 2
VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT = 3
VK_TIME_DOMAIN_MAX_ENUM_EXT = 2147483647
VK_MEMORY_OVERALLOCATION_BEHAVIOR_DEFAULT_AMD = 0
VK_MEMORY_OVERALLOCATION_BEHAVIOR_ALLOWED_AMD = 1
VK_MEMORY_OVERALLOCATION_BEHAVIOR_DISALLOWED_AMD = 2
VK_MEMORY_OVERALLOCATION_BEHAVIOR_MAX_ENUM_AMD = 2147483647
VK_PERFORMANCE_CONFIGURATION_TYPE_COMMAND_QUEUE_METRICS_DISCOVERY_ACTIVATED_INTEL = 0
VK_PERFORMANCE_CONFIGURATION_TYPE_MAX_ENUM_INTEL = 2147483647
VK_QUERY_POOL_SAMPLING_MODE_MANUAL_INTEL = 0
VK_QUERY_POOL_SAMPLING_MODE_MAX_ENUM_INTEL = 2147483647
VK_PERFORMANCE_OVERRIDE_TYPE_NULL_HARDWARE_INTEL = 0
VK_PERFORMANCE_OVERRIDE_TYPE_FLUSH_GPU_CACHES_INTEL = 1
VK_PERFORMANCE_OVERRIDE_TYPE_MAX_ENUM_INTEL = 2147483647
VK_PERFORMANCE_PARAMETER_TYPE_HW_COUNTERS_SUPPORTED_INTEL = 0
VK_PERFORMANCE_PARAMETER_TYPE_STREAM_MARKER_VALID_BITS_INTEL = 1
VK_PERFORMANCE_PARAMETER_TYPE_MAX_ENUM_INTEL = 2147483647
VK_PERFORMANCE_VALUE_TYPE_UINT32_INTEL = 0
VK_PERFORMANCE_VALUE_TYPE_UINT64_INTEL = 1
VK_PERFORMANCE_VALUE_TYPE_FLOAT_INTEL = 2
VK_PERFORMANCE_VALUE_TYPE_BOOL_INTEL = 3
VK_PERFORMANCE_VALUE_TYPE_STRING_INTEL = 4
VK_PERFORMANCE_VALUE_TYPE_MAX_ENUM_INTEL = 2147483647
VK_SHADER_CORE_PROPERTIES_FLAG_BITS_MAX_ENUM_AMD = 2147483647
VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT = 0
VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT = 1
VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT = 2
VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT = 3
VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT = 4
VK_VALIDATION_FEATURE_ENABLE_MAX_ENUM_EXT = 2147483647
VK_VALIDATION_FEATURE_DISABLE_ALL_EXT = 0
VK_VALIDATION_FEATURE_DISABLE_SHADERS_EXT = 1
VK_VALIDATION_FEATURE_DISABLE_THREAD_SAFETY_EXT = 2
VK_VALIDATION_FEATURE_DISABLE_API_PARAMETERS_EXT = 3
VK_VALIDATION_FEATURE_DISABLE_OBJECT_LIFETIMES_EXT = 4
VK_VALIDATION_FEATURE_DISABLE_CORE_CHECKS_EXT = 5
VK_VALIDATION_FEATURE_DISABLE_UNIQUE_HANDLES_EXT = 6
VK_VALIDATION_FEATURE_DISABLE_SHADER_VALIDATION_CACHE_EXT = 7
VK_VALIDATION_FEATURE_DISABLE_MAX_ENUM_EXT = 2147483647
VK_COMPONENT_TYPE_FLOAT16_NV = 0
VK_COMPONENT_TYPE_FLOAT32_NV = 1
VK_COMPONENT_TYPE_FLOAT64_NV = 2
VK_COMPONENT_TYPE_SINT8_NV = 3
VK_COMPONENT_TYPE_SINT16_NV = 4
VK_COMPONENT_TYPE_SINT32_NV = 5
VK_COMPONENT_TYPE_SINT64_NV = 6
VK_COMPONENT_TYPE_UINT8_NV = 7
VK_COMPONENT_TYPE_UINT16_NV = 8
VK_COMPONENT_TYPE_UINT32_NV = 9
VK_COMPONENT_TYPE_UINT64_NV = 10
VK_COMPONENT_TYPE_MAX_ENUM_NV = 2147483647
VK_SCOPE_DEVICE_NV = 1
VK_SCOPE_WORKGROUP_NV = 2
VK_SCOPE_SUBGROUP_NV = 3
VK_SCOPE_QUEUE_FAMILY_NV = 5
VK_SCOPE_MAX_ENUM_NV = 2147483647
VK_COVERAGE_REDUCTION_MODE_MERGE_NV = 0
VK_COVERAGE_REDUCTION_MODE_TRUNCATE_NV = 1
VK_COVERAGE_REDUCTION_MODE_MAX_ENUM_NV = 2147483647
VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT = 0
VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT = 1
VK_PROVOKING_VERTEX_MODE_MAX_ENUM_EXT = 2147483647
VK_LINE_RASTERIZATION_MODE_DEFAULT_EXT = 0
VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT = 1
VK_LINE_RASTERIZATION_MODE_BRESENHAM_EXT = 2
VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT = 3
VK_LINE_RASTERIZATION_MODE_MAX_ENUM_EXT = 2147483647
VK_INDIRECT_COMMANDS_TOKEN_TYPE_SHADER_GROUP_NV = 0
VK_INDIRECT_COMMANDS_TOKEN_TYPE_STATE_FLAGS_NV = 1
VK_INDIRECT_COMMANDS_TOKEN_TYPE_INDEX_BUFFER_NV = 2
VK_INDIRECT_COMMANDS_TOKEN_TYPE_VERTEX_BUFFER_NV = 3
VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_NV = 4
VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_NV = 5
VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_NV = 6
VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_TASKS_NV = 7
VK_INDIRECT_COMMANDS_TOKEN_TYPE_MAX_ENUM_NV = 2147483647
VK_INDIRECT_STATE_FLAG_FRONTFACE_BIT_NV = 1
VK_INDIRECT_STATE_FLAG_BITS_MAX_ENUM_NV = 2147483647
VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_NV = 1
VK_INDIRECT_COMMANDS_LAYOUT_USAGE_INDEXED_SEQUENCES_BIT_NV = 2
VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_NV = 4
VK_INDIRECT_COMMANDS_LAYOUT_USAGE_FLAG_BITS_MAX_ENUM_NV = 2147483647
VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT = 0
VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_FREE_EXT = 1
VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_IMPORT_EXT = 2
VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_UNIMPORT_EXT = 3
VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATION_FAILED_EXT = 4
VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_MAX_ENUM_EXT = 2147483647
VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV = 1
VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV = 2
VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV = 4
VK_DEVICE_DIAGNOSTICS_CONFIG_FLAG_BITS_MAX_ENUM_NV = 2147483647
VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT = 1
VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT = 2
VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT = 4
VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT = 8
VK_GRAPHICS_PIPELINE_LIBRARY_FLAG_BITS_MAX_ENUM_EXT = 2147483647
VK_FRAGMENT_SHADING_RATE_TYPE_FRAGMENT_SIZE_NV = 0
VK_FRAGMENT_SHADING_RATE_TYPE_ENUMS_NV = 1
VK_FRAGMENT_SHADING_RATE_TYPE_MAX_ENUM_NV = 2147483647
VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_PIXEL_NV = 0
VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_1X2_PIXELS_NV = 1
VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_2X1_PIXELS_NV = 4
VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_2X2_PIXELS_NV = 5
VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_2X4_PIXELS_NV = 6
VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_4X2_PIXELS_NV = 9
VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_4X4_PIXELS_NV = 10
VK_FRAGMENT_SHADING_RATE_2_INVOCATIONS_PER_PIXEL_NV = 11
VK_FRAGMENT_SHADING_RATE_4_INVOCATIONS_PER_PIXEL_NV = 12
VK_FRAGMENT_SHADING_RATE_8_INVOCATIONS_PER_PIXEL_NV = 13
VK_FRAGMENT_SHADING_RATE_16_INVOCATIONS_PER_PIXEL_NV = 14
VK_FRAGMENT_SHADING_RATE_NO_INVOCATIONS_NV = 15
VK_FRAGMENT_SHADING_RATE_MAX_ENUM_NV = 2147483647
VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV = 0
VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV = 1
VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV = 2
VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MAX_ENUM_NV = 2147483647
VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR = 0
VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR = 1
VK_BUILD_ACCELERATION_STRUCTURE_MODE_MAX_ENUM_KHR = 2147483647
VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR = 0
VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR = 1
VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR = 2
VK_ACCELERATION_STRUCTURE_BUILD_TYPE_MAX_ENUM_KHR = 2147483647
VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR = 0
VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR = 1
VK_ACCELERATION_STRUCTURE_COMPATIBILITY_MAX_ENUM_KHR = 2147483647
VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR = 1
VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV = 4
VK_ACCELERATION_STRUCTURE_CREATE_FLAG_BITS_MAX_ENUM_KHR = 2147483647
VK_SHADER_GROUP_SHADER_GENERAL_KHR = 0
VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR = 1
VK_SHADER_GROUP_SHADER_ANY_HIT_KHR = 2
VK_SHADER_GROUP_SHADER_INTERSECTION_KHR = 3
VK_SHADER_GROUP_SHADER_MAX_ENUM_KHR = 2147483647
VkBuffer_T =  c_void_p 
VkImage_T =  c_void_p 
VkInstance_T =  c_void_p 
VkPhysicalDevice_T =  c_void_p 
VkDevice_T =  c_void_p 
VkQueue_T =  c_void_p 
VkSemaphore_T =  c_void_p 
VkCommandBuffer_T =  c_void_p 
VkFence_T =  c_void_p 
VkDeviceMemory_T =  c_void_p 
VkEvent_T =  c_void_p 
VkQueryPool_T =  c_void_p 
VkBufferView_T =  c_void_p 
VkImageView_T =  c_void_p 
VkShaderModule_T =  c_void_p 
VkPipelineCache_T =  c_void_p 
VkPipelineLayout_T =  c_void_p 
VkPipeline_T =  c_void_p 
VkRenderPass_T =  c_void_p 
VkDescriptorSetLayout_T =  c_void_p 
VkSampler_T =  c_void_p 
VkDescriptorSet_T =  c_void_p 
VkDescriptorPool_T =  c_void_p 
VkFramebuffer_T =  c_void_p 
VkCommandPool_T =  c_void_p 
VkSamplerYcbcrConversion_T =  c_void_p 
VkDescriptorUpdateTemplate_T =  c_void_p 
VkPrivateDataSlot_T =  c_void_p 
VkSurfaceKHR_T =  c_void_p 
VkSwapchainKHR_T =  c_void_p 
VkDisplayKHR_T =  c_void_p 
VkDisplayModeKHR_T =  c_void_p 
VkDeferredOperationKHR_T =  c_void_p 
VkDebugReportCallbackEXT_T =  c_void_p 
VkCuModuleNVX_T =  c_void_p 
VkCuFunctionNVX_T =  c_void_p 
VkDebugUtilsMessengerEXT_T =  c_void_p 
VkValidationCacheEXT_T =  c_void_p 
VkAccelerationStructureNV_T =  c_void_p 
VkPerformanceConfigurationINTEL_T =  c_void_p 
VkIndirectCommandsLayoutNV_T =  c_void_p 
VkAccelerationStructureKHR_T =  c_void_p 
class VkExtent2D(Structure):
    pass
VkExtent2D._fields_ = [
             ("width", c_uint),
             ("height", c_uint)
    ]

class VkExtent3D(Structure):
    pass
VkExtent3D._fields_ = [
             ("width", c_uint),
             ("height", c_uint),
             ("depth", c_uint)
    ]

class VkOffset2D(Structure):
    pass
VkOffset2D._fields_ = [
             ("x", c_int),
             ("y", c_int)
    ]

class VkOffset3D(Structure):
    pass
VkOffset3D._fields_ = [
             ("x", c_int),
             ("y", c_int),
             ("z", c_int)
    ]

class VkRect2D(Structure):
    pass
VkRect2D._fields_ = [
             ("offset", VkOffset2D),
             ("extent", VkExtent2D)
    ]

class VkBaseInStructure(Structure):
    pass
VkBaseInStructure._fields_ = [
             ("sType", c_int),
             ("pNext", POINTER(VkBaseInStructure))
    ]

class VkBaseOutStructure(Structure):
    pass
VkBaseOutStructure._fields_ = [
             ("sType", c_int),
             ("pNext", POINTER(VkBaseOutStructure))
    ]

class VkBufferMemoryBarrier(Structure):
    pass
VkBufferMemoryBarrier._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcAccessMask", c_uint),
             ("dstAccessMask", c_uint),
             ("srcQueueFamilyIndex", c_uint),
             ("dstQueueFamilyIndex", c_uint),
             ("buffer", VkBuffer_T),
             ("offset", c_ulong),
             ("size", c_ulong)
    ]

class VkDispatchIndirectCommand(Structure):
    pass
VkDispatchIndirectCommand._fields_ = [
             ("x", c_uint),
             ("y", c_uint),
             ("z", c_uint)
    ]

class VkDrawIndexedIndirectCommand(Structure):
    pass
VkDrawIndexedIndirectCommand._fields_ = [
             ("indexCount", c_uint),
             ("instanceCount", c_uint),
             ("firstIndex", c_uint),
             ("vertexOffset", c_int),
             ("firstInstance", c_uint)
    ]

class VkDrawIndirectCommand(Structure):
    pass
VkDrawIndirectCommand._fields_ = [
             ("vertexCount", c_uint),
             ("instanceCount", c_uint),
             ("firstc_uintertex", c_uint),
             ("firstInstance", c_uint)
    ]

class VkImageSubresourceRange(Structure):
    pass
VkImageSubresourceRange._fields_ = [
             ("aspectMask", c_uint),
             ("baseMipLevel", c_uint),
             ("levelCount", c_uint),
             ("baseArrayLayer", c_uint),
             ("layerCount", c_uint)
    ]

class VkImageMemoryBarrier(Structure):
    pass
VkImageMemoryBarrier._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcAccessMask", c_uint),
             ("dstAccessMask", c_uint),
             ("oldLayout", c_int),
             ("newLayout", c_int),
             ("srcQueueFamilyIndex", c_uint),
             ("dstQueueFamilyIndex", c_uint),
             ("image", VkImage_T),
             ("subresourceRange", VkImageSubresourceRange)
    ]

class VkMemoryBarrier(Structure):
    pass
VkMemoryBarrier._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcAccessMask", c_uint),
             ("dstAccessMask", c_uint)
    ]

class VkPipelineCacheHeaderVersionOne(Structure):
    pass
VkPipelineCacheHeaderVersionOne._fields_ = [
             ("headerSize", c_uint),
             ("headerc_intersion", c_int),
             ("vendorID", c_uint),
             ("deviceID", c_uint),
             ("pipelineCacheUUID", c_ubyte *16)
    ]

class VkAllocationCallbacks(Structure):
    pass
VkAllocationCallbacks._fields_ = [
             ("pUserData", c_void_p),
             ("pfnAllocation", c_void_p),
             ("pfnReallocation", c_void_p),
             ("pfnFree", c_void_p),
             ("pfnInternalAllocation", c_void_p),
             ("pfnInternalFree", c_void_p)
    ]

class VkApplicationInfo(Structure):
    pass
VkApplicationInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pApplicationName", c_char_p),
             ("applicationc_uintersion", c_uint),
             ("pEngineName", c_char_p),
             ("enginec_uintersion", c_uint),
             ("apic_uintersion", c_uint)
    ]

class VkFormatProperties(Structure):
    pass
VkFormatProperties._fields_ = [
             ("linearTilingFeatures", c_uint),
             ("optimalTilingFeatures", c_uint),
             ("bufferFeatures", c_uint)
    ]

class VkImageFormatProperties(Structure):
    pass
VkImageFormatProperties._fields_ = [
             ("maxExtent", VkExtent3D),
             ("maxMipLevels", c_uint),
             ("maxArrayLayers", c_uint),
             ("sampleCounts", c_uint),
             ("maxResourceSize", c_ulong)
    ]

class VkInstanceCreateInfo(Structure):
    pass
VkInstanceCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("pApplicationInfo", VkApplicationInfo),
             ("enabledLayerCount", c_uint),
             ("ppEnabledLayerNames", POINTER(c_byte)),
             ("enabledExtensionCount", c_uint),
             ("ppEnabledExtensionNames", POINTER(c_byte))
    ]

class VkMemoryHeap(Structure):
    pass
VkMemoryHeap._fields_ = [
             ("size", c_ulong),
             ("flags", c_uint)
    ]

class VkMemoryType(Structure):
    pass
VkMemoryType._fields_ = [
             ("propertyFlags", c_uint),
             ("heapIndex", c_uint)
    ]

class VkPhysicalDeviceFeatures(Structure):
    pass
VkPhysicalDeviceFeatures._fields_ = [
             ("robustBufferAccess", c_uint),
             ("fullDrawIndexUint32", c_uint),
             ("imageCubeArray", c_uint),
             ("independentBlend", c_uint),
             ("geometryShader", c_uint),
             ("tessellationShader", c_uint),
             ("sampleRateShading", c_uint),
             ("dualSrcBlend", c_uint),
             ("logicOp", c_uint),
             ("multiDrawIndirect", c_uint),
             ("drawIndirectFirstInstance", c_uint),
             ("depthClamp", c_uint),
             ("depthBiasClamp", c_uint),
             ("fillModeNonSolid", c_uint),
             ("depthBounds", c_uint),
             ("wideLines", c_uint),
             ("largePoints", c_uint),
             ("alphaToOne", c_uint),
             ("multic_uintiewport", c_uint),
             ("samplerAnisotropy", c_uint),
             ("textureCompressionETC2", c_uint),
             ("textureCompressionASTC_LDR", c_uint),
             ("textureCompressionBC", c_uint),
             ("occlusionQueryPrecise", c_uint),
             ("pipelineStatisticsQuery", c_uint),
             ("vertexPipelineStoresAndAtomics", c_uint),
             ("fragmentStoresAndAtomics", c_uint),
             ("shaderTessellationAndGeometryPointSize", c_uint),
             ("shaderImageGatherExtended", c_uint),
             ("shaderStorageImageExtendedFormats", c_uint),
             ("shaderStorageImageMultisample", c_uint),
             ("shaderStorageImageReadWithoutFormat", c_uint),
             ("shaderStorageImageWriteWithoutFormat", c_uint),
             ("shaderUniformBufferArrayDynamicIndexing", c_uint),
             ("shaderSampledImageArrayDynamicIndexing", c_uint),
             ("shaderStorageBufferArrayDynamicIndexing", c_uint),
             ("shaderStorageImageArrayDynamicIndexing", c_uint),
             ("shaderClipDistance", c_uint),
             ("shaderCullDistance", c_uint),
             ("shaderFloat64", c_uint),
             ("shaderInt64", c_uint),
             ("shaderInt16", c_uint),
             ("shaderResourceResidency", c_uint),
             ("shaderResourceMinLod", c_uint),
             ("sparseBinding", c_uint),
             ("sparseResidencyBuffer", c_uint),
             ("sparseResidencyImage2D", c_uint),
             ("sparseResidencyImage3D", c_uint),
             ("sparseResidency2Samples", c_uint),
             ("sparseResidency4Samples", c_uint),
             ("sparseResidency8Samples", c_uint),
             ("sparseResidency16Samples", c_uint),
             ("sparseResidencyAliased", c_uint),
             ("variableMultisampleRate", c_uint),
             ("inheritedQueries", c_uint)
    ]

class VkPhysicalDeviceLimits(Structure):
    pass
VkPhysicalDeviceLimits._fields_ = [
             ("maxImageDimension1D", c_uint),
             ("maxImageDimension2D", c_uint),
             ("maxImageDimension3D", c_uint),
             ("maxImageDimensionCube", c_uint),
             ("maxImageArrayLayers", c_uint),
             ("maxTexelBufferElements", c_uint),
             ("maxUniformBufferRange", c_uint),
             ("maxStorageBufferRange", c_uint),
             ("maxPushConstantsSize", c_uint),
             ("maxMemoryAllocationCount", c_uint),
             ("maxSamplerAllocationCount", c_uint),
             ("bufferImageGranularity", c_ulong),
             ("sparseAddressSpaceSize", c_ulong),
             ("maxBoundDescriptorSets", c_uint),
             ("maxPerStageDescriptorSamplers", c_uint),
             ("maxPerStageDescriptorUniformBuffers", c_uint),
             ("maxPerStageDescriptorStorageBuffers", c_uint),
             ("maxPerStageDescriptorSampledImages", c_uint),
             ("maxPerStageDescriptorStorageImages", c_uint),
             ("maxPerStageDescriptorInputAttachments", c_uint),
             ("maxPerStageResources", c_uint),
             ("maxDescriptorSetSamplers", c_uint),
             ("maxDescriptorSetUniformBuffers", c_uint),
             ("maxDescriptorSetUniformBuffersDynamic", c_uint),
             ("maxDescriptorSetStorageBuffers", c_uint),
             ("maxDescriptorSetStorageBuffersDynamic", c_uint),
             ("maxDescriptorSetSampledImages", c_uint),
             ("maxDescriptorSetStorageImages", c_uint),
             ("maxDescriptorSetInputAttachments", c_uint),
             ("maxc_uintertexInputAttributes", c_uint),
             ("maxc_uintertexInputBindings", c_uint),
             ("maxc_uintertexInputAttributeOffset", c_uint),
             ("maxc_uintertexInputBindingStride", c_uint),
             ("maxc_uintertexOutputComponents", c_uint),
             ("maxTessellationGenerationLevel", c_uint),
             ("maxTessellationPatchSize", c_uint),
             ("maxTessellationControlPerc_uintertexInputComponents", c_uint),
             ("maxTessellationControlPerc_uintertexOutputComponents", c_uint),
             ("maxTessellationControlPerPatchOutputComponents", c_uint),
             ("maxTessellationControlTotalOutputComponents", c_uint),
             ("maxTessellationEvaluationInputComponents", c_uint),
             ("maxTessellationEvaluationOutputComponents", c_uint),
             ("maxGeometryShaderInvocations", c_uint),
             ("maxGeometryInputComponents", c_uint),
             ("maxGeometryOutputComponents", c_uint),
             ("maxGeometryOutputc_uintertices", c_uint),
             ("maxGeometryTotalOutputComponents", c_uint),
             ("maxFragmentInputComponents", c_uint),
             ("maxFragmentOutputAttachments", c_uint),
             ("maxFragmentDualSrcAttachments", c_uint),
             ("maxFragmentCombinedOutputResources", c_uint),
             ("maxComputeSharedMemorySize", c_uint),
             ("maxComputeWorkGroupCount", c_uint *3),
             ("maxComputeWorkGroupInvocations", c_uint),
             ("maxComputeWorkGroupSize", c_uint *3),
             ("subPixelPrecisionBits", c_uint),
             ("subTexelPrecisionBits", c_uint),
             ("mipmapPrecisionBits", c_uint),
             ("maxDrawIndexedIndexc_uintalue", c_uint),
             ("maxDrawIndirectCount", c_uint),
             ("maxSamplerLodBias", c_float),
             ("maxSamplerAnisotropy", c_float),
             ("maxc_uintiewports", c_uint),
             ("maxc_uint *2iewportDimensions", c_uint *2),
             ("viewportBoundsRange", c_float *2),
             ("viewportSubPixelBits", c_uint),
             ("minMemoryMapAlignment", c_ulong),
             ("minTexelBufferOffsetAlignment", c_ulong),
             ("minUniformBufferOffsetAlignment", c_ulong),
             ("minStorageBufferOffsetAlignment", c_ulong),
             ("minTexelOffset", c_int),
             ("maxTexelOffset", c_uint),
             ("minTexelGatherOffset", c_int),
             ("maxTexelGatherOffset", c_uint),
             ("minInterpolationOffset", c_float),
             ("maxInterpolationOffset", c_float),
             ("subPixelInterpolationOffsetBits", c_uint),
             ("maxFramebufferWidth", c_uint),
             ("maxFramebufferHeight", c_uint),
             ("maxFramebufferLayers", c_uint),
             ("framebufferColorSampleCounts", c_uint),
             ("framebufferDepthSampleCounts", c_uint),
             ("framebufferStencilSampleCounts", c_uint),
             ("framebufferNoAttachmentsSampleCounts", c_uint),
             ("maxColorAttachments", c_uint),
             ("sampledImageColorSampleCounts", c_uint),
             ("sampledImageIntegerSampleCounts", c_uint),
             ("sampledImageDepthSampleCounts", c_uint),
             ("sampledImageStencilSampleCounts", c_uint),
             ("storageImageSampleCounts", c_uint),
             ("maxSampleMaskWords", c_uint),
             ("timestampComputeAndGraphics", c_uint),
             ("timestampPeriod", c_float),
             ("maxClipDistances", c_uint),
             ("maxCullDistances", c_uint),
             ("maxCombinedClipAndCullDistances", c_uint),
             ("discreteQueuePriorities", c_uint),
             ("pointSizeRange", c_float *2),
             ("lineWidthRange", c_float *2),
             ("pointSizeGranularity", c_float),
             ("lineWidthGranularity", c_float),
             ("strictLines", c_uint),
             ("standardSampleLocations", c_uint),
             ("optimalBufferCopyOffsetAlignment", c_ulong),
             ("optimalBufferCopyRowPitchAlignment", c_ulong),
             ("nonCoherentAtomSize", c_ulong)
    ]

class VkPhysicalDeviceMemoryProperties(Structure):
    pass
VkPhysicalDeviceMemoryProperties._fields_ = [
             ("memoryTypeCount", c_uint),
             ("memoryTypes", VkMemoryType *32),
             ("memoryHeapCount", c_uint),
             ("memoryHeaps", VkMemoryHeap *16)
    ]

class VkPhysicalDeviceSparseProperties(Structure):
    pass
VkPhysicalDeviceSparseProperties._fields_ = [
             ("residencyStandard2DBlockShape", c_uint),
             ("residencyStandard2DMultisampleBlockShape", c_uint),
             ("residencyStandard3DBlockShape", c_uint),
             ("residencyAlignedMipSize", c_uint),
             ("residencyNonResidentStrict", c_uint)
    ]

class VkPhysicalDeviceProperties(Structure):
    pass
VkPhysicalDeviceProperties._fields_ = [
             ("apic_uintersion", c_uint),
             ("driverc_uintersion", c_uint),
             ("vendorID", c_uint),
             ("deviceID", c_uint),
             ("deviceType", c_int),
             ("deviceName", c_byte *256),
             ("pipelineCacheUUID", c_ubyte *16),
             ("limits", VkPhysicalDeviceLimits),
             ("sparseProperties", VkPhysicalDeviceSparseProperties)
    ]

class VkQueueFamilyProperties(Structure):
    pass
VkQueueFamilyProperties._fields_ = [
             ("queueFlags", c_uint),
             ("queueCount", c_uint),
             ("timestampc_uintalidBits", c_uint),
             ("minImageTransferGranularity", VkExtent3D)
    ]

class VkDeviceQueueCreateInfo(Structure):
    pass
VkDeviceQueueCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("queueFamilyIndex", c_uint),
             ("queueCount", c_uint),
             ("pQueuePriorities", POINTER(c_float))
    ]

class VkDeviceCreateInfo(Structure):
    pass
VkDeviceCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("queueCreateInfoCount", c_uint),
             ("pQueueCreateInfos", VkDeviceQueueCreateInfo),
             ("enabledLayerCount", c_uint),
             ("ppEnabledLayerNames", POINTER(c_byte)),
             ("enabledExtensionCount", c_uint),
             ("ppEnabledExtensionNames", POINTER(c_byte)),
             ("pEnabledFeatures", VkPhysicalDeviceFeatures)
    ]

class VkExtensionProperties(Structure):
    pass
VkExtensionProperties._fields_ = [
             ("extensionName", c_byte *256),
             ("specc_uintersion", c_uint)
    ]

class VkLayerProperties(Structure):
    pass
VkLayerProperties._fields_ = [
             ("layerName", c_byte *256),
             ("specc_uintersion", c_uint),
             ("implementationc_uintersion", c_uint),
             ("description", c_byte *256)
    ]

class VkSubmitInfo(Structure):
    pass
VkSubmitInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("waitSemaphoreCount", c_uint),
             ("pWaitSemaphores", POINTER(VkSemaphore_T)),
             ("pWaitDstStageMask", POINTER(c_uint)),
             ("commandBufferCount", c_uint),
             ("pCommandBuffers", POINTER(VkCommandBuffer_T)),
             ("signalSemaphoreCount", c_uint),
             ("pSignalSemaphores", POINTER(VkSemaphore_T))
    ]

class VkMappedMemoryRange(Structure):
    pass
VkMappedMemoryRange._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memory", VkDeviceMemory_T),
             ("offset", c_ulong),
             ("size", c_ulong)
    ]

class VkMemoryAllocateInfo(Structure):
    pass
VkMemoryAllocateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("allocationSize", c_ulong),
             ("memoryTypeIndex", c_uint)
    ]

class VkMemoryRequirements(Structure):
    pass
VkMemoryRequirements._fields_ = [
             ("size", c_ulong),
             ("alignment", c_ulong),
             ("memoryTypeBits", c_uint)
    ]

class VkSparseMemoryBind(Structure):
    pass
VkSparseMemoryBind._fields_ = [
             ("resourceOffset", c_ulong),
             ("size", c_ulong),
             ("memory", VkDeviceMemory_T),
             ("memoryOffset", c_ulong),
             ("flags", c_uint)
    ]

class VkSparseBufferMemoryBindInfo(Structure):
    pass
VkSparseBufferMemoryBindInfo._fields_ = [
             ("buffer", VkBuffer_T),
             ("bindCount", c_uint),
             ("pBinds", VkSparseMemoryBind)
    ]

class VkSparseImageOpaqueMemoryBindInfo(Structure):
    pass
VkSparseImageOpaqueMemoryBindInfo._fields_ = [
             ("image", VkImage_T),
             ("bindCount", c_uint),
             ("pBinds", VkSparseMemoryBind)
    ]

class VkImageSubresource(Structure):
    pass
VkImageSubresource._fields_ = [
             ("aspectMask", c_uint),
             ("mipLevel", c_uint),
             ("arrayLayer", c_uint)
    ]

class VkSparseImageMemoryBind(Structure):
    pass
VkSparseImageMemoryBind._fields_ = [
             ("subresource", VkImageSubresource),
             ("offset", VkOffset3D),
             ("extent", VkExtent3D),
             ("memory", VkDeviceMemory_T),
             ("memoryOffset", c_ulong),
             ("flags", c_uint)
    ]

class VkSparseImageMemoryBindInfo(Structure):
    pass
VkSparseImageMemoryBindInfo._fields_ = [
             ("image", VkImage_T),
             ("bindCount", c_uint),
             ("pBinds", VkSparseImageMemoryBind)
    ]

class VkBindSparseInfo(Structure):
    pass
VkBindSparseInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("waitSemaphoreCount", c_uint),
             ("pWaitSemaphores", POINTER(VkSemaphore_T)),
             ("bufferBindCount", c_uint),
             ("pBufferBinds", VkSparseBufferMemoryBindInfo),
             ("imageOpaqueBindCount", c_uint),
             ("pImageOpaqueBinds", VkSparseImageOpaqueMemoryBindInfo),
             ("imageBindCount", c_uint),
             ("pImageBinds", VkSparseImageMemoryBindInfo),
             ("signalSemaphoreCount", c_uint),
             ("pSignalSemaphores", POINTER(VkSemaphore_T))
    ]

class VkSparseImageFormatProperties(Structure):
    pass
VkSparseImageFormatProperties._fields_ = [
             ("aspectMask", c_uint),
             ("imageGranularity", VkExtent3D),
             ("flags", c_uint)
    ]

class VkSparseImageMemoryRequirements(Structure):
    pass
VkSparseImageMemoryRequirements._fields_ = [
             ("formatProperties", VkSparseImageFormatProperties),
             ("imageMipTailFirstLod", c_uint),
             ("imageMipTailSize", c_ulong),
             ("imageMipTailOffset", c_ulong),
             ("imageMipTailStride", c_ulong)
    ]

class VkFenceCreateInfo(Structure):
    pass
VkFenceCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint)
    ]

class VkSemaphoreCreateInfo(Structure):
    pass
VkSemaphoreCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint)
    ]

class VkEventCreateInfo(Structure):
    pass
VkEventCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint)
    ]

class VkQueryPoolCreateInfo(Structure):
    pass
VkQueryPoolCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("queryType", c_int),
             ("queryCount", c_uint),
             ("pipelineStatistics", c_uint)
    ]

class VkBufferCreateInfo(Structure):
    pass
VkBufferCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("size", c_ulong),
             ("usage", c_uint),
             ("sharingMode", c_int),
             ("queueFamilyIndexCount", c_uint),
             ("pQueueFamilyIndices", POINTER(c_uint))
    ]

class VkBufferViewCreateInfo(Structure):
    pass
VkBufferViewCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("buffer", VkBuffer_T),
             ("format", c_int),
             ("offset", c_ulong),
             ("range", c_ulong)
    ]

class VkImageCreateInfo(Structure):
    pass
VkImageCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("imageType", c_int),
             ("format", c_int),
             ("extent", VkExtent3D),
             ("mipLevels", c_uint),
             ("arrayLayers", c_uint),
             ("samples", c_int),
             ("tiling", c_int),
             ("usage", c_uint),
             ("sharingMode", c_int),
             ("queueFamilyIndexCount", c_uint),
             ("pQueueFamilyIndices", POINTER(c_uint)),
             ("initialLayout", c_int)
    ]

class VkSubresourceLayout(Structure):
    pass
VkSubresourceLayout._fields_ = [
             ("offset", c_ulong),
             ("size", c_ulong),
             ("rowPitch", c_ulong),
             ("arrayPitch", c_ulong),
             ("depthPitch", c_ulong)
    ]

class VkComponentMapping(Structure):
    pass
VkComponentMapping._fields_ = [
             ("r", c_int),
             ("g", c_int),
             ("b", c_int),
             ("a", c_int)
    ]

class VkImageViewCreateInfo(Structure):
    pass
VkImageViewCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("image", VkImage_T),
             ("viewType", c_int),
             ("format", c_int),
             ("components", VkComponentMapping),
             ("subresourceRange", VkImageSubresourceRange)
    ]

class VkShaderModuleCreateInfo(Structure):
    pass
VkShaderModuleCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("codeSize", c_ulong),
             ("pCode", POINTER(c_uint))
    ]

class VkPipelineCacheCreateInfo(Structure):
    pass
VkPipelineCacheCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("initialDataSize", c_ulong),
             ("pInitialData", c_void_p)
    ]

class VkSpecializationMapEntry(Structure):
    pass
VkSpecializationMapEntry._fields_ = [
             ("constantID", c_uint),
             ("offset", c_uint),
             ("size", c_ulong)
    ]

class VkSpecializationInfo(Structure):
    pass
VkSpecializationInfo._fields_ = [
             ("mapEntryCount", c_uint),
             ("pMapEntries", VkSpecializationMapEntry),
             ("dataSize", c_ulong),
             ("pData", c_void_p)
    ]

class VkPipelineShaderStageCreateInfo(Structure):
    pass
VkPipelineShaderStageCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("stage", c_int),
             ("module", VkShaderModule_T),
             ("pName", c_char_p),
             ("pSpecializationInfo", VkSpecializationInfo)
    ]

class VkComputePipelineCreateInfo(Structure):
    pass
VkComputePipelineCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("stage", VkPipelineShaderStageCreateInfo),
             ("layout", VkPipelineLayout_T),
             ("basePipelineHandle", VkPipeline_T),
             ("basePipelineIndex", c_int)
    ]

class VkVertexInputBindingDescription(Structure):
    pass
VkVertexInputBindingDescription._fields_ = [
             ("binding", c_uint),
             ("stride", c_uint),
             ("inputRate", c_int)
    ]

class VkVertexInputAttributeDescription(Structure):
    pass
VkVertexInputAttributeDescription._fields_ = [
             ("location", c_uint),
             ("binding", c_uint),
             ("format", c_int),
             ("offset", c_uint)
    ]

class VkPipelineVertexInputStateCreateInfo(Structure):
    pass
VkPipelineVertexInputStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("vertexBindingDescriptionCount", c_uint),
             ("pVkVertexInputBindingDescriptionertexBindingDescriptions", VkVertexInputBindingDescription),
             ("vertexAttributeDescriptionCount", c_uint),
             ("pVkVertexInputAttributeDescriptionertexAttributeDescriptions", VkVertexInputAttributeDescription)
    ]

class VkPipelineInputAssemblyStateCreateInfo(Structure):
    pass
VkPipelineInputAssemblyStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("topology", c_int),
             ("primitiveRestartEnable", c_uint)
    ]

class VkPipelineTessellationStateCreateInfo(Structure):
    pass
VkPipelineTessellationStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("patchControlPoints", c_uint)
    ]

class VkViewport(Structure):
    pass
VkViewport._fields_ = [
             ("x", c_float),
             ("y", c_float),
             ("width", c_float),
             ("height", c_float),
             ("minDepth", c_float),
             ("maxDepth", c_float)
    ]

class VkPipelineViewportStateCreateInfo(Structure):
    pass
VkPipelineViewportStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("viewportCount", c_uint),
             ("pVkViewportiewports", VkViewport),
             ("scissorCount", c_uint),
             ("pScissors", VkRect2D)
    ]

class VkPipelineRasterizationStateCreateInfo(Structure):
    pass
VkPipelineRasterizationStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("depthClampEnable", c_uint),
             ("rasterizerDiscardEnable", c_uint),
             ("polygonMode", c_int),
             ("cullMode", c_uint),
             ("frontFace", c_int),
             ("depthBiasEnable", c_uint),
             ("depthBiasConstantFactor", c_float),
             ("depthBiasClamp", c_float),
             ("depthBiasSlopeFactor", c_float),
             ("lineWidth", c_float)
    ]

class VkPipelineMultisampleStateCreateInfo(Structure):
    pass
VkPipelineMultisampleStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("rasterizationSamples", c_int),
             ("sampleShadingEnable", c_uint),
             ("minSampleShading", c_float),
             ("pSampleMask", POINTER(c_uint)),
             ("alphaToCoverageEnable", c_uint),
             ("alphaToOneEnable", c_uint)
    ]

class VkStencilOpState(Structure):
    pass
VkStencilOpState._fields_ = [
             ("failOp", c_int),
             ("passOp", c_int),
             ("depthFailOp", c_int),
             ("compareOp", c_int),
             ("compareMask", c_uint),
             ("writeMask", c_uint),
             ("reference", c_uint)
    ]

class VkPipelineDepthStencilStateCreateInfo(Structure):
    pass
VkPipelineDepthStencilStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("depthTestEnable", c_uint),
             ("depthWriteEnable", c_uint),
             ("depthCompareOp", c_int),
             ("depthBoundsTestEnable", c_uint),
             ("stencilTestEnable", c_uint),
             ("front", VkStencilOpState),
             ("back", VkStencilOpState),
             ("minDepthBounds", c_float),
             ("maxDepthBounds", c_float)
    ]

class VkPipelineColorBlendAttachmentState(Structure):
    pass
VkPipelineColorBlendAttachmentState._fields_ = [
             ("blendEnable", c_uint),
             ("srcColorBlendFactor", c_int),
             ("dstColorBlendFactor", c_int),
             ("colorBlendOp", c_int),
             ("srcAlphaBlendFactor", c_int),
             ("dstAlphaBlendFactor", c_int),
             ("alphaBlendOp", c_int),
             ("colorWriteMask", c_uint)
    ]

class VkPipelineColorBlendStateCreateInfo(Structure):
    pass
VkPipelineColorBlendStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("logicOpEnable", c_uint),
             ("logicOp", c_int),
             ("attachmentCount", c_uint),
             ("pAttachments", VkPipelineColorBlendAttachmentState),
             ("blendConstants", c_float *4)
    ]

class VkPipelineDynamicStateCreateInfo(Structure):
    pass
VkPipelineDynamicStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("dynamicStateCount", c_uint),
             ("pDynamicStates", POINTER(c_int))
    ]

class VkGraphicsPipelineCreateInfo(Structure):
    pass
VkGraphicsPipelineCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("stageCount", c_uint),
             ("pStages", VkPipelineShaderStageCreateInfo),
             ("pVkPipelineVertexInputStateCreateInfoertexInputState", VkPipelineVertexInputStateCreateInfo),
             ("pInputAssemblyState", VkPipelineInputAssemblyStateCreateInfo),
             ("pTessellationState", VkPipelineTessellationStateCreateInfo),
             ("pVkPipelineViewportStateCreateInfoiewportState", VkPipelineViewportStateCreateInfo),
             ("pRasterizationState", VkPipelineRasterizationStateCreateInfo),
             ("pMultisampleState", VkPipelineMultisampleStateCreateInfo),
             ("pDepthStencilState", VkPipelineDepthStencilStateCreateInfo),
             ("pColorBlendState", VkPipelineColorBlendStateCreateInfo),
             ("pDynamicState", VkPipelineDynamicStateCreateInfo),
             ("layout", VkPipelineLayout_T),
             ("renderPass", VkRenderPass_T),
             ("subpass", c_uint),
             ("basePipelineHandle", VkPipeline_T),
             ("basePipelineIndex", c_int)
    ]

class VkPushConstantRange(Structure):
    pass
VkPushConstantRange._fields_ = [
             ("stageFlags", c_uint),
             ("offset", c_uint),
             ("size", c_uint)
    ]

class VkPipelineLayoutCreateInfo(Structure):
    pass
VkPipelineLayoutCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("setLayoutCount", c_uint),
             ("pSetLayouts", POINTER(VkDescriptorSetLayout_T)),
             ("pushConstantRangeCount", c_uint),
             ("pPushConstantRanges", VkPushConstantRange)
    ]

class VkSamplerCreateInfo(Structure):
    pass
VkSamplerCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("magFilter", c_int),
             ("minFilter", c_int),
             ("mipmapMode", c_int),
             ("addressModeU", c_int),
             ("addressModec_int", c_int),
             ("addressModeW", c_int),
             ("mipLodBias", c_float),
             ("anisotropyEnable", c_uint),
             ("maxAnisotropy", c_float),
             ("compareEnable", c_uint),
             ("compareOp", c_int),
             ("minLod", c_float),
             ("maxLod", c_float),
             ("borderColor", c_int),
             ("unnormalizedCoordinates", c_uint)
    ]

class VkCopyDescriptorSet(Structure):
    pass
VkCopyDescriptorSet._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcSet", VkDescriptorSet_T),
             ("srcBinding", c_uint),
             ("srcArrayElement", c_uint),
             ("dstSet", VkDescriptorSet_T),
             ("dstBinding", c_uint),
             ("dstArrayElement", c_uint),
             ("descriptorCount", c_uint)
    ]

class VkDescriptorBufferInfo(Structure):
    pass
VkDescriptorBufferInfo._fields_ = [
             ("buffer", VkBuffer_T),
             ("offset", c_ulong),
             ("range", c_ulong)
    ]

class VkDescriptorImageInfo(Structure):
    pass
VkDescriptorImageInfo._fields_ = [
             ("sampler", VkSampler_T),
             ("imageVkImageView_Tiew", VkImageView_T),
             ("imageLayout", c_int)
    ]

class VkDescriptorPoolSize(Structure):
    pass
VkDescriptorPoolSize._fields_ = [
             ("type", c_int),
             ("descriptorCount", c_uint)
    ]

class VkDescriptorPoolCreateInfo(Structure):
    pass
VkDescriptorPoolCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("maxSets", c_uint),
             ("poolSizeCount", c_uint),
             ("pPoolSizes", VkDescriptorPoolSize)
    ]

class VkDescriptorSetAllocateInfo(Structure):
    pass
VkDescriptorSetAllocateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("descriptorPool", VkDescriptorPool_T),
             ("descriptorSetCount", c_uint),
             ("pSetLayouts", POINTER(VkDescriptorSetLayout_T))
    ]

class VkDescriptorSetLayoutBinding(Structure):
    pass
VkDescriptorSetLayoutBinding._fields_ = [
             ("binding", c_uint),
             ("descriptorType", c_int),
             ("descriptorCount", c_uint),
             ("stageFlags", c_uint),
             ("pImmutableSamplers", POINTER(VkSampler_T))
    ]

class VkDescriptorSetLayoutCreateInfo(Structure):
    pass
VkDescriptorSetLayoutCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("bindingCount", c_uint),
             ("pBindings", VkDescriptorSetLayoutBinding)
    ]

class VkWriteDescriptorSet(Structure):
    pass
VkWriteDescriptorSet._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("dstSet", VkDescriptorSet_T),
             ("dstBinding", c_uint),
             ("dstArrayElement", c_uint),
             ("descriptorCount", c_uint),
             ("descriptorType", c_int),
             ("pImageInfo", VkDescriptorImageInfo),
             ("pBufferInfo", VkDescriptorBufferInfo),
             ("pTexelBufferPOINTER(VkBufferView_T)iew", POINTER(VkBufferView_T))
    ]

class VkAttachmentDescription(Structure):
    pass
VkAttachmentDescription._fields_ = [
             ("flags", c_uint),
             ("format", c_int),
             ("samples", c_int),
             ("loadOp", c_int),
             ("storeOp", c_int),
             ("stencilLoadOp", c_int),
             ("stencilStoreOp", c_int),
             ("initialLayout", c_int),
             ("finalLayout", c_int)
    ]

class VkAttachmentReference(Structure):
    pass
VkAttachmentReference._fields_ = [
             ("attachment", c_uint),
             ("layout", c_int)
    ]

class VkFramebufferCreateInfo(Structure):
    pass
VkFramebufferCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("renderPass", VkRenderPass_T),
             ("attachmentCount", c_uint),
             ("pAttachments", POINTER(VkImageView_T)),
             ("width", c_uint),
             ("height", c_uint),
             ("layers", c_uint)
    ]

class VkSubpassDescription(Structure):
    pass
VkSubpassDescription._fields_ = [
             ("flags", c_uint),
             ("pipelineBindPoint", c_int),
             ("inputAttachmentCount", c_uint),
             ("pInputAttachments", VkAttachmentReference),
             ("colorAttachmentCount", c_uint),
             ("pColorAttachments", VkAttachmentReference),
             ("pResolveAttachments", VkAttachmentReference),
             ("pDepthStencilAttachment", VkAttachmentReference),
             ("preserveAttachmentCount", c_uint),
             ("pPreserveAttachments", POINTER(c_uint))
    ]

class VkSubpassDependency(Structure):
    pass
VkSubpassDependency._fields_ = [
             ("srcSubpass", c_uint),
             ("dstSubpass", c_uint),
             ("srcStageMask", c_uint),
             ("dstStageMask", c_uint),
             ("srcAccessMask", c_uint),
             ("dstAccessMask", c_uint),
             ("dependencyFlags", c_uint)
    ]

class VkRenderPassCreateInfo(Structure):
    pass
VkRenderPassCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("attachmentCount", c_uint),
             ("pAttachments", VkAttachmentDescription),
             ("subpassCount", c_uint),
             ("pSubpasses", VkSubpassDescription),
             ("dependencyCount", c_uint),
             ("pDependencies", VkSubpassDependency)
    ]

class VkCommandPoolCreateInfo(Structure):
    pass
VkCommandPoolCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("queueFamilyIndex", c_uint)
    ]

class VkCommandBufferAllocateInfo(Structure):
    pass
VkCommandBufferAllocateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("commandPool", VkCommandPool_T),
             ("level", c_int),
             ("commandBufferCount", c_uint)
    ]

class VkCommandBufferInheritanceInfo(Structure):
    pass
VkCommandBufferInheritanceInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("renderPass", VkRenderPass_T),
             ("subpass", c_uint),
             ("framebuffer", VkFramebuffer_T),
             ("occlusionQueryEnable", c_uint),
             ("queryFlags", c_uint),
             ("pipelineStatistics", c_uint)
    ]

class VkCommandBufferBeginInfo(Structure):
    pass
VkCommandBufferBeginInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("pInheritanceInfo", VkCommandBufferInheritanceInfo)
    ]

class VkBufferCopy(Structure):
    pass
VkBufferCopy._fields_ = [
             ("srcOffset", c_ulong),
             ("dstOffset", c_ulong),
             ("size", c_ulong)
    ]

class VkImageSubresourceLayers(Structure):
    pass
VkImageSubresourceLayers._fields_ = [
             ("aspectMask", c_uint),
             ("mipLevel", c_uint),
             ("baseArrayLayer", c_uint),
             ("layerCount", c_uint)
    ]

class VkBufferImageCopy(Structure):
    pass
VkBufferImageCopy._fields_ = [
             ("bufferOffset", c_ulong),
             ("bufferRowLength", c_uint),
             ("bufferImageHeight", c_uint),
             ("imageSubresource", VkImageSubresourceLayers),
             ("imageOffset", VkOffset3D),
             ("imageExtent", VkExtent3D)
    ]

class VkClearColorValue(Structure):
    pass
VkClearColorValue._fields_ = [
             ("float32", c_float *4),
             ("int32", c_int *4),
             ("uint32", c_uint *4)
    ]

class VkClearDepthStencilValue(Structure):
    pass
VkClearDepthStencilValue._fields_ = [
             ("depth", c_float),
             ("stencil", c_uint)
    ]

class VkClearValue(Structure):
    pass
VkClearValue._fields_ = [
             ("color", VkClearColorValue),
             ("depthStencil", VkClearDepthStencilValue)
    ]

class VkClearAttachment(Structure):
    pass
VkClearAttachment._fields_ = [
             ("aspectMask", c_uint),
             ("colorAttachment", c_uint),
             ("clearVkClearValuealue", VkClearValue)
    ]

class VkClearRect(Structure):
    pass
VkClearRect._fields_ = [
             ("rect", VkRect2D),
             ("baseArrayLayer", c_uint),
             ("layerCount", c_uint)
    ]

class VkImageBlit(Structure):
    pass
VkImageBlit._fields_ = [
             ("srcSubresource", VkImageSubresourceLayers),
             ("srcOffsets", VkOffset3D *2),
             ("dstSubresource", VkImageSubresourceLayers),
             ("dstOffsets", VkOffset3D *2)
    ]

class VkImageCopy(Structure):
    pass
VkImageCopy._fields_ = [
             ("srcSubresource", VkImageSubresourceLayers),
             ("srcOffset", VkOffset3D),
             ("dstSubresource", VkImageSubresourceLayers),
             ("dstOffset", VkOffset3D),
             ("extent", VkExtent3D)
    ]

class VkImageResolve(Structure):
    pass
VkImageResolve._fields_ = [
             ("srcSubresource", VkImageSubresourceLayers),
             ("srcOffset", VkOffset3D),
             ("dstSubresource", VkImageSubresourceLayers),
             ("dstOffset", VkOffset3D),
             ("extent", VkExtent3D)
    ]

class VkRenderPassBeginInfo(Structure):
    pass
VkRenderPassBeginInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("renderPass", VkRenderPass_T),
             ("framebuffer", VkFramebuffer_T),
             ("renderArea", VkRect2D),
             ("clearc_uintalueCount", c_uint),
             ("pClearVkClearValuealues", VkClearValue)
    ]

class VkPhysicalDeviceSubgroupProperties(Structure):
    pass
VkPhysicalDeviceSubgroupProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("subgroupSize", c_uint),
             ("supportedStages", c_uint),
             ("supportedOperations", c_uint),
             ("quadOperationsInAllStages", c_uint)
    ]

class VkBindBufferMemoryInfo(Structure):
    pass
VkBindBufferMemoryInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("buffer", VkBuffer_T),
             ("memory", VkDeviceMemory_T),
             ("memoryOffset", c_ulong)
    ]

class VkBindImageMemoryInfo(Structure):
    pass
VkBindImageMemoryInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("image", VkImage_T),
             ("memory", VkDeviceMemory_T),
             ("memoryOffset", c_ulong)
    ]

class VkPhysicalDevice16BitStorageFeatures(Structure):
    pass
VkPhysicalDevice16BitStorageFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("storageBuffer16BitAccess", c_uint),
             ("uniformAndStorageBuffer16BitAccess", c_uint),
             ("storagePushConstant16", c_uint),
             ("storageInputOutput16", c_uint)
    ]

class VkMemoryDedicatedRequirements(Structure):
    pass
VkMemoryDedicatedRequirements._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("prefersDedicatedAllocation", c_uint),
             ("requiresDedicatedAllocation", c_uint)
    ]

class VkMemoryDedicatedAllocateInfo(Structure):
    pass
VkMemoryDedicatedAllocateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("image", VkImage_T),
             ("buffer", VkBuffer_T)
    ]

class VkMemoryAllocateFlagsInfo(Structure):
    pass
VkMemoryAllocateFlagsInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("deviceMask", c_uint)
    ]

class VkDeviceGroupRenderPassBeginInfo(Structure):
    pass
VkDeviceGroupRenderPassBeginInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceMask", c_uint),
             ("deviceRenderAreaCount", c_uint),
             ("pDeviceRenderAreas", VkRect2D)
    ]

class VkDeviceGroupCommandBufferBeginInfo(Structure):
    pass
VkDeviceGroupCommandBufferBeginInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceMask", c_uint)
    ]

class VkDeviceGroupSubmitInfo(Structure):
    pass
VkDeviceGroupSubmitInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("waitSemaphoreCount", c_uint),
             ("pWaitSemaphoreDeviceIndices", POINTER(c_uint)),
             ("commandBufferCount", c_uint),
             ("pCommandBufferDeviceMasks", POINTER(c_uint)),
             ("signalSemaphoreCount", c_uint),
             ("pSignalSemaphoreDeviceIndices", POINTER(c_uint))
    ]

class VkDeviceGroupBindSparseInfo(Structure):
    pass
VkDeviceGroupBindSparseInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("resourceDeviceIndex", c_uint),
             ("memoryDeviceIndex", c_uint)
    ]

class VkBindBufferMemoryDeviceGroupInfo(Structure):
    pass
VkBindBufferMemoryDeviceGroupInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceIndexCount", c_uint),
             ("pDeviceIndices", POINTER(c_uint))
    ]

class VkBindImageMemoryDeviceGroupInfo(Structure):
    pass
VkBindImageMemoryDeviceGroupInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceIndexCount", c_uint),
             ("pDeviceIndices", POINTER(c_uint)),
             ("splitInstanceBindRegionCount", c_uint),
             ("pSplitInstanceBindRegions", VkRect2D)
    ]

class VkPhysicalDeviceGroupProperties(Structure):
    pass
VkPhysicalDeviceGroupProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("physicalDeviceCount", c_uint),
             ("physicalDevices", VkPhysicalDevice_T *32),
             ("subsetAllocation", c_uint)
    ]

class VkDeviceGroupDeviceCreateInfo(Structure):
    pass
VkDeviceGroupDeviceCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("physicalDeviceCount", c_uint),
             ("pPhysicalDevices", POINTER(VkPhysicalDevice_T))
    ]

class VkBufferMemoryRequirementsInfo2(Structure):
    pass
VkBufferMemoryRequirementsInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("buffer", VkBuffer_T)
    ]

class VkImageMemoryRequirementsInfo2(Structure):
    pass
VkImageMemoryRequirementsInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("image", VkImage_T)
    ]

class VkImageSparseMemoryRequirementsInfo2(Structure):
    pass
VkImageSparseMemoryRequirementsInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("image", VkImage_T)
    ]

class VkMemoryRequirements2(Structure):
    pass
VkMemoryRequirements2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memoryRequirements", VkMemoryRequirements)
    ]

class VkSparseImageMemoryRequirements2(Structure):
    pass
VkSparseImageMemoryRequirements2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memoryRequirements", VkSparseImageMemoryRequirements)
    ]

class VkPhysicalDeviceFeatures2(Structure):
    pass
VkPhysicalDeviceFeatures2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("features", VkPhysicalDeviceFeatures)
    ]

class VkPhysicalDeviceProperties2(Structure):
    pass
VkPhysicalDeviceProperties2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("properties", VkPhysicalDeviceProperties)
    ]

class VkFormatProperties2(Structure):
    pass
VkFormatProperties2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("formatProperties", VkFormatProperties)
    ]

class VkImageFormatProperties2(Structure):
    pass
VkImageFormatProperties2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("imageFormatProperties", VkImageFormatProperties)
    ]

class VkPhysicalDeviceImageFormatInfo2(Structure):
    pass
VkPhysicalDeviceImageFormatInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("format", c_int),
             ("type", c_int),
             ("tiling", c_int),
             ("usage", c_uint),
             ("flags", c_uint)
    ]

class VkQueueFamilyProperties2(Structure):
    pass
VkQueueFamilyProperties2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("queueFamilyProperties", VkQueueFamilyProperties)
    ]

class VkPhysicalDeviceMemoryProperties2(Structure):
    pass
VkPhysicalDeviceMemoryProperties2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memoryProperties", VkPhysicalDeviceMemoryProperties)
    ]

class VkSparseImageFormatProperties2(Structure):
    pass
VkSparseImageFormatProperties2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("properties", VkSparseImageFormatProperties)
    ]

class VkPhysicalDeviceSparseImageFormatInfo2(Structure):
    pass
VkPhysicalDeviceSparseImageFormatInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("format", c_int),
             ("type", c_int),
             ("samples", c_int),
             ("usage", c_uint),
             ("tiling", c_int)
    ]

class VkPhysicalDevicePointClippingProperties(Structure):
    pass
VkPhysicalDevicePointClippingProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pointClippingBehavior", c_int)
    ]

class VkInputAttachmentAspectReference(Structure):
    pass
VkInputAttachmentAspectReference._fields_ = [
             ("subpass", c_uint),
             ("inputAttachmentIndex", c_uint),
             ("aspectMask", c_uint)
    ]

class VkRenderPassInputAttachmentAspectCreateInfo(Structure):
    pass
VkRenderPassInputAttachmentAspectCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("aspectReferenceCount", c_uint),
             ("pAspectReferences", VkInputAttachmentAspectReference)
    ]

class VkImageViewUsageCreateInfo(Structure):
    pass
VkImageViewUsageCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("usage", c_uint)
    ]

class VkPipelineTessellationDomainOriginStateCreateInfo(Structure):
    pass
VkPipelineTessellationDomainOriginStateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("domainOrigin", c_int)
    ]

class VkRenderPassMultiviewCreateInfo(Structure):
    pass
VkRenderPassMultiviewCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("subpassCount", c_uint),
             ("pPOINTER(c_uint)iewMasks", POINTER(c_uint)),
             ("dependencyCount", c_uint),
             ("pPOINTER(c_int)iewOffsets", POINTER(c_int)),
             ("correlationMaskCount", c_uint),
             ("pCorrelationMasks", POINTER(c_uint))
    ]

class VkPhysicalDeviceMultiviewFeatures(Structure):
    pass
VkPhysicalDeviceMultiviewFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("multiview", c_uint),
             ("multiviewGeometryShader", c_uint),
             ("multiviewTessellationShader", c_uint)
    ]

class VkPhysicalDeviceMultiviewProperties(Structure):
    pass
VkPhysicalDeviceMultiviewProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxMultiviewc_uintiewCount", c_uint),
             ("maxMultiviewInstanceIndex", c_uint)
    ]

class VkPhysicalDeviceVariablePointersFeatures(Structure):
    pass
VkPhysicalDeviceVariablePointersFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("variablePointersStorageBuffer", c_uint),
             ("variablePointers", c_uint)
    ]

class VkPhysicalDeviceProtectedMemoryFeatures(Structure):
    pass
VkPhysicalDeviceProtectedMemoryFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("protectedMemory", c_uint)
    ]

class VkPhysicalDeviceProtectedMemoryProperties(Structure):
    pass
VkPhysicalDeviceProtectedMemoryProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("protectedNoFault", c_uint)
    ]

class VkDeviceQueueInfo2(Structure):
    pass
VkDeviceQueueInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("queueFamilyIndex", c_uint),
             ("queueIndex", c_uint)
    ]

class VkProtectedSubmitInfo(Structure):
    pass
VkProtectedSubmitInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("protectedSubmit", c_uint)
    ]

class VkSamplerYcbcrConversionCreateInfo(Structure):
    pass
VkSamplerYcbcrConversionCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("format", c_int),
             ("ycbcrModel", c_int),
             ("ycbcrRange", c_int),
             ("components", VkComponentMapping),
             ("xChromaOffset", c_int),
             ("yChromaOffset", c_int),
             ("chromaFilter", c_int),
             ("forceExplicitReconstruction", c_uint)
    ]

class VkSamplerYcbcrConversionInfo(Structure):
    pass
VkSamplerYcbcrConversionInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("conversion", VkSamplerYcbcrConversion_T)
    ]

class VkBindImagePlaneMemoryInfo(Structure):
    pass
VkBindImagePlaneMemoryInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("planeAspect", c_int)
    ]

class VkImagePlaneMemoryRequirementsInfo(Structure):
    pass
VkImagePlaneMemoryRequirementsInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("planeAspect", c_int)
    ]

class VkPhysicalDeviceSamplerYcbcrConversionFeatures(Structure):
    pass
VkPhysicalDeviceSamplerYcbcrConversionFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("samplerYcbcrConversion", c_uint)
    ]

class VkSamplerYcbcrConversionImageFormatProperties(Structure):
    pass
VkSamplerYcbcrConversionImageFormatProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("combinedImageSamplerDescriptorCount", c_uint)
    ]

class VkDescriptorUpdateTemplateEntry(Structure):
    pass
VkDescriptorUpdateTemplateEntry._fields_ = [
             ("dstBinding", c_uint),
             ("dstArrayElement", c_uint),
             ("descriptorCount", c_uint),
             ("descriptorType", c_int),
             ("offset", c_ulong),
             ("stride", c_ulong)
    ]

class VkDescriptorUpdateTemplateCreateInfo(Structure):
    pass
VkDescriptorUpdateTemplateCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("descriptorUpdateEntryCount", c_uint),
             ("pDescriptorUpdateEntries", VkDescriptorUpdateTemplateEntry),
             ("templateType", c_int),
             ("descriptorSetLayout", VkDescriptorSetLayout_T),
             ("pipelineBindPoint", c_int),
             ("pipelineLayout", VkPipelineLayout_T),
             ("set", c_uint)
    ]

class VkExternalMemoryProperties(Structure):
    pass
VkExternalMemoryProperties._fields_ = [
             ("externalMemoryFeatures", c_uint),
             ("exportFromImportedHandleTypes", c_uint),
             ("compatibleHandleTypes", c_uint)
    ]

class VkPhysicalDeviceExternalImageFormatInfo(Structure):
    pass
VkPhysicalDeviceExternalImageFormatInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleType", c_int)
    ]

class VkExternalImageFormatProperties(Structure):
    pass
VkExternalImageFormatProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("externalMemoryProperties", VkExternalMemoryProperties)
    ]

class VkPhysicalDeviceExternalBufferInfo(Structure):
    pass
VkPhysicalDeviceExternalBufferInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("usage", c_uint),
             ("handleType", c_int)
    ]

class VkExternalBufferProperties(Structure):
    pass
VkExternalBufferProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("externalMemoryProperties", VkExternalMemoryProperties)
    ]

class VkPhysicalDeviceIDProperties(Structure):
    pass
VkPhysicalDeviceIDProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceUUID", c_ubyte *16),
             ("driverUUID", c_ubyte *16),
             ("deviceLUID", c_ubyte *8),
             ("deviceNodeMask", c_uint),
             ("deviceLUIDc_uintalid", c_uint)
    ]

class VkExternalMemoryImageCreateInfo(Structure):
    pass
VkExternalMemoryImageCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleTypes", c_uint)
    ]

class VkExternalMemoryBufferCreateInfo(Structure):
    pass
VkExternalMemoryBufferCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleTypes", c_uint)
    ]

class VkExportMemoryAllocateInfo(Structure):
    pass
VkExportMemoryAllocateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleTypes", c_uint)
    ]

class VkPhysicalDeviceExternalFenceInfo(Structure):
    pass
VkPhysicalDeviceExternalFenceInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleType", c_int)
    ]

class VkExternalFenceProperties(Structure):
    pass
VkExternalFenceProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("exportFromImportedHandleTypes", c_uint),
             ("compatibleHandleTypes", c_uint),
             ("externalFenceFeatures", c_uint)
    ]

class VkExportFenceCreateInfo(Structure):
    pass
VkExportFenceCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleTypes", c_uint)
    ]

class VkExportSemaphoreCreateInfo(Structure):
    pass
VkExportSemaphoreCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleTypes", c_uint)
    ]

class VkPhysicalDeviceExternalSemaphoreInfo(Structure):
    pass
VkPhysicalDeviceExternalSemaphoreInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleType", c_int)
    ]

class VkExternalSemaphoreProperties(Structure):
    pass
VkExternalSemaphoreProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("exportFromImportedHandleTypes", c_uint),
             ("compatibleHandleTypes", c_uint),
             ("externalSemaphoreFeatures", c_uint)
    ]

class VkPhysicalDeviceMaintenance3Properties(Structure):
    pass
VkPhysicalDeviceMaintenance3Properties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxPerSetDescriptors", c_uint),
             ("maxMemoryAllocationSize", c_ulong)
    ]

class VkDescriptorSetLayoutSupport(Structure):
    pass
VkDescriptorSetLayoutSupport._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("supported", c_uint)
    ]

class VkPhysicalDeviceShaderDrawParametersFeatures(Structure):
    pass
VkPhysicalDeviceShaderDrawParametersFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderDrawParameters", c_uint)
    ]

class VkPhysicalDeviceVulkan11Features(Structure):
    pass
VkPhysicalDeviceVulkan11Features._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("storageBuffer16BitAccess", c_uint),
             ("uniformAndStorageBuffer16BitAccess", c_uint),
             ("storagePushConstant16", c_uint),
             ("storageInputOutput16", c_uint),
             ("multiview", c_uint),
             ("multiviewGeometryShader", c_uint),
             ("multiviewTessellationShader", c_uint),
             ("variablePointersStorageBuffer", c_uint),
             ("variablePointers", c_uint),
             ("protectedMemory", c_uint),
             ("samplerYcbcrConversion", c_uint),
             ("shaderDrawParameters", c_uint)
    ]

class VkPhysicalDeviceVulkan11Properties(Structure):
    pass
VkPhysicalDeviceVulkan11Properties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceUUID", c_ubyte *16),
             ("driverUUID", c_ubyte *16),
             ("deviceLUID", c_ubyte *8),
             ("deviceNodeMask", c_uint),
             ("deviceLUIDc_uintalid", c_uint),
             ("subgroupSize", c_uint),
             ("subgroupSupportedStages", c_uint),
             ("subgroupSupportedOperations", c_uint),
             ("subgroupQuadOperationsInAllStages", c_uint),
             ("pointClippingBehavior", c_int),
             ("maxMultiviewc_uintiewCount", c_uint),
             ("maxMultiviewInstanceIndex", c_uint),
             ("protectedNoFault", c_uint),
             ("maxPerSetDescriptors", c_uint),
             ("maxMemoryAllocationSize", c_ulong)
    ]

class VkPhysicalDeviceVulkan12Features(Structure):
    pass
VkPhysicalDeviceVulkan12Features._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("samplerMirrorClampToEdge", c_uint),
             ("drawIndirectCount", c_uint),
             ("storageBuffer8BitAccess", c_uint),
             ("uniformAndStorageBuffer8BitAccess", c_uint),
             ("storagePushConstant8", c_uint),
             ("shaderBufferInt64Atomics", c_uint),
             ("shaderSharedInt64Atomics", c_uint),
             ("shaderFloat16", c_uint),
             ("shaderInt8", c_uint),
             ("descriptorIndexing", c_uint),
             ("shaderInputAttachmentArrayDynamicIndexing", c_uint),
             ("shaderUniformTexelBufferArrayDynamicIndexing", c_uint),
             ("shaderStorageTexelBufferArrayDynamicIndexing", c_uint),
             ("shaderUniformBufferArrayNonUniformIndexing", c_uint),
             ("shaderSampledImageArrayNonUniformIndexing", c_uint),
             ("shaderStorageBufferArrayNonUniformIndexing", c_uint),
             ("shaderStorageImageArrayNonUniformIndexing", c_uint),
             ("shaderInputAttachmentArrayNonUniformIndexing", c_uint),
             ("shaderUniformTexelBufferArrayNonUniformIndexing", c_uint),
             ("shaderStorageTexelBufferArrayNonUniformIndexing", c_uint),
             ("descriptorBindingUniformBufferUpdateAfterBind", c_uint),
             ("descriptorBindingSampledImageUpdateAfterBind", c_uint),
             ("descriptorBindingStorageImageUpdateAfterBind", c_uint),
             ("descriptorBindingStorageBufferUpdateAfterBind", c_uint),
             ("descriptorBindingUniformTexelBufferUpdateAfterBind", c_uint),
             ("descriptorBindingStorageTexelBufferUpdateAfterBind", c_uint),
             ("descriptorBindingUpdateUnusedWhilePending", c_uint),
             ("descriptorBindingPartiallyBound", c_uint),
             ("descriptorBindingc_uintariableDescriptorCount", c_uint),
             ("runtimeDescriptorArray", c_uint),
             ("samplerFilterMinmax", c_uint),
             ("scalarBlockLayout", c_uint),
             ("imagelessFramebuffer", c_uint),
             ("uniformBufferStandardLayout", c_uint),
             ("shaderSubgroupExtendedTypes", c_uint),
             ("separateDepthStencilLayouts", c_uint),
             ("hostQueryReset", c_uint),
             ("timelineSemaphore", c_uint),
             ("bufferDeviceAddress", c_uint),
             ("bufferDeviceAddressCaptureReplay", c_uint),
             ("bufferDeviceAddressMultiDevice", c_uint),
             ("vulkanMemoryModel", c_uint),
             ("vulkanMemoryModelDeviceScope", c_uint),
             ("vulkanMemoryModelAvailabilityc_uintisibilityChains", c_uint),
             ("shaderOutputc_uintiewportIndex", c_uint),
             ("shaderOutputLayer", c_uint),
             ("subgroupBroadcastDynamicId", c_uint)
    ]

class VkConformanceVersion(Structure):
    pass
VkConformanceVersion._fields_ = [
             ("major", c_ubyte),
             ("minor", c_ubyte),
             ("subminor", c_ubyte),
             ("patch", c_ubyte)
    ]

class VkPhysicalDeviceVulkan12Properties(Structure):
    pass
VkPhysicalDeviceVulkan12Properties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("driverID", c_int),
             ("driverName", c_byte *256),
             ("driverInfo", c_byte *256),
             ("conformanceVkConformanceVersionersion", VkConformanceVersion),
             ("denormBehaviorIndependence", c_int),
             ("roundingModeIndependence", c_int),
             ("shaderSignedZeroInfNanPreserveFloat16", c_uint),
             ("shaderSignedZeroInfNanPreserveFloat32", c_uint),
             ("shaderSignedZeroInfNanPreserveFloat64", c_uint),
             ("shaderDenormPreserveFloat16", c_uint),
             ("shaderDenormPreserveFloat32", c_uint),
             ("shaderDenormPreserveFloat64", c_uint),
             ("shaderDenormFlushToZeroFloat16", c_uint),
             ("shaderDenormFlushToZeroFloat32", c_uint),
             ("shaderDenormFlushToZeroFloat64", c_uint),
             ("shaderRoundingModeRTEFloat16", c_uint),
             ("shaderRoundingModeRTEFloat32", c_uint),
             ("shaderRoundingModeRTEFloat64", c_uint),
             ("shaderRoundingModeRTZFloat16", c_uint),
             ("shaderRoundingModeRTZFloat32", c_uint),
             ("shaderRoundingModeRTZFloat64", c_uint),
             ("maxUpdateAfterBindDescriptorsInAllPools", c_uint),
             ("shaderUniformBufferArrayNonUniformIndexingNative", c_uint),
             ("shaderSampledImageArrayNonUniformIndexingNative", c_uint),
             ("shaderStorageBufferArrayNonUniformIndexingNative", c_uint),
             ("shaderStorageImageArrayNonUniformIndexingNative", c_uint),
             ("shaderInputAttachmentArrayNonUniformIndexingNative", c_uint),
             ("robustBufferAccessUpdateAfterBind", c_uint),
             ("quadDivergentImplicitLod", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindSamplers", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindUniformBuffers", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindStorageBuffers", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindSampledImages", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindStorageImages", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindInputAttachments", c_uint),
             ("maxPerStageUpdateAfterBindResources", c_uint),
             ("maxDescriptorSetUpdateAfterBindSamplers", c_uint),
             ("maxDescriptorSetUpdateAfterBindUniformBuffers", c_uint),
             ("maxDescriptorSetUpdateAfterBindUniformBuffersDynamic", c_uint),
             ("maxDescriptorSetUpdateAfterBindStorageBuffers", c_uint),
             ("maxDescriptorSetUpdateAfterBindStorageBuffersDynamic", c_uint),
             ("maxDescriptorSetUpdateAfterBindSampledImages", c_uint),
             ("maxDescriptorSetUpdateAfterBindStorageImages", c_uint),
             ("maxDescriptorSetUpdateAfterBindInputAttachments", c_uint),
             ("supportedDepthResolveModes", c_uint),
             ("supportedStencilResolveModes", c_uint),
             ("independentResolveNone", c_uint),
             ("independentResolve", c_uint),
             ("filterMinmaxSingleComponentFormats", c_uint),
             ("filterMinmaxImageComponentMapping", c_uint),
             ("maxTimelineSemaphorec_ulongalueDifference", c_ulong),
             ("framebufferIntegerColorSampleCounts", c_uint)
    ]

class VkImageFormatListCreateInfo(Structure):
    pass
VkImageFormatListCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("viewFormatCount", c_uint),
             ("pPOINTER(c_int)iewFormats", POINTER(c_int))
    ]

class VkAttachmentDescription2(Structure):
    pass
VkAttachmentDescription2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("format", c_int),
             ("samples", c_int),
             ("loadOp", c_int),
             ("storeOp", c_int),
             ("stencilLoadOp", c_int),
             ("stencilStoreOp", c_int),
             ("initialLayout", c_int),
             ("finalLayout", c_int)
    ]

class VkAttachmentReference2(Structure):
    pass
VkAttachmentReference2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("attachment", c_uint),
             ("layout", c_int),
             ("aspectMask", c_uint)
    ]

class VkSubpassDescription2(Structure):
    pass
VkSubpassDescription2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("pipelineBindPoint", c_int),
             ("viewMask", c_uint),
             ("inputAttachmentCount", c_uint),
             ("pInputAttachments", VkAttachmentReference2),
             ("colorAttachmentCount", c_uint),
             ("pColorAttachments", VkAttachmentReference2),
             ("pResolveAttachments", VkAttachmentReference2),
             ("pDepthStencilAttachment", VkAttachmentReference2),
             ("preserveAttachmentCount", c_uint),
             ("pPreserveAttachments", POINTER(c_uint))
    ]

class VkSubpassDependency2(Structure):
    pass
VkSubpassDependency2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcSubpass", c_uint),
             ("dstSubpass", c_uint),
             ("srcStageMask", c_uint),
             ("dstStageMask", c_uint),
             ("srcAccessMask", c_uint),
             ("dstAccessMask", c_uint),
             ("dependencyFlags", c_uint),
             ("viewOffset", c_int)
    ]

class VkRenderPassCreateInfo2(Structure):
    pass
VkRenderPassCreateInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("attachmentCount", c_uint),
             ("pAttachments", VkAttachmentDescription2),
             ("subpassCount", c_uint),
             ("pSubpasses", VkSubpassDescription2),
             ("dependencyCount", c_uint),
             ("pDependencies", VkSubpassDependency2),
             ("correlatedc_uintiewMaskCount", c_uint),
             ("pCorrelatedPOINTER(c_uint)iewMasks", POINTER(c_uint))
    ]

class VkSubpassBeginInfo(Structure):
    pass
VkSubpassBeginInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("contents", c_int)
    ]

class VkSubpassEndInfo(Structure):
    pass
VkSubpassEndInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p)
    ]

class VkPhysicalDevice8BitStorageFeatures(Structure):
    pass
VkPhysicalDevice8BitStorageFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("storageBuffer8BitAccess", c_uint),
             ("uniformAndStorageBuffer8BitAccess", c_uint),
             ("storagePushConstant8", c_uint)
    ]

class VkPhysicalDeviceDriverProperties(Structure):
    pass
VkPhysicalDeviceDriverProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("driverID", c_int),
             ("driverName", c_byte *256),
             ("driverInfo", c_byte *256),
             ("conformanceVkConformanceVersionersion", VkConformanceVersion)
    ]

class VkPhysicalDeviceShaderAtomicInt64Features(Structure):
    pass
VkPhysicalDeviceShaderAtomicInt64Features._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderBufferInt64Atomics", c_uint),
             ("shaderSharedInt64Atomics", c_uint)
    ]

class VkPhysicalDeviceShaderFloat16Int8Features(Structure):
    pass
VkPhysicalDeviceShaderFloat16Int8Features._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderFloat16", c_uint),
             ("shaderInt8", c_uint)
    ]

class VkPhysicalDeviceFloatControlsProperties(Structure):
    pass
VkPhysicalDeviceFloatControlsProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("denormBehaviorIndependence", c_int),
             ("roundingModeIndependence", c_int),
             ("shaderSignedZeroInfNanPreserveFloat16", c_uint),
             ("shaderSignedZeroInfNanPreserveFloat32", c_uint),
             ("shaderSignedZeroInfNanPreserveFloat64", c_uint),
             ("shaderDenormPreserveFloat16", c_uint),
             ("shaderDenormPreserveFloat32", c_uint),
             ("shaderDenormPreserveFloat64", c_uint),
             ("shaderDenormFlushToZeroFloat16", c_uint),
             ("shaderDenormFlushToZeroFloat32", c_uint),
             ("shaderDenormFlushToZeroFloat64", c_uint),
             ("shaderRoundingModeRTEFloat16", c_uint),
             ("shaderRoundingModeRTEFloat32", c_uint),
             ("shaderRoundingModeRTEFloat64", c_uint),
             ("shaderRoundingModeRTZFloat16", c_uint),
             ("shaderRoundingModeRTZFloat32", c_uint),
             ("shaderRoundingModeRTZFloat64", c_uint)
    ]

class VkDescriptorSetLayoutBindingFlagsCreateInfo(Structure):
    pass
VkDescriptorSetLayoutBindingFlagsCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("bindingCount", c_uint),
             ("pBindingFlags", POINTER(c_uint))
    ]

class VkPhysicalDeviceDescriptorIndexingFeatures(Structure):
    pass
VkPhysicalDeviceDescriptorIndexingFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderInputAttachmentArrayDynamicIndexing", c_uint),
             ("shaderUniformTexelBufferArrayDynamicIndexing", c_uint),
             ("shaderStorageTexelBufferArrayDynamicIndexing", c_uint),
             ("shaderUniformBufferArrayNonUniformIndexing", c_uint),
             ("shaderSampledImageArrayNonUniformIndexing", c_uint),
             ("shaderStorageBufferArrayNonUniformIndexing", c_uint),
             ("shaderStorageImageArrayNonUniformIndexing", c_uint),
             ("shaderInputAttachmentArrayNonUniformIndexing", c_uint),
             ("shaderUniformTexelBufferArrayNonUniformIndexing", c_uint),
             ("shaderStorageTexelBufferArrayNonUniformIndexing", c_uint),
             ("descriptorBindingUniformBufferUpdateAfterBind", c_uint),
             ("descriptorBindingSampledImageUpdateAfterBind", c_uint),
             ("descriptorBindingStorageImageUpdateAfterBind", c_uint),
             ("descriptorBindingStorageBufferUpdateAfterBind", c_uint),
             ("descriptorBindingUniformTexelBufferUpdateAfterBind", c_uint),
             ("descriptorBindingStorageTexelBufferUpdateAfterBind", c_uint),
             ("descriptorBindingUpdateUnusedWhilePending", c_uint),
             ("descriptorBindingPartiallyBound", c_uint),
             ("descriptorBindingc_uintariableDescriptorCount", c_uint),
             ("runtimeDescriptorArray", c_uint)
    ]

class VkPhysicalDeviceDescriptorIndexingProperties(Structure):
    pass
VkPhysicalDeviceDescriptorIndexingProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxUpdateAfterBindDescriptorsInAllPools", c_uint),
             ("shaderUniformBufferArrayNonUniformIndexingNative", c_uint),
             ("shaderSampledImageArrayNonUniformIndexingNative", c_uint),
             ("shaderStorageBufferArrayNonUniformIndexingNative", c_uint),
             ("shaderStorageImageArrayNonUniformIndexingNative", c_uint),
             ("shaderInputAttachmentArrayNonUniformIndexingNative", c_uint),
             ("robustBufferAccessUpdateAfterBind", c_uint),
             ("quadDivergentImplicitLod", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindSamplers", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindUniformBuffers", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindStorageBuffers", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindSampledImages", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindStorageImages", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindInputAttachments", c_uint),
             ("maxPerStageUpdateAfterBindResources", c_uint),
             ("maxDescriptorSetUpdateAfterBindSamplers", c_uint),
             ("maxDescriptorSetUpdateAfterBindUniformBuffers", c_uint),
             ("maxDescriptorSetUpdateAfterBindUniformBuffersDynamic", c_uint),
             ("maxDescriptorSetUpdateAfterBindStorageBuffers", c_uint),
             ("maxDescriptorSetUpdateAfterBindStorageBuffersDynamic", c_uint),
             ("maxDescriptorSetUpdateAfterBindSampledImages", c_uint),
             ("maxDescriptorSetUpdateAfterBindStorageImages", c_uint),
             ("maxDescriptorSetUpdateAfterBindInputAttachments", c_uint)
    ]

class VkDescriptorSetVariableDescriptorCountAllocateInfo(Structure):
    pass
VkDescriptorSetVariableDescriptorCountAllocateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("descriptorSetCount", c_uint),
             ("pDescriptorCounts", POINTER(c_uint))
    ]

class VkDescriptorSetVariableDescriptorCountLayoutSupport(Structure):
    pass
VkDescriptorSetVariableDescriptorCountLayoutSupport._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxc_uintariableDescriptorCount", c_uint)
    ]

class VkSubpassDescriptionDepthStencilResolve(Structure):
    pass
VkSubpassDescriptionDepthStencilResolve._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("depthResolveMode", c_int),
             ("stencilResolveMode", c_int),
             ("pDepthStencilResolveAttachment", VkAttachmentReference2)
    ]

class VkPhysicalDeviceDepthStencilResolveProperties(Structure):
    pass
VkPhysicalDeviceDepthStencilResolveProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("supportedDepthResolveModes", c_uint),
             ("supportedStencilResolveModes", c_uint),
             ("independentResolveNone", c_uint),
             ("independentResolve", c_uint)
    ]

class VkPhysicalDeviceScalarBlockLayoutFeatures(Structure):
    pass
VkPhysicalDeviceScalarBlockLayoutFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("scalarBlockLayout", c_uint)
    ]

class VkImageStencilUsageCreateInfo(Structure):
    pass
VkImageStencilUsageCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("stencilUsage", c_uint)
    ]

class VkSamplerReductionModeCreateInfo(Structure):
    pass
VkSamplerReductionModeCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("reductionMode", c_int)
    ]

class VkPhysicalDeviceSamplerFilterMinmaxProperties(Structure):
    pass
VkPhysicalDeviceSamplerFilterMinmaxProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("filterMinmaxSingleComponentFormats", c_uint),
             ("filterMinmaxImageComponentMapping", c_uint)
    ]

class VkPhysicalDeviceVulkanMemoryModelFeatures(Structure):
    pass
VkPhysicalDeviceVulkanMemoryModelFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("vulkanMemoryModel", c_uint),
             ("vulkanMemoryModelDeviceScope", c_uint),
             ("vulkanMemoryModelAvailabilityc_uintisibilityChains", c_uint)
    ]

class VkPhysicalDeviceImagelessFramebufferFeatures(Structure):
    pass
VkPhysicalDeviceImagelessFramebufferFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("imagelessFramebuffer", c_uint)
    ]

class VkFramebufferAttachmentImageInfo(Structure):
    pass
VkFramebufferAttachmentImageInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("usage", c_uint),
             ("width", c_uint),
             ("height", c_uint),
             ("layerCount", c_uint),
             ("viewFormatCount", c_uint),
             ("pPOINTER(c_int)iewFormats", POINTER(c_int))
    ]

class VkFramebufferAttachmentsCreateInfo(Structure):
    pass
VkFramebufferAttachmentsCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("attachmentImageInfoCount", c_uint),
             ("pAttachmentImageInfos", VkFramebufferAttachmentImageInfo)
    ]

class VkRenderPassAttachmentBeginInfo(Structure):
    pass
VkRenderPassAttachmentBeginInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("attachmentCount", c_uint),
             ("pAttachments", POINTER(VkImageView_T))
    ]

class VkPhysicalDeviceUniformBufferStandardLayoutFeatures(Structure):
    pass
VkPhysicalDeviceUniformBufferStandardLayoutFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("uniformBufferStandardLayout", c_uint)
    ]

class VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures(Structure):
    pass
VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderSubgroupExtendedTypes", c_uint)
    ]

class VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures(Structure):
    pass
VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("separateDepthStencilLayouts", c_uint)
    ]

class VkAttachmentReferenceStencilLayout(Structure):
    pass
VkAttachmentReferenceStencilLayout._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("stencilLayout", c_int)
    ]

class VkAttachmentDescriptionStencilLayout(Structure):
    pass
VkAttachmentDescriptionStencilLayout._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("stencilInitialLayout", c_int),
             ("stencilFinalLayout", c_int)
    ]

class VkPhysicalDeviceHostQueryResetFeatures(Structure):
    pass
VkPhysicalDeviceHostQueryResetFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("hostQueryReset", c_uint)
    ]

class VkPhysicalDeviceTimelineSemaphoreFeatures(Structure):
    pass
VkPhysicalDeviceTimelineSemaphoreFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("timelineSemaphore", c_uint)
    ]

class VkPhysicalDeviceTimelineSemaphoreProperties(Structure):
    pass
VkPhysicalDeviceTimelineSemaphoreProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxTimelineSemaphorec_ulongalueDifference", c_ulong)
    ]

class VkSemaphoreTypeCreateInfo(Structure):
    pass
VkSemaphoreTypeCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("semaphoreType", c_int),
             ("initialc_ulongalue", c_ulong)
    ]

class VkTimelineSemaphoreSubmitInfo(Structure):
    pass
VkTimelineSemaphoreSubmitInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("waitSemaphorec_uintalueCount", c_uint),
             ("pWaitSemaphorePOINTER(c_ulong)alues", POINTER(c_ulong)),
             ("signalSemaphorec_uintalueCount", c_uint),
             ("pSignalSemaphorePOINTER(c_ulong)alues", POINTER(c_ulong))
    ]

class VkSemaphoreWaitInfo(Structure):
    pass
VkSemaphoreWaitInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("semaphoreCount", c_uint),
             ("pSemaphores", POINTER(VkSemaphore_T)),
             ("pPOINTER(c_ulong)alues", POINTER(c_ulong))
    ]

class VkSemaphoreSignalInfo(Structure):
    pass
VkSemaphoreSignalInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("semaphore", VkSemaphore_T),
             ("value", c_ulong)
    ]

class VkPhysicalDeviceBufferDeviceAddressFeatures(Structure):
    pass
VkPhysicalDeviceBufferDeviceAddressFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("bufferDeviceAddress", c_uint),
             ("bufferDeviceAddressCaptureReplay", c_uint),
             ("bufferDeviceAddressMultiDevice", c_uint)
    ]

class VkBufferDeviceAddressInfo(Structure):
    pass
VkBufferDeviceAddressInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("buffer", VkBuffer_T)
    ]

class VkBufferOpaqueCaptureAddressCreateInfo(Structure):
    pass
VkBufferOpaqueCaptureAddressCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("opaqueCaptureAddress", c_ulong)
    ]

class VkMemoryOpaqueCaptureAddressAllocateInfo(Structure):
    pass
VkMemoryOpaqueCaptureAddressAllocateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("opaqueCaptureAddress", c_ulong)
    ]

class VkDeviceMemoryOpaqueCaptureAddressInfo(Structure):
    pass
VkDeviceMemoryOpaqueCaptureAddressInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memory", VkDeviceMemory_T)
    ]

class VkPhysicalDeviceVulkan13Features(Structure):
    pass
VkPhysicalDeviceVulkan13Features._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("robustImageAccess", c_uint),
             ("inlineUniformBlock", c_uint),
             ("descriptorBindingInlineUniformBlockUpdateAfterBind", c_uint),
             ("pipelineCreationCacheControl", c_uint),
             ("privateData", c_uint),
             ("shaderDemoteToHelperInvocation", c_uint),
             ("shaderTerminateInvocation", c_uint),
             ("subgroupSizeControl", c_uint),
             ("computeFullSubgroups", c_uint),
             ("synchronization2", c_uint),
             ("textureCompressionASTC_HDR", c_uint),
             ("shaderZeroInitializeWorkgroupMemory", c_uint),
             ("dynamicRendering", c_uint),
             ("shaderIntegerDotProduct", c_uint),
             ("maintenance4", c_uint)
    ]

class VkPhysicalDeviceVulkan13Properties(Structure):
    pass
VkPhysicalDeviceVulkan13Properties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("minSubgroupSize", c_uint),
             ("maxSubgroupSize", c_uint),
             ("maxComputeWorkgroupSubgroups", c_uint),
             ("requiredSubgroupSizeStages", c_uint),
             ("maxInlineUniformBlockSize", c_uint),
             ("maxPerStageDescriptorInlineUniformBlocks", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks", c_uint),
             ("maxDescriptorSetInlineUniformBlocks", c_uint),
             ("maxDescriptorSetUpdateAfterBindInlineUniformBlocks", c_uint),
             ("maxInlineUniformTotalSize", c_uint),
             ("integerDotProduct8BitUnsignedAccelerated", c_uint),
             ("integerDotProduct8BitSignedAccelerated", c_uint),
             ("integerDotProduct8BitMixedSignednessAccelerated", c_uint),
             ("integerDotProduct4x8BitPackedUnsignedAccelerated", c_uint),
             ("integerDotProduct4x8BitPackedSignedAccelerated", c_uint),
             ("integerDotProduct4x8BitPackedMixedSignednessAccelerated", c_uint),
             ("integerDotProduct16BitUnsignedAccelerated", c_uint),
             ("integerDotProduct16BitSignedAccelerated", c_uint),
             ("integerDotProduct16BitMixedSignednessAccelerated", c_uint),
             ("integerDotProduct32BitUnsignedAccelerated", c_uint),
             ("integerDotProduct32BitSignedAccelerated", c_uint),
             ("integerDotProduct32BitMixedSignednessAccelerated", c_uint),
             ("integerDotProduct64BitUnsignedAccelerated", c_uint),
             ("integerDotProduct64BitSignedAccelerated", c_uint),
             ("integerDotProduct64BitMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating8BitUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating8BitSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating16BitUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating16BitSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating32BitUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating32BitSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating64BitUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating64BitSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated", c_uint),
             ("storageTexelBufferOffsetAlignmentBytes", c_ulong),
             ("storageTexelBufferOffsetSingleTexelAlignment", c_uint),
             ("uniformTexelBufferOffsetAlignmentBytes", c_ulong),
             ("uniformTexelBufferOffsetSingleTexelAlignment", c_uint),
             ("maxBufferSize", c_ulong)
    ]

class VkPipelineCreationFeedback(Structure):
    pass
VkPipelineCreationFeedback._fields_ = [
             ("flags", c_uint),
             ("duration", c_ulong)
    ]

class VkPipelineCreationFeedbackCreateInfo(Structure):
    pass
VkPipelineCreationFeedbackCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pPipelineCreationFeedback", VkPipelineCreationFeedback),
             ("pipelineStageCreationFeedbackCount", c_uint),
             ("pPipelineStageCreationFeedbacks", VkPipelineCreationFeedback)
    ]

class VkPhysicalDeviceShaderTerminateInvocationFeatures(Structure):
    pass
VkPhysicalDeviceShaderTerminateInvocationFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderTerminateInvocation", c_uint)
    ]

class VkPhysicalDeviceToolProperties(Structure):
    pass
VkPhysicalDeviceToolProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("name", c_byte *256),
             ("version", c_byte *256),
             ("purposes", c_uint),
             ("description", c_byte *256),
             ("layer", c_byte *256)
    ]

class VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures(Structure):
    pass
VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderDemoteToHelperInvocation", c_uint)
    ]

class VkPhysicalDevicePrivateDataFeatures(Structure):
    pass
VkPhysicalDevicePrivateDataFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("privateData", c_uint)
    ]

class VkDevicePrivateDataCreateInfo(Structure):
    pass
VkDevicePrivateDataCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("privateDataSlotRequestCount", c_uint)
    ]

class VkPrivateDataSlotCreateInfo(Structure):
    pass
VkPrivateDataSlotCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint)
    ]

class VkPhysicalDevicePipelineCreationCacheControlFeatures(Structure):
    pass
VkPhysicalDevicePipelineCreationCacheControlFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pipelineCreationCacheControl", c_uint)
    ]

class VkMemoryBarrier2(Structure):
    pass
VkMemoryBarrier2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcStageMask", c_ulong),
             ("srcAccessMask", c_ulong),
             ("dstStageMask", c_ulong),
             ("dstAccessMask", c_ulong)
    ]

class VkBufferMemoryBarrier2(Structure):
    pass
VkBufferMemoryBarrier2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcStageMask", c_ulong),
             ("srcAccessMask", c_ulong),
             ("dstStageMask", c_ulong),
             ("dstAccessMask", c_ulong),
             ("srcQueueFamilyIndex", c_uint),
             ("dstQueueFamilyIndex", c_uint),
             ("buffer", VkBuffer_T),
             ("offset", c_ulong),
             ("size", c_ulong)
    ]

class VkImageMemoryBarrier2(Structure):
    pass
VkImageMemoryBarrier2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcStageMask", c_ulong),
             ("srcAccessMask", c_ulong),
             ("dstStageMask", c_ulong),
             ("dstAccessMask", c_ulong),
             ("oldLayout", c_int),
             ("newLayout", c_int),
             ("srcQueueFamilyIndex", c_uint),
             ("dstQueueFamilyIndex", c_uint),
             ("image", VkImage_T),
             ("subresourceRange", VkImageSubresourceRange)
    ]

class VkDependencyInfo(Structure):
    pass
VkDependencyInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("dependencyFlags", c_uint),
             ("memoryBarrierCount", c_uint),
             ("pMemoryBarriers", VkMemoryBarrier2),
             ("bufferMemoryBarrierCount", c_uint),
             ("pBufferMemoryBarriers", VkBufferMemoryBarrier2),
             ("imageMemoryBarrierCount", c_uint),
             ("pImageMemoryBarriers", VkImageMemoryBarrier2)
    ]

class VkSemaphoreSubmitInfo(Structure):
    pass
VkSemaphoreSubmitInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("semaphore", VkSemaphore_T),
             ("value", c_ulong),
             ("stageMask", c_ulong),
             ("deviceIndex", c_uint)
    ]

class VkCommandBufferSubmitInfo(Structure):
    pass
VkCommandBufferSubmitInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("commandBuffer", VkCommandBuffer_T),
             ("deviceMask", c_uint)
    ]

class VkSubmitInfo2(Structure):
    pass
VkSubmitInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("waitSemaphoreInfoCount", c_uint),
             ("pWaitSemaphoreInfos", VkSemaphoreSubmitInfo),
             ("commandBufferInfoCount", c_uint),
             ("pCommandBufferInfos", VkCommandBufferSubmitInfo),
             ("signalSemaphoreInfoCount", c_uint),
             ("pSignalSemaphoreInfos", VkSemaphoreSubmitInfo)
    ]

class VkPhysicalDeviceSynchronization2Features(Structure):
    pass
VkPhysicalDeviceSynchronization2Features._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("synchronization2", c_uint)
    ]

class VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures(Structure):
    pass
VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderZeroInitializeWorkgroupMemory", c_uint)
    ]

class VkPhysicalDeviceImageRobustnessFeatures(Structure):
    pass
VkPhysicalDeviceImageRobustnessFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("robustImageAccess", c_uint)
    ]

class VkBufferCopy2(Structure):
    pass
VkBufferCopy2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcOffset", c_ulong),
             ("dstOffset", c_ulong),
             ("size", c_ulong)
    ]

class VkCopyBufferInfo2(Structure):
    pass
VkCopyBufferInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcBuffer", VkBuffer_T),
             ("dstBuffer", VkBuffer_T),
             ("regionCount", c_uint),
             ("pRegions", VkBufferCopy2)
    ]

class VkImageCopy2(Structure):
    pass
VkImageCopy2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcSubresource", VkImageSubresourceLayers),
             ("srcOffset", VkOffset3D),
             ("dstSubresource", VkImageSubresourceLayers),
             ("dstOffset", VkOffset3D),
             ("extent", VkExtent3D)
    ]

class VkCopyImageInfo2(Structure):
    pass
VkCopyImageInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcImage", VkImage_T),
             ("srcImageLayout", c_int),
             ("dstImage", VkImage_T),
             ("dstImageLayout", c_int),
             ("regionCount", c_uint),
             ("pRegions", VkImageCopy2)
    ]

class VkBufferImageCopy2(Structure):
    pass
VkBufferImageCopy2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("bufferOffset", c_ulong),
             ("bufferRowLength", c_uint),
             ("bufferImageHeight", c_uint),
             ("imageSubresource", VkImageSubresourceLayers),
             ("imageOffset", VkOffset3D),
             ("imageExtent", VkExtent3D)
    ]

class VkCopyBufferToImageInfo2(Structure):
    pass
VkCopyBufferToImageInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcBuffer", VkBuffer_T),
             ("dstImage", VkImage_T),
             ("dstImageLayout", c_int),
             ("regionCount", c_uint),
             ("pRegions", VkBufferImageCopy2)
    ]

class VkCopyImageToBufferInfo2(Structure):
    pass
VkCopyImageToBufferInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcImage", VkImage_T),
             ("srcImageLayout", c_int),
             ("dstBuffer", VkBuffer_T),
             ("regionCount", c_uint),
             ("pRegions", VkBufferImageCopy2)
    ]

class VkImageBlit2(Structure):
    pass
VkImageBlit2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcSubresource", VkImageSubresourceLayers),
             ("srcOffsets", VkOffset3D *2),
             ("dstSubresource", VkImageSubresourceLayers),
             ("dstOffsets", VkOffset3D *2)
    ]

class VkBlitImageInfo2(Structure):
    pass
VkBlitImageInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcImage", VkImage_T),
             ("srcImageLayout", c_int),
             ("dstImage", VkImage_T),
             ("dstImageLayout", c_int),
             ("regionCount", c_uint),
             ("pRegions", VkImageBlit2),
             ("filter", c_int)
    ]

class VkImageResolve2(Structure):
    pass
VkImageResolve2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcSubresource", VkImageSubresourceLayers),
             ("srcOffset", VkOffset3D),
             ("dstSubresource", VkImageSubresourceLayers),
             ("dstOffset", VkOffset3D),
             ("extent", VkExtent3D)
    ]

class VkResolveImageInfo2(Structure):
    pass
VkResolveImageInfo2._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcImage", VkImage_T),
             ("srcImageLayout", c_int),
             ("dstImage", VkImage_T),
             ("dstImageLayout", c_int),
             ("regionCount", c_uint),
             ("pRegions", VkImageResolve2)
    ]

class VkPhysicalDeviceSubgroupSizeControlFeatures(Structure):
    pass
VkPhysicalDeviceSubgroupSizeControlFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("subgroupSizeControl", c_uint),
             ("computeFullSubgroups", c_uint)
    ]

class VkPhysicalDeviceSubgroupSizeControlProperties(Structure):
    pass
VkPhysicalDeviceSubgroupSizeControlProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("minSubgroupSize", c_uint),
             ("maxSubgroupSize", c_uint),
             ("maxComputeWorkgroupSubgroups", c_uint),
             ("requiredSubgroupSizeStages", c_uint)
    ]

class VkPipelineShaderStageRequiredSubgroupSizeCreateInfo(Structure):
    pass
VkPipelineShaderStageRequiredSubgroupSizeCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("requiredSubgroupSize", c_uint)
    ]

class VkPhysicalDeviceInlineUniformBlockFeatures(Structure):
    pass
VkPhysicalDeviceInlineUniformBlockFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("inlineUniformBlock", c_uint),
             ("descriptorBindingInlineUniformBlockUpdateAfterBind", c_uint)
    ]

class VkPhysicalDeviceInlineUniformBlockProperties(Structure):
    pass
VkPhysicalDeviceInlineUniformBlockProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxInlineUniformBlockSize", c_uint),
             ("maxPerStageDescriptorInlineUniformBlocks", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks", c_uint),
             ("maxDescriptorSetInlineUniformBlocks", c_uint),
             ("maxDescriptorSetUpdateAfterBindInlineUniformBlocks", c_uint)
    ]

class VkWriteDescriptorSetInlineUniformBlock(Structure):
    pass
VkWriteDescriptorSetInlineUniformBlock._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("dataSize", c_uint),
             ("pData", c_void_p)
    ]

class VkDescriptorPoolInlineUniformBlockCreateInfo(Structure):
    pass
VkDescriptorPoolInlineUniformBlockCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxInlineUniformBlockBindings", c_uint)
    ]

class VkPhysicalDeviceTextureCompressionASTCHDRFeatures(Structure):
    pass
VkPhysicalDeviceTextureCompressionASTCHDRFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("textureCompressionASTC_HDR", c_uint)
    ]

class VkRenderingAttachmentInfo(Structure):
    pass
VkRenderingAttachmentInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("imageVkImageView_Tiew", VkImageView_T),
             ("imageLayout", c_int),
             ("resolveMode", c_int),
             ("resolveImageVkImageView_Tiew", VkImageView_T),
             ("resolveImageLayout", c_int),
             ("loadOp", c_int),
             ("storeOp", c_int),
             ("clearVkClearValuealue", VkClearValue)
    ]

class VkRenderingInfo(Structure):
    pass
VkRenderingInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("renderArea", VkRect2D),
             ("layerCount", c_uint),
             ("viewMask", c_uint),
             ("colorAttachmentCount", c_uint),
             ("pColorAttachments", VkRenderingAttachmentInfo),
             ("pDepthAttachment", VkRenderingAttachmentInfo),
             ("pStencilAttachment", VkRenderingAttachmentInfo)
    ]

class VkPipelineRenderingCreateInfo(Structure):
    pass
VkPipelineRenderingCreateInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("viewMask", c_uint),
             ("colorAttachmentCount", c_uint),
             ("pColorAttachmentFormats", POINTER(c_int)),
             ("depthAttachmentFormat", c_int),
             ("stencilAttachmentFormat", c_int)
    ]

class VkPhysicalDeviceDynamicRenderingFeatures(Structure):
    pass
VkPhysicalDeviceDynamicRenderingFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("dynamicRendering", c_uint)
    ]

class VkCommandBufferInheritanceRenderingInfo(Structure):
    pass
VkCommandBufferInheritanceRenderingInfo._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("viewMask", c_uint),
             ("colorAttachmentCount", c_uint),
             ("pColorAttachmentFormats", POINTER(c_int)),
             ("depthAttachmentFormat", c_int),
             ("stencilAttachmentFormat", c_int),
             ("rasterizationSamples", c_int)
    ]

class VkPhysicalDeviceShaderIntegerDotProductFeatures(Structure):
    pass
VkPhysicalDeviceShaderIntegerDotProductFeatures._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderIntegerDotProduct", c_uint)
    ]

class VkPhysicalDeviceShaderIntegerDotProductProperties(Structure):
    pass
VkPhysicalDeviceShaderIntegerDotProductProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("integerDotProduct8BitUnsignedAccelerated", c_uint),
             ("integerDotProduct8BitSignedAccelerated", c_uint),
             ("integerDotProduct8BitMixedSignednessAccelerated", c_uint),
             ("integerDotProduct4x8BitPackedUnsignedAccelerated", c_uint),
             ("integerDotProduct4x8BitPackedSignedAccelerated", c_uint),
             ("integerDotProduct4x8BitPackedMixedSignednessAccelerated", c_uint),
             ("integerDotProduct16BitUnsignedAccelerated", c_uint),
             ("integerDotProduct16BitSignedAccelerated", c_uint),
             ("integerDotProduct16BitMixedSignednessAccelerated", c_uint),
             ("integerDotProduct32BitUnsignedAccelerated", c_uint),
             ("integerDotProduct32BitSignedAccelerated", c_uint),
             ("integerDotProduct32BitMixedSignednessAccelerated", c_uint),
             ("integerDotProduct64BitUnsignedAccelerated", c_uint),
             ("integerDotProduct64BitSignedAccelerated", c_uint),
             ("integerDotProduct64BitMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating8BitUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating8BitSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating16BitUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating16BitSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating32BitUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating32BitSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating64BitUnsignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating64BitSignedAccelerated", c_uint),
             ("integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated", c_uint)
    ]

class VkPhysicalDeviceTexelBufferAlignmentProperties(Structure):
    pass
VkPhysicalDeviceTexelBufferAlignmentProperties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("storageTexelBufferOffsetAlignmentBytes", c_ulong),
             ("storageTexelBufferOffsetSingleTexelAlignment", c_uint),
             ("uniformTexelBufferOffsetAlignmentBytes", c_ulong),
             ("uniformTexelBufferOffsetSingleTexelAlignment", c_uint)
    ]

class VkFormatProperties3(Structure):
    pass
VkFormatProperties3._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("linearTilingFeatures", c_ulong),
             ("optimalTilingFeatures", c_ulong),
             ("bufferFeatures", c_ulong)
    ]

class VkPhysicalDeviceMaintenance4Features(Structure):
    pass
VkPhysicalDeviceMaintenance4Features._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maintenance4", c_uint)
    ]

class VkPhysicalDeviceMaintenance4Properties(Structure):
    pass
VkPhysicalDeviceMaintenance4Properties._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxBufferSize", c_ulong)
    ]

class VkDeviceBufferMemoryRequirements(Structure):
    pass
VkDeviceBufferMemoryRequirements._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pCreateInfo", VkBufferCreateInfo)
    ]

class VkDeviceImageMemoryRequirements(Structure):
    pass
VkDeviceImageMemoryRequirements._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pCreateInfo", VkImageCreateInfo),
             ("planeAspect", c_int)
    ]

class VkSurfaceCapabilitiesKHR(Structure):
    pass
VkSurfaceCapabilitiesKHR._fields_ = [
             ("minImageCount", c_uint),
             ("maxImageCount", c_uint),
             ("currentExtent", VkExtent2D),
             ("minImageExtent", VkExtent2D),
             ("maxImageExtent", VkExtent2D),
             ("maxImageArrayLayers", c_uint),
             ("supportedTransforms", c_uint),
             ("currentTransform", c_int),
             ("supportedCompositeAlpha", c_uint),
             ("supportedUsageFlags", c_uint)
    ]

class VkSurfaceFormatKHR(Structure):
    pass
VkSurfaceFormatKHR._fields_ = [
             ("format", c_int),
             ("colorSpace", c_int)
    ]

class VkSwapchainCreateInfoKHR(Structure):
    pass
VkSwapchainCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("surface", VkSurfaceKHR_T),
             ("minImageCount", c_uint),
             ("imageFormat", c_int),
             ("imageColorSpace", c_int),
             ("imageExtent", VkExtent2D),
             ("imageArrayLayers", c_uint),
             ("imageUsage", c_uint),
             ("imageSharingMode", c_int),
             ("queueFamilyIndexCount", c_uint),
             ("pQueueFamilyIndices", POINTER(c_uint)),
             ("preTransform", c_int),
             ("compositeAlpha", c_int),
             ("presentMode", c_int),
             ("clipped", c_uint),
             ("oldSwapchain", VkSwapchainKHR_T)
    ]

class VkPresentInfoKHR(Structure):
    pass
VkPresentInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("waitSemaphoreCount", c_uint),
             ("pWaitSemaphores", POINTER(VkSemaphore_T)),
             ("swapchainCount", c_uint),
             ("pSwapchains", POINTER(VkSwapchainKHR_T)),
             ("pImageIndices", POINTER(c_uint)),
             ("pResults", POINTER(c_int))
    ]

class VkImageSwapchainCreateInfoKHR(Structure):
    pass
VkImageSwapchainCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("swapchain", VkSwapchainKHR_T)
    ]

class VkBindImageMemorySwapchainInfoKHR(Structure):
    pass
VkBindImageMemorySwapchainInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("swapchain", VkSwapchainKHR_T),
             ("imageIndex", c_uint)
    ]

class VkAcquireNextImageInfoKHR(Structure):
    pass
VkAcquireNextImageInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("swapchain", VkSwapchainKHR_T),
             ("timeout", c_ulong),
             ("semaphore", VkSemaphore_T),
             ("fence", VkFence_T),
             ("deviceMask", c_uint)
    ]

class VkDeviceGroupPresentCapabilitiesKHR(Structure):
    pass
VkDeviceGroupPresentCapabilitiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("presentMask", c_uint *32),
             ("modes", c_uint)
    ]

class VkDeviceGroupPresentInfoKHR(Structure):
    pass
VkDeviceGroupPresentInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("swapchainCount", c_uint),
             ("pDeviceMasks", POINTER(c_uint)),
             ("mode", c_int)
    ]

class VkDeviceGroupSwapchainCreateInfoKHR(Structure):
    pass
VkDeviceGroupSwapchainCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("modes", c_uint)
    ]

class VkDisplayModeParametersKHR(Structure):
    pass
VkDisplayModeParametersKHR._fields_ = [
             ("visibleRegion", VkExtent2D),
             ("refreshRate", c_uint)
    ]

class VkDisplayModeCreateInfoKHR(Structure):
    pass
VkDisplayModeCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("parameters", VkDisplayModeParametersKHR)
    ]

class VkDisplayModePropertiesKHR(Structure):
    pass
VkDisplayModePropertiesKHR._fields_ = [
             ("displayMode", VkDisplayModeKHR_T),
             ("parameters", VkDisplayModeParametersKHR)
    ]

class VkDisplayPlaneCapabilitiesKHR(Structure):
    pass
VkDisplayPlaneCapabilitiesKHR._fields_ = [
             ("supportedAlpha", c_uint),
             ("minSrcPosition", VkOffset2D),
             ("maxSrcPosition", VkOffset2D),
             ("minSrcExtent", VkExtent2D),
             ("maxSrcExtent", VkExtent2D),
             ("minDstPosition", VkOffset2D),
             ("maxDstPosition", VkOffset2D),
             ("minDstExtent", VkExtent2D),
             ("maxDstExtent", VkExtent2D)
    ]

class VkDisplayPlanePropertiesKHR(Structure):
    pass
VkDisplayPlanePropertiesKHR._fields_ = [
             ("currentDisplay", VkDisplayKHR_T),
             ("currentStackIndex", c_uint)
    ]

class VkDisplayPropertiesKHR(Structure):
    pass
VkDisplayPropertiesKHR._fields_ = [
             ("display", VkDisplayKHR_T),
             ("displayName", c_char_p),
             ("physicalDimensions", VkExtent2D),
             ("physicalResolution", VkExtent2D),
             ("supportedTransforms", c_uint),
             ("planeReorderPossible", c_uint),
             ("persistentContent", c_uint)
    ]

class VkDisplaySurfaceCreateInfoKHR(Structure):
    pass
VkDisplaySurfaceCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("displayMode", VkDisplayModeKHR_T),
             ("planeIndex", c_uint),
             ("planeStackIndex", c_uint),
             ("transform", c_int),
             ("globalAlpha", c_float),
             ("alphaMode", c_int),
             ("imageExtent", VkExtent2D)
    ]

class VkDisplayPresentInfoKHR(Structure):
    pass
VkDisplayPresentInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcRect", VkRect2D),
             ("dstRect", VkRect2D),
             ("persistent", c_uint)
    ]

class VkRenderingFragmentShadingRateAttachmentInfoKHR(Structure):
    pass
VkRenderingFragmentShadingRateAttachmentInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("imageVkImageView_Tiew", VkImageView_T),
             ("imageLayout", c_int),
             ("shadingRateAttachmentTexelSize", VkExtent2D)
    ]

class VkRenderingFragmentDensityMapAttachmentInfoEXT(Structure):
    pass
VkRenderingFragmentDensityMapAttachmentInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("imageVkImageView_Tiew", VkImageView_T),
             ("imageLayout", c_int)
    ]

class VkAttachmentSampleCountInfoAMD(Structure):
    pass
VkAttachmentSampleCountInfoAMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("colorAttachmentCount", c_uint),
             ("pColorAttachmentSamples", POINTER(c_int)),
             ("depthStencilAttachmentSamples", c_int)
    ]

class VkMultiviewPerViewAttributesInfoNVX(Structure):
    pass
VkMultiviewPerViewAttributesInfoNVX._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("perc_uintiewAttributes", c_uint),
             ("perc_uintiewAttributesPositionXOnly", c_uint)
    ]

class VkImportMemoryFdInfoKHR(Structure):
    pass
VkImportMemoryFdInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleType", c_int),
             ("fd", c_int)
    ]

class VkMemoryFdPropertiesKHR(Structure):
    pass
VkMemoryFdPropertiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memoryTypeBits", c_uint)
    ]

class VkMemoryGetFdInfoKHR(Structure):
    pass
VkMemoryGetFdInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memory", VkDeviceMemory_T),
             ("handleType", c_int)
    ]

class VkImportSemaphoreFdInfoKHR(Structure):
    pass
VkImportSemaphoreFdInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("semaphore", VkSemaphore_T),
             ("flags", c_uint),
             ("handleType", c_int),
             ("fd", c_int)
    ]

class VkSemaphoreGetFdInfoKHR(Structure):
    pass
VkSemaphoreGetFdInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("semaphore", VkSemaphore_T),
             ("handleType", c_int)
    ]

class VkPhysicalDevicePushDescriptorPropertiesKHR(Structure):
    pass
VkPhysicalDevicePushDescriptorPropertiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxPushDescriptors", c_uint)
    ]

class VkRectLayerKHR(Structure):
    pass
VkRectLayerKHR._fields_ = [
             ("offset", VkOffset2D),
             ("extent", VkExtent2D),
             ("layer", c_uint)
    ]

class VkPresentRegionKHR(Structure):
    pass
VkPresentRegionKHR._fields_ = [
             ("rectangleCount", c_uint),
             ("pRectangles", VkRectLayerKHR)
    ]

class VkPresentRegionsKHR(Structure):
    pass
VkPresentRegionsKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("swapchainCount", c_uint),
             ("pRegions", VkPresentRegionKHR)
    ]

class VkSharedPresentSurfaceCapabilitiesKHR(Structure):
    pass
VkSharedPresentSurfaceCapabilitiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("sharedPresentSupportedUsageFlags", c_uint)
    ]

class VkImportFenceFdInfoKHR(Structure):
    pass
VkImportFenceFdInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fence", VkFence_T),
             ("flags", c_uint),
             ("handleType", c_int),
             ("fd", c_int)
    ]

class VkFenceGetFdInfoKHR(Structure):
    pass
VkFenceGetFdInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fence", VkFence_T),
             ("handleType", c_int)
    ]

class VkPhysicalDevicePerformanceQueryFeaturesKHR(Structure):
    pass
VkPhysicalDevicePerformanceQueryFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("performanceCounterQueryPools", c_uint),
             ("performanceCounterMultipleQueryPools", c_uint)
    ]

class VkPhysicalDevicePerformanceQueryPropertiesKHR(Structure):
    pass
VkPhysicalDevicePerformanceQueryPropertiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("allowCommandBufferQueryCopies", c_uint)
    ]

class VkPerformanceCounterKHR(Structure):
    pass
VkPerformanceCounterKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("unit", c_int),
             ("scope", c_int),
             ("storage", c_int),
             ("uuid", c_ubyte *16)
    ]

class VkPerformanceCounterDescriptionKHR(Structure):
    pass
VkPerformanceCounterDescriptionKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("name", c_byte *256),
             ("category", c_byte *256),
             ("description", c_byte *256)
    ]

class VkQueryPoolPerformanceCreateInfoKHR(Structure):
    pass
VkQueryPoolPerformanceCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("queueFamilyIndex", c_uint),
             ("counterIndexCount", c_uint),
             ("pCounterIndices", POINTER(c_uint))
    ]

class VkPerformanceCounterResultKHR(Structure):
    pass
VkPerformanceCounterResultKHR._fields_ = [
             ("int32", c_int),
             ("int64", c_long),
             ("uint32", c_uint),
             ("uint64", c_ulong),
             ("float32", c_float),
             ("float64", c_double)
    ]

class VkAcquireProfilingLockInfoKHR(Structure):
    pass
VkAcquireProfilingLockInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("timeout", c_ulong)
    ]

class VkPerformanceQuerySubmitInfoKHR(Structure):
    pass
VkPerformanceQuerySubmitInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("counterPassIndex", c_uint)
    ]

class VkPhysicalDeviceSurfaceInfo2KHR(Structure):
    pass
VkPhysicalDeviceSurfaceInfo2KHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("surface", VkSurfaceKHR_T)
    ]

class VkSurfaceCapabilities2KHR(Structure):
    pass
VkSurfaceCapabilities2KHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("surfaceCapabilities", VkSurfaceCapabilitiesKHR)
    ]

class VkSurfaceFormat2KHR(Structure):
    pass
VkSurfaceFormat2KHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("surfaceFormat", VkSurfaceFormatKHR)
    ]

class VkDisplayProperties2KHR(Structure):
    pass
VkDisplayProperties2KHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("displayProperties", VkDisplayPropertiesKHR)
    ]

class VkDisplayPlaneProperties2KHR(Structure):
    pass
VkDisplayPlaneProperties2KHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("displayPlaneProperties", VkDisplayPlanePropertiesKHR)
    ]

class VkDisplayModeProperties2KHR(Structure):
    pass
VkDisplayModeProperties2KHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("displayModeProperties", VkDisplayModePropertiesKHR)
    ]

class VkDisplayPlaneInfo2KHR(Structure):
    pass
VkDisplayPlaneInfo2KHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("mode", VkDisplayModeKHR_T),
             ("planeIndex", c_uint)
    ]

class VkDisplayPlaneCapabilities2KHR(Structure):
    pass
VkDisplayPlaneCapabilities2KHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("capabilities", VkDisplayPlaneCapabilitiesKHR)
    ]

class VkPhysicalDeviceShaderClockFeaturesKHR(Structure):
    pass
VkPhysicalDeviceShaderClockFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderSubgroupClock", c_uint),
             ("shaderDeviceClock", c_uint)
    ]

class VkDeviceQueueGlobalPriorityCreateInfoKHR(Structure):
    pass
VkDeviceQueueGlobalPriorityCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("globalPriority", c_int)
    ]

class VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR(Structure):
    pass
VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("globalPriorityQuery", c_uint)
    ]

class VkQueueFamilyGlobalPriorityPropertiesKHR(Structure):
    pass
VkQueueFamilyGlobalPriorityPropertiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("priorityCount", c_uint),
             ("priorities", c_int *16)
    ]

class VkFragmentShadingRateAttachmentInfoKHR(Structure):
    pass
VkFragmentShadingRateAttachmentInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pFragmentShadingRateAttachment", VkAttachmentReference2),
             ("shadingRateAttachmentTexelSize", VkExtent2D)
    ]

class VkPipelineFragmentShadingRateStateCreateInfoKHR(Structure):
    pass
VkPipelineFragmentShadingRateStateCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentSize", VkExtent2D),
             ("combinerOps", c_int *2)
    ]

class VkPhysicalDeviceFragmentShadingRateFeaturesKHR(Structure):
    pass
VkPhysicalDeviceFragmentShadingRateFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pipelineFragmentShadingRate", c_uint),
             ("primitiveFragmentShadingRate", c_uint),
             ("attachmentFragmentShadingRate", c_uint)
    ]

class VkPhysicalDeviceFragmentShadingRatePropertiesKHR(Structure):
    pass
VkPhysicalDeviceFragmentShadingRatePropertiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("minFragmentShadingRateAttachmentTexelSize", VkExtent2D),
             ("maxFragmentShadingRateAttachmentTexelSize", VkExtent2D),
             ("maxFragmentShadingRateAttachmentTexelSizeAspectRatio", c_uint),
             ("primitiveFragmentShadingRateWithMultiplec_uintiewports", c_uint),
             ("layeredShadingRateAttachments", c_uint),
             ("fragmentShadingRateNonTrivialCombinerOps", c_uint),
             ("maxFragmentSize", VkExtent2D),
             ("maxFragmentSizeAspectRatio", c_uint),
             ("maxFragmentShadingRateCoverageSamples", c_uint),
             ("maxFragmentShadingRateRasterizationSamples", c_int),
             ("fragmentShadingRateWithShaderDepthStencilWrites", c_uint),
             ("fragmentShadingRateWithSampleMask", c_uint),
             ("fragmentShadingRateWithShaderSampleMask", c_uint),
             ("fragmentShadingRateWithConservativeRasterization", c_uint),
             ("fragmentShadingRateWithFragmentShaderInterlock", c_uint),
             ("fragmentShadingRateWithCustomSampleLocations", c_uint),
             ("fragmentShadingRateStrictMultiplyCombiner", c_uint)
    ]

class VkPhysicalDeviceFragmentShadingRateKHR(Structure):
    pass
VkPhysicalDeviceFragmentShadingRateKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("sampleCounts", c_uint),
             ("fragmentSize", VkExtent2D)
    ]

class VkSurfaceProtectedCapabilitiesKHR(Structure):
    pass
VkSurfaceProtectedCapabilitiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("supportsProtected", c_uint)
    ]

class VkPhysicalDevicePresentWaitFeaturesKHR(Structure):
    pass
VkPhysicalDevicePresentWaitFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("presentWait", c_uint)
    ]

class VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR(Structure):
    pass
VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pipelineExecutableInfo", c_uint)
    ]

class VkPipelineInfoKHR(Structure):
    pass
VkPipelineInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pipeline", VkPipeline_T)
    ]

class VkPipelineExecutablePropertiesKHR(Structure):
    pass
VkPipelineExecutablePropertiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("stages", c_uint),
             ("name", c_byte *256),
             ("description", c_byte *256),
             ("subgroupSize", c_uint)
    ]

class VkPipelineExecutableInfoKHR(Structure):
    pass
VkPipelineExecutableInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pipeline", VkPipeline_T),
             ("executableIndex", c_uint)
    ]

class VkPipelineExecutableStatisticValueKHR(Structure):
    pass
VkPipelineExecutableStatisticValueKHR._fields_ = [
             ("b32", c_uint),
             ("i64", c_long),
             ("u64", c_ulong),
             ("f64", c_double)
    ]

class VkPipelineExecutableStatisticKHR(Structure):
    pass
VkPipelineExecutableStatisticKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("name", c_byte *256),
             ("description", c_byte *256),
             ("format", c_int),
             ("value", VkPipelineExecutableStatisticValueKHR)
    ]

class VkPipelineExecutableInternalRepresentationKHR(Structure):
    pass
VkPipelineExecutableInternalRepresentationKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("name", c_byte *256),
             ("description", c_byte *256),
             ("isText", c_uint),
             ("dataSize", c_ulong),
             ("pData", c_void_p)
    ]

class VkPipelineLibraryCreateInfoKHR(Structure):
    pass
VkPipelineLibraryCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("libraryCount", c_uint),
             ("pLibraries", POINTER(VkPipeline_T))
    ]

class VkPresentIdKHR(Structure):
    pass
VkPresentIdKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("swapchainCount", c_uint),
             ("pPresentIds", POINTER(c_ulong))
    ]

class VkPhysicalDevicePresentIdFeaturesKHR(Structure):
    pass
VkPhysicalDevicePresentIdFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("presentId", c_uint)
    ]

class VkQueueFamilyCheckpointProperties2NV(Structure):
    pass
VkQueueFamilyCheckpointProperties2NV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("checkpointExecutionStageMask", c_ulong)
    ]

class VkCheckpointData2NV(Structure):
    pass
VkCheckpointData2NV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("stage", c_ulong),
             ("pCheckpointMarker", c_void_p)
    ]

class VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR(Structure):
    pass
VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderSubgroupUniformControlFlow", c_uint)
    ]

class VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR(Structure):
    pass
VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("workgroupMemoryExplicitLayout", c_uint),
             ("workgroupMemoryExplicitLayoutScalarBlockLayout", c_uint),
             ("workgroupMemoryExplicitLayout8BitAccess", c_uint),
             ("workgroupMemoryExplicitLayout16BitAccess", c_uint)
    ]

class VkDebugReportCallbackCreateInfoEXT(Structure):
    pass
VkDebugReportCallbackCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("pfnCallback", c_void_p),
             ("pUserData", c_void_p)
    ]

class VkPipelineRasterizationStateRasterizationOrderAMD(Structure):
    pass
VkPipelineRasterizationStateRasterizationOrderAMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("rasterizationOrder", c_int)
    ]

class VkDebugMarkerObjectNameInfoEXT(Structure):
    pass
VkDebugMarkerObjectNameInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("objectType", c_int),
             ("object", c_ulong),
             ("pObjectName", c_char_p)
    ]

class VkDebugMarkerObjectTagInfoEXT(Structure):
    pass
VkDebugMarkerObjectTagInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("objectType", c_int),
             ("object", c_ulong),
             ("tagName", c_ulong),
             ("tagSize", c_ulong),
             ("pTag", c_void_p)
    ]

class VkDebugMarkerMarkerInfoEXT(Structure):
    pass
VkDebugMarkerMarkerInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pMarkerName", c_char_p),
             ("color", c_float *4)
    ]

class VkDedicatedAllocationImageCreateInfoNV(Structure):
    pass
VkDedicatedAllocationImageCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("dedicatedAllocation", c_uint)
    ]

class VkDedicatedAllocationBufferCreateInfoNV(Structure):
    pass
VkDedicatedAllocationBufferCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("dedicatedAllocation", c_uint)
    ]

class VkDedicatedAllocationMemoryAllocateInfoNV(Structure):
    pass
VkDedicatedAllocationMemoryAllocateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("image", VkImage_T),
             ("buffer", VkBuffer_T)
    ]

class VkPhysicalDeviceTransformFeedbackFeaturesEXT(Structure):
    pass
VkPhysicalDeviceTransformFeedbackFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("transformFeedback", c_uint),
             ("geometryStreams", c_uint)
    ]

class VkPhysicalDeviceTransformFeedbackPropertiesEXT(Structure):
    pass
VkPhysicalDeviceTransformFeedbackPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxTransformFeedbackStreams", c_uint),
             ("maxTransformFeedbackBuffers", c_uint),
             ("maxTransformFeedbackBufferSize", c_ulong),
             ("maxTransformFeedbackStreamDataSize", c_uint),
             ("maxTransformFeedbackBufferDataSize", c_uint),
             ("maxTransformFeedbackBufferDataStride", c_uint),
             ("transformFeedbackQueries", c_uint),
             ("transformFeedbackStreamsLinesTriangles", c_uint),
             ("transformFeedbackRasterizationStreamSelect", c_uint),
             ("transformFeedbackDraw", c_uint)
    ]

class VkPipelineRasterizationStateStreamCreateInfoEXT(Structure):
    pass
VkPipelineRasterizationStateStreamCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("rasterizationStream", c_uint)
    ]

class VkCuModuleCreateInfoNVX(Structure):
    pass
VkCuModuleCreateInfoNVX._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("dataSize", c_ulong),
             ("pData", c_void_p)
    ]

class VkCuFunctionCreateInfoNVX(Structure):
    pass
VkCuFunctionCreateInfoNVX._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("module", VkCuModuleNVX_T),
             ("pName", c_char_p)
    ]

class VkCuLaunchInfoNVX(Structure):
    pass
VkCuLaunchInfoNVX._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("function", VkCuFunctionNVX_T),
             ("gridDimX", c_uint),
             ("gridDimY", c_uint),
             ("gridDimZ", c_uint),
             ("blockDimX", c_uint),
             ("blockDimY", c_uint),
             ("blockDimZ", c_uint),
             ("sharedMemBytes", c_uint),
             ("paramCount", c_ulong),
             ("pParams", POINTER(c_void_p)),
             ("extraCount", c_ulong),
             ("pExtras", POINTER(c_void_p))
    ]

class VkImageViewHandleInfoNVX(Structure):
    pass
VkImageViewHandleInfoNVX._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("imageVkImageView_Tiew", VkImageView_T),
             ("descriptorType", c_int),
             ("sampler", VkSampler_T)
    ]

class VkImageViewAddressPropertiesNVX(Structure):
    pass
VkImageViewAddressPropertiesNVX._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceAddress", c_ulong),
             ("size", c_ulong)
    ]

class VkTextureLODGatherFormatPropertiesAMD(Structure):
    pass
VkTextureLODGatherFormatPropertiesAMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("supportsTextureGatherLODBiasAMD", c_uint)
    ]

class VkShaderResourceUsageAMD(Structure):
    pass
VkShaderResourceUsageAMD._fields_ = [
             ("numUsedc_uintgprs", c_uint),
             ("numUsedSgprs", c_uint),
             ("ldsSizePerLocalWorkGroup", c_uint),
             ("ldsUsageSizeInBytes", c_ulong),
             ("scratchMemUsageInBytes", c_ulong)
    ]

class VkShaderStatisticsInfoAMD(Structure):
    pass
VkShaderStatisticsInfoAMD._fields_ = [
             ("shaderStageMask", c_uint),
             ("resourceUsage", VkShaderResourceUsageAMD),
             ("numPhysicalc_uintgprs", c_uint),
             ("numPhysicalSgprs", c_uint),
             ("numAvailablec_uintgprs", c_uint),
             ("numAvailableSgprs", c_uint),
             ("computeWorkGroupSize", c_uint *3)
    ]

class VkPhysicalDeviceCornerSampledImageFeaturesNV(Structure):
    pass
VkPhysicalDeviceCornerSampledImageFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("cornerSampledImage", c_uint)
    ]

class VkExternalImageFormatPropertiesNV(Structure):
    pass
VkExternalImageFormatPropertiesNV._fields_ = [
             ("imageFormatProperties", VkImageFormatProperties),
             ("externalMemoryFeatures", c_uint),
             ("exportFromImportedHandleTypes", c_uint),
             ("compatibleHandleTypes", c_uint)
    ]

class VkExternalMemoryImageCreateInfoNV(Structure):
    pass
VkExternalMemoryImageCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleTypes", c_uint)
    ]

class VkExportMemoryAllocateInfoNV(Structure):
    pass
VkExportMemoryAllocateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleTypes", c_uint)
    ]

class VkValidationFlagsEXT(Structure):
    pass
VkValidationFlagsEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("disabledc_uintalidationCheckCount", c_uint),
             ("pDisabledPOINTER(c_int)alidationChecks", POINTER(c_int))
    ]

class VkImageViewASTCDecodeModeEXT(Structure):
    pass
VkImageViewASTCDecodeModeEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("decodeMode", c_int)
    ]

class VkPhysicalDeviceASTCDecodeFeaturesEXT(Structure):
    pass
VkPhysicalDeviceASTCDecodeFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("decodeModeSharedExponent", c_uint)
    ]

class VkConditionalRenderingBeginInfoEXT(Structure):
    pass
VkConditionalRenderingBeginInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("buffer", VkBuffer_T),
             ("offset", c_ulong),
             ("flags", c_uint)
    ]

class VkPhysicalDeviceConditionalRenderingFeaturesEXT(Structure):
    pass
VkPhysicalDeviceConditionalRenderingFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("conditionalRendering", c_uint),
             ("inheritedConditionalRendering", c_uint)
    ]

class VkCommandBufferInheritanceConditionalRenderingInfoEXT(Structure):
    pass
VkCommandBufferInheritanceConditionalRenderingInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("conditionalRenderingEnable", c_uint)
    ]

class VkViewportWScalingNV(Structure):
    pass
VkViewportWScalingNV._fields_ = [
             ("xcoeff", c_float),
             ("ycoeff", c_float)
    ]

class VkPipelineViewportWScalingStateCreateInfoNV(Structure):
    pass
VkPipelineViewportWScalingStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("viewportWScalingEnable", c_uint),
             ("viewportCount", c_uint),
             ("pVkViewportWScalingNViewportWScalings", VkViewportWScalingNV)
    ]

class VkSurfaceCapabilities2EXT(Structure):
    pass
VkSurfaceCapabilities2EXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("minImageCount", c_uint),
             ("maxImageCount", c_uint),
             ("currentExtent", VkExtent2D),
             ("minImageExtent", VkExtent2D),
             ("maxImageExtent", VkExtent2D),
             ("maxImageArrayLayers", c_uint),
             ("supportedTransforms", c_uint),
             ("currentTransform", c_int),
             ("supportedCompositeAlpha", c_uint),
             ("supportedUsageFlags", c_uint),
             ("supportedSurfaceCounters", c_uint)
    ]

class VkDisplayPowerInfoEXT(Structure):
    pass
VkDisplayPowerInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("powerState", c_int)
    ]

class VkDeviceEventInfoEXT(Structure):
    pass
VkDeviceEventInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceEvent", c_int)
    ]

class VkDisplayEventInfoEXT(Structure):
    pass
VkDisplayEventInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("displayEvent", c_int)
    ]

class VkSwapchainCounterCreateInfoEXT(Structure):
    pass
VkSwapchainCounterCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("surfaceCounters", c_uint)
    ]

class VkRefreshCycleDurationGOOGLE(Structure):
    pass
VkRefreshCycleDurationGOOGLE._fields_ = [
             ("refreshDuration", c_ulong)
    ]

class VkPastPresentationTimingGOOGLE(Structure):
    pass
VkPastPresentationTimingGOOGLE._fields_ = [
             ("presentID", c_uint),
             ("desiredPresentTime", c_ulong),
             ("actualPresentTime", c_ulong),
             ("earliestPresentTime", c_ulong),
             ("presentMargin", c_ulong)
    ]

class VkPresentTimeGOOGLE(Structure):
    pass
VkPresentTimeGOOGLE._fields_ = [
             ("presentID", c_uint),
             ("desiredPresentTime", c_ulong)
    ]

class VkPresentTimesInfoGOOGLE(Structure):
    pass
VkPresentTimesInfoGOOGLE._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("swapchainCount", c_uint),
             ("pTimes", VkPresentTimeGOOGLE)
    ]

class VkPhysicalDeviceMultiviewPerViewAttributesPropertiesNVX(Structure):
    pass
VkPhysicalDeviceMultiviewPerViewAttributesPropertiesNVX._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("perc_uintiewPositionAllComponents", c_uint)
    ]

class VkViewportSwizzleNV(Structure):
    pass
VkViewportSwizzleNV._fields_ = [
             ("x", c_int),
             ("y", c_int),
             ("z", c_int),
             ("w", c_int)
    ]

class VkPipelineViewportSwizzleStateCreateInfoNV(Structure):
    pass
VkPipelineViewportSwizzleStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("viewportCount", c_uint),
             ("pVkViewportSwizzleNViewportSwizzles", VkViewportSwizzleNV)
    ]

class VkPhysicalDeviceDiscardRectanglePropertiesEXT(Structure):
    pass
VkPhysicalDeviceDiscardRectanglePropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxDiscardRectangles", c_uint)
    ]

class VkPipelineDiscardRectangleStateCreateInfoEXT(Structure):
    pass
VkPipelineDiscardRectangleStateCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("discardRectangleMode", c_int),
             ("discardRectangleCount", c_uint),
             ("pDiscardRectangles", VkRect2D)
    ]

class VkPhysicalDeviceConservativeRasterizationPropertiesEXT(Structure):
    pass
VkPhysicalDeviceConservativeRasterizationPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("primitiveOverestimationSize", c_float),
             ("maxExtraPrimitiveOverestimationSize", c_float),
             ("extraPrimitiveOverestimationSizeGranularity", c_float),
             ("primitiveUnderestimation", c_uint),
             ("conservativePointAndLineRasterization", c_uint),
             ("degenerateTrianglesRasterized", c_uint),
             ("degenerateLinesRasterized", c_uint),
             ("fullyCoveredFragmentShaderInputc_uintariable", c_uint),
             ("conservativeRasterizationPostDepthCoverage", c_uint)
    ]

class VkPipelineRasterizationConservativeStateCreateInfoEXT(Structure):
    pass
VkPipelineRasterizationConservativeStateCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("conservativeRasterizationMode", c_int),
             ("extraPrimitiveOverestimationSize", c_float)
    ]

class VkPhysicalDeviceDepthClipEnableFeaturesEXT(Structure):
    pass
VkPhysicalDeviceDepthClipEnableFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("depthClipEnable", c_uint)
    ]

class VkPipelineRasterizationDepthClipStateCreateInfoEXT(Structure):
    pass
VkPipelineRasterizationDepthClipStateCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("depthClipEnable", c_uint)
    ]

class VkXYColorEXT(Structure):
    pass
VkXYColorEXT._fields_ = [
             ("x", c_float),
             ("y", c_float)
    ]

class VkHdrMetadataEXT(Structure):
    pass
VkHdrMetadataEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("displayPrimaryRed", VkXYColorEXT),
             ("displayPrimaryGreen", VkXYColorEXT),
             ("displayPrimaryBlue", VkXYColorEXT),
             ("whitePoint", VkXYColorEXT),
             ("maxLuminance", c_float),
             ("minLuminance", c_float),
             ("maxContentLightLevel", c_float),
             ("maxFrameAverageLightLevel", c_float)
    ]

class VkDebugUtilsLabelEXT(Structure):
    pass
VkDebugUtilsLabelEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pLabelName", c_char_p),
             ("color", c_float *4)
    ]

class VkDebugUtilsObjectNameInfoEXT(Structure):
    pass
VkDebugUtilsObjectNameInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("objectType", c_int),
             ("objectHandle", c_ulong),
             ("pObjectName", c_char_p)
    ]

class VkDebugUtilsMessengerCallbackDataEXT(Structure):
    pass
VkDebugUtilsMessengerCallbackDataEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("pMessageIdName", c_char_p),
             ("messageIdNumber", c_int),
             ("pMessage", c_char_p),
             ("queueLabelCount", c_uint),
             ("pQueueLabels", VkDebugUtilsLabelEXT),
             ("cmdBufLabelCount", c_uint),
             ("pCmdBufLabels", VkDebugUtilsLabelEXT),
             ("objectCount", c_uint),
             ("pObjects", VkDebugUtilsObjectNameInfoEXT)
    ]

class VkDebugUtilsMessengerCreateInfoEXT(Structure):
    pass
VkDebugUtilsMessengerCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("messageSeverity", c_uint),
             ("messageType", c_uint),
             ("pfnUserCallback", c_void_p),
             ("pUserData", c_void_p)
    ]

class VkDebugUtilsObjectTagInfoEXT(Structure):
    pass
VkDebugUtilsObjectTagInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("objectType", c_int),
             ("objectHandle", c_ulong),
             ("tagName", c_ulong),
             ("tagSize", c_ulong),
             ("pTag", c_void_p)
    ]

class VkSampleLocationEXT(Structure):
    pass
VkSampleLocationEXT._fields_ = [
             ("x", c_float),
             ("y", c_float)
    ]

class VkSampleLocationsInfoEXT(Structure):
    pass
VkSampleLocationsInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("sampleLocationsPerPixel", c_int),
             ("sampleLocationGridSize", VkExtent2D),
             ("sampleLocationsCount", c_uint),
             ("pSampleLocations", VkSampleLocationEXT)
    ]

class VkAttachmentSampleLocationsEXT(Structure):
    pass
VkAttachmentSampleLocationsEXT._fields_ = [
             ("attachmentIndex", c_uint),
             ("sampleLocationsInfo", VkSampleLocationsInfoEXT)
    ]

class VkSubpassSampleLocationsEXT(Structure):
    pass
VkSubpassSampleLocationsEXT._fields_ = [
             ("subpassIndex", c_uint),
             ("sampleLocationsInfo", VkSampleLocationsInfoEXT)
    ]

class VkRenderPassSampleLocationsBeginInfoEXT(Structure):
    pass
VkRenderPassSampleLocationsBeginInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("attachmentInitialSampleLocationsCount", c_uint),
             ("pAttachmentInitialSampleLocations", VkAttachmentSampleLocationsEXT),
             ("postSubpassSampleLocationsCount", c_uint),
             ("pPostSubpassSampleLocations", VkSubpassSampleLocationsEXT)
    ]

class VkPipelineSampleLocationsStateCreateInfoEXT(Structure):
    pass
VkPipelineSampleLocationsStateCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("sampleLocationsEnable", c_uint),
             ("sampleLocationsInfo", VkSampleLocationsInfoEXT)
    ]

class VkPhysicalDeviceSampleLocationsPropertiesEXT(Structure):
    pass
VkPhysicalDeviceSampleLocationsPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("sampleLocationSampleCounts", c_uint),
             ("maxSampleLocationGridSize", VkExtent2D),
             ("sampleLocationCoordinateRange", c_float *2),
             ("sampleLocationSubPixelBits", c_uint),
             ("variableSampleLocations", c_uint)
    ]

class VkMultisamplePropertiesEXT(Structure):
    pass
VkMultisamplePropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxSampleLocationGridSize", VkExtent2D)
    ]

class VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT(Structure):
    pass
VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("advancedBlendCoherentOperations", c_uint)
    ]

class VkPhysicalDeviceBlendOperationAdvancedPropertiesEXT(Structure):
    pass
VkPhysicalDeviceBlendOperationAdvancedPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("advancedBlendMaxColorAttachments", c_uint),
             ("advancedBlendIndependentBlend", c_uint),
             ("advancedBlendNonPremultipliedSrcColor", c_uint),
             ("advancedBlendNonPremultipliedDstColor", c_uint),
             ("advancedBlendCorrelatedOverlap", c_uint),
             ("advancedBlendAllOperations", c_uint)
    ]

class VkPipelineColorBlendAdvancedStateCreateInfoEXT(Structure):
    pass
VkPipelineColorBlendAdvancedStateCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("srcPremultiplied", c_uint),
             ("dstPremultiplied", c_uint),
             ("blendOverlap", c_int)
    ]

class VkPipelineCoverageToColorStateCreateInfoNV(Structure):
    pass
VkPipelineCoverageToColorStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("coverageToColorEnable", c_uint),
             ("coverageToColorLocation", c_uint)
    ]

class VkPipelineCoverageModulationStateCreateInfoNV(Structure):
    pass
VkPipelineCoverageModulationStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("coverageModulationMode", c_int),
             ("coverageModulationTableEnable", c_uint),
             ("coverageModulationTableCount", c_uint),
             ("pCoverageModulationTable", POINTER(c_float))
    ]

class VkPhysicalDeviceShaderSMBuiltinsPropertiesNV(Structure):
    pass
VkPhysicalDeviceShaderSMBuiltinsPropertiesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderSMCount", c_uint),
             ("shaderWarpsPerSM", c_uint)
    ]

class VkPhysicalDeviceShaderSMBuiltinsFeaturesNV(Structure):
    pass
VkPhysicalDeviceShaderSMBuiltinsFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderSMBuiltins", c_uint)
    ]

class VkDrmFormatModifierPropertiesEXT(Structure):
    pass
VkDrmFormatModifierPropertiesEXT._fields_ = [
             ("drmFormatModifier", c_ulong),
             ("drmFormatModifierPlaneCount", c_uint),
             ("drmFormatModifierTilingFeatures", c_uint)
    ]

class VkDrmFormatModifierPropertiesListEXT(Structure):
    pass
VkDrmFormatModifierPropertiesListEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("drmFormatModifierCount", c_uint),
             ("pDrmFormatModifierProperties", VkDrmFormatModifierPropertiesEXT)
    ]

class VkPhysicalDeviceImageDrmFormatModifierInfoEXT(Structure):
    pass
VkPhysicalDeviceImageDrmFormatModifierInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("drmFormatModifier", c_ulong),
             ("sharingMode", c_int),
             ("queueFamilyIndexCount", c_uint),
             ("pQueueFamilyIndices", POINTER(c_uint))
    ]

class VkImageDrmFormatModifierListCreateInfoEXT(Structure):
    pass
VkImageDrmFormatModifierListCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("drmFormatModifierCount", c_uint),
             ("pDrmFormatModifiers", POINTER(c_ulong))
    ]

class VkImageDrmFormatModifierExplicitCreateInfoEXT(Structure):
    pass
VkImageDrmFormatModifierExplicitCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("drmFormatModifier", c_ulong),
             ("drmFormatModifierPlaneCount", c_uint),
             ("pPlaneLayouts", VkSubresourceLayout)
    ]

class VkImageDrmFormatModifierPropertiesEXT(Structure):
    pass
VkImageDrmFormatModifierPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("drmFormatModifier", c_ulong)
    ]

class VkDrmFormatModifierProperties2EXT(Structure):
    pass
VkDrmFormatModifierProperties2EXT._fields_ = [
             ("drmFormatModifier", c_ulong),
             ("drmFormatModifierPlaneCount", c_uint),
             ("drmFormatModifierTilingFeatures", c_ulong)
    ]

class VkDrmFormatModifierPropertiesList2EXT(Structure):
    pass
VkDrmFormatModifierPropertiesList2EXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("drmFormatModifierCount", c_uint),
             ("pDrmFormatModifierProperties", VkDrmFormatModifierProperties2EXT)
    ]

class VkValidationCacheCreateInfoEXT(Structure):
    pass
VkValidationCacheCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("initialDataSize", c_ulong),
             ("pInitialData", c_void_p)
    ]

class VkShaderModuleValidationCacheCreateInfoEXT(Structure):
    pass
VkShaderModuleValidationCacheCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("validationCache", VkValidationCacheEXT_T)
    ]

class VkShadingRatePaletteNV(Structure):
    pass
VkShadingRatePaletteNV._fields_ = [
             ("shadingRatePaletteEntryCount", c_uint),
             ("pShadingRatePaletteEntries", POINTER(c_int))
    ]

class VkPipelineViewportShadingRateImageStateCreateInfoNV(Structure):
    pass
VkPipelineViewportShadingRateImageStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shadingRateImageEnable", c_uint),
             ("viewportCount", c_uint),
             ("pShadingRatePalettes", VkShadingRatePaletteNV)
    ]

class VkPhysicalDeviceShadingRateImageFeaturesNV(Structure):
    pass
VkPhysicalDeviceShadingRateImageFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shadingRateImage", c_uint),
             ("shadingRateCoarseSampleOrder", c_uint)
    ]

class VkPhysicalDeviceShadingRateImagePropertiesNV(Structure):
    pass
VkPhysicalDeviceShadingRateImagePropertiesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shadingRateTexelSize", VkExtent2D),
             ("shadingRatePaletteSize", c_uint),
             ("shadingRateMaxCoarseSamples", c_uint)
    ]

class VkCoarseSampleLocationNV(Structure):
    pass
VkCoarseSampleLocationNV._fields_ = [
             ("pixelX", c_uint),
             ("pixelY", c_uint),
             ("sample", c_uint)
    ]

class VkCoarseSampleOrderCustomNV(Structure):
    pass
VkCoarseSampleOrderCustomNV._fields_ = [
             ("shadingRate", c_int),
             ("sampleCount", c_uint),
             ("sampleLocationCount", c_uint),
             ("pSampleLocations", VkCoarseSampleLocationNV)
    ]

class VkPipelineViewportCoarseSampleOrderStateCreateInfoNV(Structure):
    pass
VkPipelineViewportCoarseSampleOrderStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("sampleOrderType", c_int),
             ("customSampleOrderCount", c_uint),
             ("pCustomSampleOrders", VkCoarseSampleOrderCustomNV)
    ]

class VkRayTracingShaderGroupCreateInfoNV(Structure):
    pass
VkRayTracingShaderGroupCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("type", c_int),
             ("generalShader", c_uint),
             ("closestHitShader", c_uint),
             ("anyHitShader", c_uint),
             ("intersectionShader", c_uint)
    ]

class VkRayTracingPipelineCreateInfoNV(Structure):
    pass
VkRayTracingPipelineCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("stageCount", c_uint),
             ("pStages", VkPipelineShaderStageCreateInfo),
             ("groupCount", c_uint),
             ("pGroups", VkRayTracingShaderGroupCreateInfoNV),
             ("maxRecursionDepth", c_uint),
             ("layout", VkPipelineLayout_T),
             ("basePipelineHandle", VkPipeline_T),
             ("basePipelineIndex", c_int)
    ]

class VkGeometryTrianglesNV(Structure):
    pass
VkGeometryTrianglesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("vertexData", VkBuffer_T),
             ("vertexOffset", c_ulong),
             ("vertexCount", c_uint),
             ("vertexStride", c_ulong),
             ("vertexFormat", c_int),
             ("indexData", VkBuffer_T),
             ("indexOffset", c_ulong),
             ("indexCount", c_uint),
             ("indexType", c_int),
             ("transformData", VkBuffer_T),
             ("transformOffset", c_ulong)
    ]

class VkGeometryAABBNV(Structure):
    pass
VkGeometryAABBNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("aabbData", VkBuffer_T),
             ("numAABBs", c_uint),
             ("stride", c_uint),
             ("offset", c_ulong)
    ]

class VkGeometryDataNV(Structure):
    pass
VkGeometryDataNV._fields_ = [
             ("triangles", VkGeometryTrianglesNV),
             ("aabbs", VkGeometryAABBNV)
    ]

class VkGeometryNV(Structure):
    pass
VkGeometryNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("geometryType", c_int),
             ("geometry", VkGeometryDataNV),
             ("flags", c_uint)
    ]

class VkAccelerationStructureInfoNV(Structure):
    pass
VkAccelerationStructureInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("type", c_int),
             ("flags", c_uint),
             ("instanceCount", c_uint),
             ("geometryCount", c_uint),
             ("pGeometries", VkGeometryNV)
    ]

class VkAccelerationStructureCreateInfoNV(Structure):
    pass
VkAccelerationStructureCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("compactedSize", c_ulong),
             ("info", VkAccelerationStructureInfoNV)
    ]

class VkBindAccelerationStructureMemoryInfoNV(Structure):
    pass
VkBindAccelerationStructureMemoryInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("accelerationStructure", VkAccelerationStructureNV_T),
             ("memory", VkDeviceMemory_T),
             ("memoryOffset", c_ulong),
             ("deviceIndexCount", c_uint),
             ("pDeviceIndices", POINTER(c_uint))
    ]

class VkWriteDescriptorSetAccelerationStructureNV(Structure):
    pass
VkWriteDescriptorSetAccelerationStructureNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("accelerationStructureCount", c_uint),
             ("pAccelerationStructures", POINTER(VkAccelerationStructureNV_T))
    ]

class VkAccelerationStructureMemoryRequirementsInfoNV(Structure):
    pass
VkAccelerationStructureMemoryRequirementsInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("type", c_int),
             ("accelerationStructure", VkAccelerationStructureNV_T)
    ]

class VkPhysicalDeviceRayTracingPropertiesNV(Structure):
    pass
VkPhysicalDeviceRayTracingPropertiesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderGroupHandleSize", c_uint),
             ("maxRecursionDepth", c_uint),
             ("maxShaderGroupStride", c_uint),
             ("shaderGroupBaseAlignment", c_uint),
             ("maxGeometryCount", c_ulong),
             ("maxInstanceCount", c_ulong),
             ("maxTriangleCount", c_ulong),
             ("maxDescriptorSetAccelerationStructures", c_uint)
    ]

class VkTransformMatrixKHR(Structure):
    pass
VkTransformMatrixKHR._fields_ = [
             ("matrix", c_float *9)
    ]

class VkAabbPositionsKHR(Structure):
    pass
VkAabbPositionsKHR._fields_ = [
             ("minX", c_float),
             ("minY", c_float),
             ("minZ", c_float),
             ("maxX", c_float),
             ("maxY", c_float),
             ("maxZ", c_float)
    ]

class VkAccelerationStructureInstanceKHR(Structure):
    pass
VkAccelerationStructureInstanceKHR._fields_ = [
             ("transform", VkTransformMatrixKHR),
             ("instanceCustomIndex", c_uint),
             ("mask", c_uint),
             ("instanceShaderBindingTableRecordOffset", c_uint),
             ("flags", c_uint),
             ("accelerationStructureReference", c_ulong)
    ]

class VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV(Structure):
    pass
VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("representativeFragmentTest", c_uint)
    ]

class VkPipelineRepresentativeFragmentTestStateCreateInfoNV(Structure):
    pass
VkPipelineRepresentativeFragmentTestStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("representativeFragmentTestEnable", c_uint)
    ]

class VkPhysicalDeviceImageViewImageFormatInfoEXT(Structure):
    pass
VkPhysicalDeviceImageViewImageFormatInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("imagec_intiewType", c_int)
    ]

class VkFilterCubicImageViewImageFormatPropertiesEXT(Structure):
    pass
VkFilterCubicImageViewImageFormatPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("filterCubic", c_uint),
             ("filterCubicMinmax", c_uint)
    ]

class VkImportMemoryHostPointerInfoEXT(Structure):
    pass
VkImportMemoryHostPointerInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("handleType", c_int),
             ("pHostPointer", c_void_p)
    ]

class VkMemoryHostPointerPropertiesEXT(Structure):
    pass
VkMemoryHostPointerPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memoryTypeBits", c_uint)
    ]

class VkPhysicalDeviceExternalMemoryHostPropertiesEXT(Structure):
    pass
VkPhysicalDeviceExternalMemoryHostPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("minImportedHostPointerAlignment", c_ulong)
    ]

class VkPipelineCompilerControlCreateInfoAMD(Structure):
    pass
VkPipelineCompilerControlCreateInfoAMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("compilerControlFlags", c_uint)
    ]

class VkCalibratedTimestampInfoEXT(Structure):
    pass
VkCalibratedTimestampInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("timeDomain", c_int)
    ]

class VkPhysicalDeviceShaderCorePropertiesAMD(Structure):
    pass
VkPhysicalDeviceShaderCorePropertiesAMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderEngineCount", c_uint),
             ("shaderArraysPerEngineCount", c_uint),
             ("computeUnitsPerShaderArray", c_uint),
             ("simdPerComputeUnit", c_uint),
             ("wavefrontsPerSimd", c_uint),
             ("wavefrontSize", c_uint),
             ("sgprsPerSimd", c_uint),
             ("minSgprAllocation", c_uint),
             ("maxSgprAllocation", c_uint),
             ("sgprAllocationGranularity", c_uint),
             ("vgprsPerSimd", c_uint),
             ("minc_uintgprAllocation", c_uint),
             ("maxc_uintgprAllocation", c_uint),
             ("vgprAllocationGranularity", c_uint)
    ]

class VkDeviceMemoryOverallocationCreateInfoAMD(Structure):
    pass
VkDeviceMemoryOverallocationCreateInfoAMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("overallocationBehavior", c_int)
    ]

class VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT(Structure):
    pass
VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxc_uintertexAttribDivisor", c_uint)
    ]

class VkVertexInputBindingDivisorDescriptionEXT(Structure):
    pass
VkVertexInputBindingDivisorDescriptionEXT._fields_ = [
             ("binding", c_uint),
             ("divisor", c_uint)
    ]

class VkPipelineVertexInputDivisorStateCreateInfoEXT(Structure):
    pass
VkPipelineVertexInputDivisorStateCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("vertexBindingDivisorCount", c_uint),
             ("pVkVertexInputBindingDivisorDescriptionEXTertexBindingDivisors", VkVertexInputBindingDivisorDescriptionEXT)
    ]

class VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT(Structure):
    pass
VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("vertexAttributeInstanceRateDivisor", c_uint),
             ("vertexAttributeInstanceRateZeroDivisor", c_uint)
    ]

class VkPhysicalDeviceComputeShaderDerivativesFeaturesNV(Structure):
    pass
VkPhysicalDeviceComputeShaderDerivativesFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("computeDerivativeGroupQuads", c_uint),
             ("computeDerivativeGroupLinear", c_uint)
    ]

class VkPhysicalDeviceMeshShaderFeaturesNV(Structure):
    pass
VkPhysicalDeviceMeshShaderFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("taskShader", c_uint),
             ("meshShader", c_uint)
    ]

class VkPhysicalDeviceMeshShaderPropertiesNV(Structure):
    pass
VkPhysicalDeviceMeshShaderPropertiesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxDrawMeshTasksCount", c_uint),
             ("maxTaskWorkGroupInvocations", c_uint),
             ("maxTaskWorkGroupSize", c_uint *3),
             ("maxTaskTotalMemorySize", c_uint),
             ("maxTaskOutputCount", c_uint),
             ("maxMeshWorkGroupInvocations", c_uint),
             ("maxMeshWorkGroupSize", c_uint *3),
             ("maxMeshTotalMemorySize", c_uint),
             ("maxMeshOutputc_uintertices", c_uint),
             ("maxMeshOutputPrimitives", c_uint),
             ("maxMeshMultiviewc_uintiewCount", c_uint),
             ("meshOutputPerc_uintertexGranularity", c_uint),
             ("meshOutputPerPrimitiveGranularity", c_uint)
    ]

class VkDrawMeshTasksIndirectCommandNV(Structure):
    pass
VkDrawMeshTasksIndirectCommandNV._fields_ = [
             ("taskCount", c_uint),
             ("firstTask", c_uint)
    ]

class VkPhysicalDeviceFragmentShaderBarycentricFeaturesNV(Structure):
    pass
VkPhysicalDeviceFragmentShaderBarycentricFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentShaderBarycentric", c_uint)
    ]

class VkPhysicalDeviceShaderImageFootprintFeaturesNV(Structure):
    pass
VkPhysicalDeviceShaderImageFootprintFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("imageFootprint", c_uint)
    ]

class VkPipelineViewportExclusiveScissorStateCreateInfoNV(Structure):
    pass
VkPipelineViewportExclusiveScissorStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("exclusiveScissorCount", c_uint),
             ("pExclusiveScissors", VkRect2D)
    ]

class VkPhysicalDeviceExclusiveScissorFeaturesNV(Structure):
    pass
VkPhysicalDeviceExclusiveScissorFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("exclusiveScissor", c_uint)
    ]

class VkQueueFamilyCheckpointPropertiesNV(Structure):
    pass
VkQueueFamilyCheckpointPropertiesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("checkpointExecutionStageMask", c_uint)
    ]

class VkCheckpointDataNV(Structure):
    pass
VkCheckpointDataNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("stage", c_int),
             ("pCheckpointMarker", c_void_p)
    ]

class VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL(Structure):
    pass
VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderIntegerFunctions2", c_uint)
    ]

class VkPerformanceValueDataINTEL(Structure):
    pass
VkPerformanceValueDataINTEL._fields_ = [
             ("value32", c_uint),
             ("value64", c_ulong),
             ("valueFloat", c_float),
             ("valueBool", c_uint),
             ("valueString", c_char_p)
    ]

class VkPerformanceValueINTEL(Structure):
    pass
VkPerformanceValueINTEL._fields_ = [
             ("type", c_int),
             ("data", VkPerformanceValueDataINTEL)
    ]

class VkInitializePerformanceApiInfoINTEL(Structure):
    pass
VkInitializePerformanceApiInfoINTEL._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pUserData", c_void_p)
    ]

class VkQueryPoolPerformanceQueryCreateInfoINTEL(Structure):
    pass
VkQueryPoolPerformanceQueryCreateInfoINTEL._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("performanceCountersSampling", c_int)
    ]

class VkPerformanceMarkerInfoINTEL(Structure):
    pass
VkPerformanceMarkerInfoINTEL._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("marker", c_ulong)
    ]

class VkPerformanceStreamMarkerInfoINTEL(Structure):
    pass
VkPerformanceStreamMarkerInfoINTEL._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("marker", c_uint)
    ]

class VkPerformanceOverrideInfoINTEL(Structure):
    pass
VkPerformanceOverrideInfoINTEL._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("type", c_int),
             ("enable", c_uint),
             ("parameter", c_ulong)
    ]

class VkPerformanceConfigurationAcquireInfoINTEL(Structure):
    pass
VkPerformanceConfigurationAcquireInfoINTEL._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("type", c_int)
    ]

class VkPhysicalDevicePCIBusInfoPropertiesEXT(Structure):
    pass
VkPhysicalDevicePCIBusInfoPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pciDomain", c_uint),
             ("pciBus", c_uint),
             ("pciDevice", c_uint),
             ("pciFunction", c_uint)
    ]

class VkDisplayNativeHdrSurfaceCapabilitiesAMD(Structure):
    pass
VkDisplayNativeHdrSurfaceCapabilitiesAMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("localDimmingSupport", c_uint)
    ]

class VkSwapchainDisplayNativeHdrCreateInfoAMD(Structure):
    pass
VkSwapchainDisplayNativeHdrCreateInfoAMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("localDimmingEnable", c_uint)
    ]

class VkPhysicalDeviceFragmentDensityMapFeaturesEXT(Structure):
    pass
VkPhysicalDeviceFragmentDensityMapFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentDensityMap", c_uint),
             ("fragmentDensityMapDynamic", c_uint),
             ("fragmentDensityMapNonSubsampledImages", c_uint)
    ]

class VkPhysicalDeviceFragmentDensityMapPropertiesEXT(Structure):
    pass
VkPhysicalDeviceFragmentDensityMapPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("minFragmentDensityTexelSize", VkExtent2D),
             ("maxFragmentDensityTexelSize", VkExtent2D),
             ("fragmentDensityInvocations", c_uint)
    ]

class VkRenderPassFragmentDensityMapCreateInfoEXT(Structure):
    pass
VkRenderPassFragmentDensityMapCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentDensityMapAttachment", VkAttachmentReference)
    ]

class VkPhysicalDeviceShaderCoreProperties2AMD(Structure):
    pass
VkPhysicalDeviceShaderCoreProperties2AMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderCoreFeatures", c_uint),
             ("activeComputeUnitCount", c_uint)
    ]

class VkPhysicalDeviceCoherentMemoryFeaturesAMD(Structure):
    pass
VkPhysicalDeviceCoherentMemoryFeaturesAMD._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceCoherentMemory", c_uint)
    ]

class VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT(Structure):
    pass
VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderImageInt64Atomics", c_uint),
             ("sparseImageInt64Atomics", c_uint)
    ]

class VkPhysicalDeviceMemoryBudgetPropertiesEXT(Structure):
    pass
VkPhysicalDeviceMemoryBudgetPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("heapBudget", c_ulong *16),
             ("heapUsage", c_ulong *16)
    ]

class VkPhysicalDeviceMemoryPriorityFeaturesEXT(Structure):
    pass
VkPhysicalDeviceMemoryPriorityFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memoryPriority", c_uint)
    ]

class VkMemoryPriorityAllocateInfoEXT(Structure):
    pass
VkMemoryPriorityAllocateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("priority", c_float)
    ]

class VkPhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV(Structure):
    pass
VkPhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("dedicatedAllocationImageAliasing", c_uint)
    ]

class VkPhysicalDeviceBufferDeviceAddressFeaturesEXT(Structure):
    pass
VkPhysicalDeviceBufferDeviceAddressFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("bufferDeviceAddress", c_uint),
             ("bufferDeviceAddressCaptureReplay", c_uint),
             ("bufferDeviceAddressMultiDevice", c_uint)
    ]

class VkBufferDeviceAddressCreateInfoEXT(Structure):
    pass
VkBufferDeviceAddressCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceAddress", c_ulong)
    ]

class VkValidationFeaturesEXT(Structure):
    pass
VkValidationFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("enabledc_uintalidationFeatureCount", c_uint),
             ("pEnabledPOINTER(c_int)alidationFeatures", POINTER(c_int)),
             ("disabledc_uintalidationFeatureCount", c_uint),
             ("pDisabledPOINTER(c_int)alidationFeatures", POINTER(c_int))
    ]

class VkCooperativeMatrixPropertiesNV(Structure):
    pass
VkCooperativeMatrixPropertiesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("MSize", c_uint),
             ("NSize", c_uint),
             ("KSize", c_uint),
             ("AType", c_int),
             ("BType", c_int),
             ("CType", c_int),
             ("DType", c_int),
             ("scope", c_int)
    ]

class VkPhysicalDeviceCooperativeMatrixFeaturesNV(Structure):
    pass
VkPhysicalDeviceCooperativeMatrixFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("cooperativeMatrix", c_uint),
             ("cooperativeMatrixRobustBufferAccess", c_uint)
    ]

class VkPhysicalDeviceCooperativeMatrixPropertiesNV(Structure):
    pass
VkPhysicalDeviceCooperativeMatrixPropertiesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("cooperativeMatrixSupportedStages", c_uint)
    ]

class VkPhysicalDeviceCoverageReductionModeFeaturesNV(Structure):
    pass
VkPhysicalDeviceCoverageReductionModeFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("coverageReductionMode", c_uint)
    ]

class VkPipelineCoverageReductionStateCreateInfoNV(Structure):
    pass
VkPipelineCoverageReductionStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("coverageReductionMode", c_int)
    ]

class VkFramebufferMixedSamplesCombinationNV(Structure):
    pass
VkFramebufferMixedSamplesCombinationNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("coverageReductionMode", c_int),
             ("rasterizationSamples", c_int),
             ("depthStencilSamples", c_uint),
             ("colorSamples", c_uint)
    ]

class VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT(Structure):
    pass
VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentShaderSampleInterlock", c_uint),
             ("fragmentShaderPixelInterlock", c_uint),
             ("fragmentShaderShadingRateInterlock", c_uint)
    ]

class VkPhysicalDeviceYcbcrImageArraysFeaturesEXT(Structure):
    pass
VkPhysicalDeviceYcbcrImageArraysFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("ycbcrImageArrays", c_uint)
    ]

class VkPhysicalDeviceProvokingVertexFeaturesEXT(Structure):
    pass
VkPhysicalDeviceProvokingVertexFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("provokingc_uintertexLast", c_uint),
             ("transformFeedbackPreservesProvokingc_uintertex", c_uint)
    ]

class VkPhysicalDeviceProvokingVertexPropertiesEXT(Structure):
    pass
VkPhysicalDeviceProvokingVertexPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("provokingc_uintertexModePerPipeline", c_uint),
             ("transformFeedbackPreservesTriangleFanProvokingc_uintertex", c_uint)
    ]

class VkPipelineRasterizationProvokingVertexStateCreateInfoEXT(Structure):
    pass
VkPipelineRasterizationProvokingVertexStateCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("provokingc_intertexMode", c_int)
    ]

class VkHeadlessSurfaceCreateInfoEXT(Structure):
    pass
VkHeadlessSurfaceCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint)
    ]

class VkPhysicalDeviceLineRasterizationFeaturesEXT(Structure):
    pass
VkPhysicalDeviceLineRasterizationFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("rectangularLines", c_uint),
             ("bresenhamLines", c_uint),
             ("smoothLines", c_uint),
             ("stippledRectangularLines", c_uint),
             ("stippledBresenhamLines", c_uint),
             ("stippledSmoothLines", c_uint)
    ]

class VkPhysicalDeviceLineRasterizationPropertiesEXT(Structure):
    pass
VkPhysicalDeviceLineRasterizationPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("lineSubPixelPrecisionBits", c_uint)
    ]

class VkPipelineRasterizationLineStateCreateInfoEXT(Structure):
    pass
VkPipelineRasterizationLineStateCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("lineRasterizationMode", c_int),
             ("stippledLineEnable", c_uint),
             ("lineStippleFactor", c_uint),
             ("lineStipplePattern", c_ushort)
    ]

class VkPhysicalDeviceShaderAtomicFloatFeaturesEXT(Structure):
    pass
VkPhysicalDeviceShaderAtomicFloatFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderBufferFloat32Atomics", c_uint),
             ("shaderBufferFloat32AtomicAdd", c_uint),
             ("shaderBufferFloat64Atomics", c_uint),
             ("shaderBufferFloat64AtomicAdd", c_uint),
             ("shaderSharedFloat32Atomics", c_uint),
             ("shaderSharedFloat32AtomicAdd", c_uint),
             ("shaderSharedFloat64Atomics", c_uint),
             ("shaderSharedFloat64AtomicAdd", c_uint),
             ("shaderImageFloat32Atomics", c_uint),
             ("shaderImageFloat32AtomicAdd", c_uint),
             ("sparseImageFloat32Atomics", c_uint),
             ("sparseImageFloat32AtomicAdd", c_uint)
    ]

class VkPhysicalDeviceIndexTypeUint8FeaturesEXT(Structure):
    pass
VkPhysicalDeviceIndexTypeUint8FeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("indexTypeUint8", c_uint)
    ]

class VkPhysicalDeviceExtendedDynamicStateFeaturesEXT(Structure):
    pass
VkPhysicalDeviceExtendedDynamicStateFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("extendedDynamicState", c_uint)
    ]

class VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT(Structure):
    pass
VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderBufferFloat16Atomics", c_uint),
             ("shaderBufferFloat16AtomicAdd", c_uint),
             ("shaderBufferFloat16AtomicMinMax", c_uint),
             ("shaderBufferFloat32AtomicMinMax", c_uint),
             ("shaderBufferFloat64AtomicMinMax", c_uint),
             ("shaderSharedFloat16Atomics", c_uint),
             ("shaderSharedFloat16AtomicAdd", c_uint),
             ("shaderSharedFloat16AtomicMinMax", c_uint),
             ("shaderSharedFloat32AtomicMinMax", c_uint),
             ("shaderSharedFloat64AtomicMinMax", c_uint),
             ("shaderImageFloat32AtomicMinMax", c_uint),
             ("sparseImageFloat32AtomicMinMax", c_uint)
    ]

class VkPhysicalDeviceDeviceGeneratedCommandsPropertiesNV(Structure):
    pass
VkPhysicalDeviceDeviceGeneratedCommandsPropertiesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxGraphicsShaderGroupCount", c_uint),
             ("maxIndirectSequenceCount", c_uint),
             ("maxIndirectCommandsTokenCount", c_uint),
             ("maxIndirectCommandsStreamCount", c_uint),
             ("maxIndirectCommandsTokenOffset", c_uint),
             ("maxIndirectCommandsStreamStride", c_uint),
             ("minSequencesCountBufferOffsetAlignment", c_uint),
             ("minSequencesIndexBufferOffsetAlignment", c_uint),
             ("minIndirectCommandsBufferOffsetAlignment", c_uint)
    ]

class VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV(Structure):
    pass
VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceGeneratedCommands", c_uint)
    ]

class VkGraphicsShaderGroupCreateInfoNV(Structure):
    pass
VkGraphicsShaderGroupCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("stageCount", c_uint),
             ("pStages", VkPipelineShaderStageCreateInfo),
             ("pVkPipelineVertexInputStateCreateInfoertexInputState", VkPipelineVertexInputStateCreateInfo),
             ("pTessellationState", VkPipelineTessellationStateCreateInfo)
    ]

class VkGraphicsPipelineShaderGroupsCreateInfoNV(Structure):
    pass
VkGraphicsPipelineShaderGroupsCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("groupCount", c_uint),
             ("pGroups", VkGraphicsShaderGroupCreateInfoNV),
             ("pipelineCount", c_uint),
             ("pPipelines", POINTER(VkPipeline_T))
    ]

class VkBindShaderGroupIndirectCommandNV(Structure):
    pass
VkBindShaderGroupIndirectCommandNV._fields_ = [
             ("groupIndex", c_uint)
    ]

class VkBindIndexBufferIndirectCommandNV(Structure):
    pass
VkBindIndexBufferIndirectCommandNV._fields_ = [
             ("bufferAddress", c_ulong),
             ("size", c_uint),
             ("indexType", c_int)
    ]

class VkBindVertexBufferIndirectCommandNV(Structure):
    pass
VkBindVertexBufferIndirectCommandNV._fields_ = [
             ("bufferAddress", c_ulong),
             ("size", c_uint),
             ("stride", c_uint)
    ]

class VkSetStateFlagsIndirectCommandNV(Structure):
    pass
VkSetStateFlagsIndirectCommandNV._fields_ = [
             ("data", c_uint)
    ]

class VkIndirectCommandsStreamNV(Structure):
    pass
VkIndirectCommandsStreamNV._fields_ = [
             ("buffer", VkBuffer_T),
             ("offset", c_ulong)
    ]

class VkIndirectCommandsLayoutTokenNV(Structure):
    pass
VkIndirectCommandsLayoutTokenNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("tokenType", c_int),
             ("stream", c_uint),
             ("offset", c_uint),
             ("vertexBindingUnit", c_uint),
             ("vertexDynamicStride", c_uint),
             ("pushconstantPipelineLayout", VkPipelineLayout_T),
             ("pushconstantShaderStageFlags", c_uint),
             ("pushconstantOffset", c_uint),
             ("pushconstantSize", c_uint),
             ("indirectStateFlags", c_uint),
             ("indexTypeCount", c_uint),
             ("pIndexTypes", POINTER(c_int)),
             ("pIndexTypePOINTER(c_uint)alues", POINTER(c_uint))
    ]

class VkIndirectCommandsLayoutCreateInfoNV(Structure):
    pass
VkIndirectCommandsLayoutCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("pipelineBindPoint", c_int),
             ("tokenCount", c_uint),
             ("pTokens", VkIndirectCommandsLayoutTokenNV),
             ("streamCount", c_uint),
             ("pStreamStrides", POINTER(c_uint))
    ]

class VkGeneratedCommandsInfoNV(Structure):
    pass
VkGeneratedCommandsInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pipelineBindPoint", c_int),
             ("pipeline", VkPipeline_T),
             ("indirectCommandsLayout", VkIndirectCommandsLayoutNV_T),
             ("streamCount", c_uint),
             ("pStreams", VkIndirectCommandsStreamNV),
             ("sequencesCount", c_uint),
             ("preprocessBuffer", VkBuffer_T),
             ("preprocessOffset", c_ulong),
             ("preprocessSize", c_ulong),
             ("sequencesCountBuffer", VkBuffer_T),
             ("sequencesCountOffset", c_ulong),
             ("sequencesIndexBuffer", VkBuffer_T),
             ("sequencesIndexOffset", c_ulong)
    ]

class VkGeneratedCommandsMemoryRequirementsInfoNV(Structure):
    pass
VkGeneratedCommandsMemoryRequirementsInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pipelineBindPoint", c_int),
             ("pipeline", VkPipeline_T),
             ("indirectCommandsLayout", VkIndirectCommandsLayoutNV_T),
             ("maxSequencesCount", c_uint)
    ]

class VkPhysicalDeviceInheritedViewportScissorFeaturesNV(Structure):
    pass
VkPhysicalDeviceInheritedViewportScissorFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("inheritedc_uintiewportScissor2D", c_uint)
    ]

class VkCommandBufferInheritanceViewportScissorInfoNV(Structure):
    pass
VkCommandBufferInheritanceViewportScissorInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("viewportScissor2D", c_uint),
             ("viewportDepthCount", c_uint),
             ("pVkViewportiewportDepths", VkViewport)
    ]

class VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT(Structure):
    pass
VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("texelBufferAlignment", c_uint)
    ]

class VkRenderPassTransformBeginInfoQCOM(Structure):
    pass
VkRenderPassTransformBeginInfoQCOM._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("transform", c_int)
    ]

class VkCommandBufferInheritanceRenderPassTransformInfoQCOM(Structure):
    pass
VkCommandBufferInheritanceRenderPassTransformInfoQCOM._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("transform", c_int),
             ("renderArea", VkRect2D)
    ]

class VkPhysicalDeviceDeviceMemoryReportFeaturesEXT(Structure):
    pass
VkPhysicalDeviceDeviceMemoryReportFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("deviceMemoryReport", c_uint)
    ]

class VkDeviceMemoryReportCallbackDataEXT(Structure):
    pass
VkDeviceMemoryReportCallbackDataEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("type", c_int),
             ("memoryObjectId", c_ulong),
             ("size", c_ulong),
             ("objectType", c_int),
             ("objectHandle", c_ulong),
             ("heapIndex", c_uint)
    ]

class VkDeviceDeviceMemoryReportCreateInfoEXT(Structure):
    pass
VkDeviceDeviceMemoryReportCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("pfnUserCallback", c_void_p),
             ("pUserData", c_void_p)
    ]

class VkPhysicalDeviceRobustness2FeaturesEXT(Structure):
    pass
VkPhysicalDeviceRobustness2FeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("robustBufferAccess2", c_uint),
             ("robustImageAccess2", c_uint),
             ("nullDescriptor", c_uint)
    ]

class VkPhysicalDeviceRobustness2PropertiesEXT(Structure):
    pass
VkPhysicalDeviceRobustness2PropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("robustStorageBufferAccessSizeAlignment", c_ulong),
             ("robustUniformBufferAccessSizeAlignment", c_ulong)
    ]

class VkSamplerCustomBorderColorCreateInfoEXT(Structure):
    pass
VkSamplerCustomBorderColorCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("customBorderColor", VkClearColorValue),
             ("format", c_int)
    ]

class VkPhysicalDeviceCustomBorderColorPropertiesEXT(Structure):
    pass
VkPhysicalDeviceCustomBorderColorPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxCustomBorderColorSamplers", c_uint)
    ]

class VkPhysicalDeviceCustomBorderColorFeaturesEXT(Structure):
    pass
VkPhysicalDeviceCustomBorderColorFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("customBorderColors", c_uint),
             ("customBorderColorWithoutFormat", c_uint)
    ]

class VkPhysicalDeviceDiagnosticsConfigFeaturesNV(Structure):
    pass
VkPhysicalDeviceDiagnosticsConfigFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("diagnosticsConfig", c_uint)
    ]

class VkDeviceDiagnosticsConfigCreateInfoNV(Structure):
    pass
VkDeviceDiagnosticsConfigCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint)
    ]

class VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT(Structure):
    pass
VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("graphicsPipelineLibrary", c_uint)
    ]

class VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT(Structure):
    pass
VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("graphicsPipelineLibraryFastLinking", c_uint),
             ("graphicsPipelineLibraryIndependentInterpolationDecoration", c_uint)
    ]

class VkGraphicsPipelineLibraryCreateInfoEXT(Structure):
    pass
VkGraphicsPipelineLibraryCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint)
    ]

class VkPhysicalDeviceFragmentShadingRateEnumsFeaturesNV(Structure):
    pass
VkPhysicalDeviceFragmentShadingRateEnumsFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentShadingRateEnums", c_uint),
             ("supersampleFragmentShadingRates", c_uint),
             ("noInvocationFragmentShadingRates", c_uint)
    ]

class VkPhysicalDeviceFragmentShadingRateEnumsPropertiesNV(Structure):
    pass
VkPhysicalDeviceFragmentShadingRateEnumsPropertiesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxFragmentShadingRateInvocationCount", c_int)
    ]

class VkPipelineFragmentShadingRateEnumStateCreateInfoNV(Structure):
    pass
VkPipelineFragmentShadingRateEnumStateCreateInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shadingRateType", c_int),
             ("shadingRate", c_int),
             ("combinerOps", c_int *2)
    ]

class VkDeviceOrHostAddressConstKHR(Structure):
    pass
VkDeviceOrHostAddressConstKHR._fields_ = [
             ("deviceAddress", c_ulong),
             ("hostAddress", c_void_p)
    ]

class VkAccelerationStructureGeometryMotionTrianglesDataNV(Structure):
    pass
VkAccelerationStructureGeometryMotionTrianglesDataNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("vertexData", VkDeviceOrHostAddressConstKHR)
    ]

class VkAccelerationStructureMotionInfoNV(Structure):
    pass
VkAccelerationStructureMotionInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxInstances", c_uint),
             ("flags", c_uint)
    ]

class VkAccelerationStructureMatrixMotionInstanceNV(Structure):
    pass
VkAccelerationStructureMatrixMotionInstanceNV._fields_ = [
             ("transformT0", VkTransformMatrixKHR),
             ("transformT1", VkTransformMatrixKHR),
             ("instanceCustomIndex", c_uint),
             ("mask", c_uint),
             ("instanceShaderBindingTableRecordOffset", c_uint),
             ("flags", c_uint),
             ("accelerationStructureReference", c_ulong)
    ]

class VkSRTDataNV(Structure):
    pass
VkSRTDataNV._fields_ = [
             ("sx", c_float),
             ("a", c_float),
             ("b", c_float),
             ("pvx", c_float),
             ("sy", c_float),
             ("c", c_float),
             ("pvy", c_float),
             ("sz", c_float),
             ("pvz", c_float),
             ("qx", c_float),
             ("qy", c_float),
             ("qz", c_float),
             ("qw", c_float),
             ("tx", c_float),
             ("ty", c_float),
             ("tz", c_float)
    ]

class VkAccelerationStructureSRTMotionInstanceNV(Structure):
    pass
VkAccelerationStructureSRTMotionInstanceNV._fields_ = [
             ("transformT0", VkSRTDataNV),
             ("transformT1", VkSRTDataNV),
             ("instanceCustomIndex", c_uint),
             ("mask", c_uint),
             ("instanceShaderBindingTableRecordOffset", c_uint),
             ("flags", c_uint),
             ("accelerationStructureReference", c_ulong)
    ]

class VkAccelerationStructureMotionInstanceDataNV(Structure):
    pass
VkAccelerationStructureMotionInstanceDataNV._fields_ = [
             ("staticInstance", VkAccelerationStructureInstanceKHR),
             ("matrixMotionInstance", VkAccelerationStructureMatrixMotionInstanceNV),
             ("srtMotionInstance", VkAccelerationStructureSRTMotionInstanceNV)
    ]

class VkAccelerationStructureMotionInstanceNV(Structure):
    pass
VkAccelerationStructureMotionInstanceNV._fields_ = [
             ("type", c_int),
             ("flags", c_uint),
             ("data", VkAccelerationStructureMotionInstanceDataNV)
    ]

class VkPhysicalDeviceRayTracingMotionBlurFeaturesNV(Structure):
    pass
VkPhysicalDeviceRayTracingMotionBlurFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("rayTracingMotionBlur", c_uint),
             ("rayTracingMotionBlurPipelineTraceRaysIndirect", c_uint)
    ]

class VkPhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT(Structure):
    pass
VkPhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("ycbcr2plane444Formats", c_uint)
    ]

class VkPhysicalDeviceFragmentDensityMap2FeaturesEXT(Structure):
    pass
VkPhysicalDeviceFragmentDensityMap2FeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentDensityMapDeferred", c_uint)
    ]

class VkPhysicalDeviceFragmentDensityMap2PropertiesEXT(Structure):
    pass
VkPhysicalDeviceFragmentDensityMap2PropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("subsampledLoads", c_uint),
             ("subsampledCoarseReconstructionEarlyAccess", c_uint),
             ("maxSubsampledArrayLayers", c_uint),
             ("maxDescriptorSetSubsampledSamplers", c_uint)
    ]

class VkCopyCommandTransformInfoQCOM(Structure):
    pass
VkCopyCommandTransformInfoQCOM._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("transform", c_int)
    ]

class VkPhysicalDevice4444FormatsFeaturesEXT(Structure):
    pass
VkPhysicalDevice4444FormatsFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("formatA4R4G4B4", c_uint),
             ("formatA4B4G4R4", c_uint)
    ]

class VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM(Structure):
    pass
VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("rasterizationOrderColorAttachmentAccess", c_uint),
             ("rasterizationOrderDepthAttachmentAccess", c_uint),
             ("rasterizationOrderStencilAttachmentAccess", c_uint)
    ]

class VkPhysicalDeviceRGBA10X6FormatsFeaturesEXT(Structure):
    pass
VkPhysicalDeviceRGBA10X6FormatsFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("formatRgba10x6WithoutYCbCrSampler", c_uint)
    ]

class VkPhysicalDeviceMutableDescriptorTypeFeaturesVALVE(Structure):
    pass
VkPhysicalDeviceMutableDescriptorTypeFeaturesVALVE._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("mutableDescriptorType", c_uint)
    ]

class VkMutableDescriptorTypeListVALVE(Structure):
    pass
VkMutableDescriptorTypeListVALVE._fields_ = [
             ("descriptorTypeCount", c_uint),
             ("pDescriptorTypes", POINTER(c_int))
    ]

class VkMutableDescriptorTypeCreateInfoVALVE(Structure):
    pass
VkMutableDescriptorTypeCreateInfoVALVE._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("mutableDescriptorTypeListCount", c_uint),
             ("pMutableDescriptorTypeLists", VkMutableDescriptorTypeListVALVE)
    ]

class VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT(Structure):
    pass
VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("vertexInputDynamicState", c_uint)
    ]

class VkVertexInputBindingDescription2EXT(Structure):
    pass
VkVertexInputBindingDescription2EXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("binding", c_uint),
             ("stride", c_uint),
             ("inputRate", c_int),
             ("divisor", c_uint)
    ]

class VkVertexInputAttributeDescription2EXT(Structure):
    pass
VkVertexInputAttributeDescription2EXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("location", c_uint),
             ("binding", c_uint),
             ("format", c_int),
             ("offset", c_uint)
    ]

class VkPhysicalDeviceDrmPropertiesEXT(Structure):
    pass
VkPhysicalDeviceDrmPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("hasPrimary", c_uint),
             ("hasRender", c_uint),
             ("primaryMajor", c_long),
             ("primaryMinor", c_long),
             ("renderMajor", c_long),
             ("renderMinor", c_long)
    ]

class VkPhysicalDeviceDepthClipControlFeaturesEXT(Structure):
    pass
VkPhysicalDeviceDepthClipControlFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("depthClipControl", c_uint)
    ]

class VkPipelineViewportDepthClipControlCreateInfoEXT(Structure):
    pass
VkPipelineViewportDepthClipControlCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("negativeOneToOne", c_uint)
    ]

class VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT(Structure):
    pass
VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("primitiveTopologyListRestart", c_uint),
             ("primitiveTopologyPatchListRestart", c_uint)
    ]

class VkSubpassShadingPipelineCreateInfoHUAWEI(Structure):
    pass
VkSubpassShadingPipelineCreateInfoHUAWEI._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("renderPass", VkRenderPass_T),
             ("subpass", c_uint)
    ]

class VkPhysicalDeviceSubpassShadingFeaturesHUAWEI(Structure):
    pass
VkPhysicalDeviceSubpassShadingFeaturesHUAWEI._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("subpassShading", c_uint)
    ]

class VkPhysicalDeviceSubpassShadingPropertiesHUAWEI(Structure):
    pass
VkPhysicalDeviceSubpassShadingPropertiesHUAWEI._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxSubpassShadingWorkgroupSizeAspectRatio", c_uint)
    ]

class VkPhysicalDeviceInvocationMaskFeaturesHUAWEI(Structure):
    pass
VkPhysicalDeviceInvocationMaskFeaturesHUAWEI._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("invocationMask", c_uint)
    ]

class VkMemoryGetRemoteAddressInfoNV(Structure):
    pass
VkMemoryGetRemoteAddressInfoNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("memory", VkDeviceMemory_T),
             ("handleType", c_int)
    ]

class VkPhysicalDeviceExternalMemoryRDMAFeaturesNV(Structure):
    pass
VkPhysicalDeviceExternalMemoryRDMAFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("externalMemoryRDMA", c_uint)
    ]

class VkPhysicalDeviceExtendedDynamicState2FeaturesEXT(Structure):
    pass
VkPhysicalDeviceExtendedDynamicState2FeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("extendedDynamicState2", c_uint),
             ("extendedDynamicState2LogicOp", c_uint),
             ("extendedDynamicState2PatchControlPoints", c_uint)
    ]

class VkPhysicalDeviceColorWriteEnableFeaturesEXT(Structure):
    pass
VkPhysicalDeviceColorWriteEnableFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("colorWriteEnable", c_uint)
    ]

class VkPipelineColorWriteCreateInfoEXT(Structure):
    pass
VkPipelineColorWriteCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("attachmentCount", c_uint),
             ("pColorWriteEnables", POINTER(c_uint))
    ]

class VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT(Structure):
    pass
VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("primitivesGeneratedQuery", c_uint),
             ("primitivesGeneratedQueryWithRasterizerDiscard", c_uint),
             ("primitivesGeneratedQueryWithNonZeroStreams", c_uint)
    ]

class VkPhysicalDeviceImageViewMinLodFeaturesEXT(Structure):
    pass
VkPhysicalDeviceImageViewMinLodFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("minLod", c_uint)
    ]

class VkImageViewMinLodCreateInfoEXT(Structure):
    pass
VkImageViewMinLodCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("minLod", c_float)
    ]

class VkPhysicalDeviceMultiDrawFeaturesEXT(Structure):
    pass
VkPhysicalDeviceMultiDrawFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("multiDraw", c_uint)
    ]

class VkPhysicalDeviceMultiDrawPropertiesEXT(Structure):
    pass
VkPhysicalDeviceMultiDrawPropertiesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxMultiDrawCount", c_uint)
    ]

class VkMultiDrawInfoEXT(Structure):
    pass
VkMultiDrawInfoEXT._fields_ = [
             ("firstc_uintertex", c_uint),
             ("vertexCount", c_uint)
    ]

class VkMultiDrawIndexedInfoEXT(Structure):
    pass
VkMultiDrawIndexedInfoEXT._fields_ = [
             ("firstIndex", c_uint),
             ("indexCount", c_uint),
             ("vertexOffset", c_int)
    ]

class VkPhysicalDeviceImage2DViewOf3DFeaturesEXT(Structure):
    pass
VkPhysicalDeviceImage2DViewOf3DFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("image2Dc_uintiewOf3D", c_uint),
             ("sampler2Dc_uintiewOf3D", c_uint)
    ]

class VkPhysicalDeviceBorderColorSwizzleFeaturesEXT(Structure):
    pass
VkPhysicalDeviceBorderColorSwizzleFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("borderColorSwizzle", c_uint),
             ("borderColorSwizzleFromImage", c_uint)
    ]

class VkSamplerBorderColorComponentMappingCreateInfoEXT(Structure):
    pass
VkSamplerBorderColorComponentMappingCreateInfoEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("components", VkComponentMapping),
             ("srgb", c_uint)
    ]

class VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT(Structure):
    pass
VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pageableDeviceLocalMemory", c_uint)
    ]

class VkPhysicalDeviceDescriptorSetHostMappingFeaturesVALVE(Structure):
    pass
VkPhysicalDeviceDescriptorSetHostMappingFeaturesVALVE._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("descriptorSetHostMapping", c_uint)
    ]

class VkDescriptorSetBindingReferenceVALVE(Structure):
    pass
VkDescriptorSetBindingReferenceVALVE._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("descriptorSetLayout", VkDescriptorSetLayout_T),
             ("binding", c_uint)
    ]

class VkDescriptorSetLayoutHostMappingInfoVALVE(Structure):
    pass
VkDescriptorSetLayoutHostMappingInfoVALVE._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("descriptorOffset", c_ulong),
             ("descriptorSize", c_uint)
    ]

class VkPhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM(Structure):
    pass
VkPhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentDensityMapOffset", c_uint)
    ]

class VkPhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM(Structure):
    pass
VkPhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentDensityOffsetGranularity", VkExtent2D)
    ]

class VkSubpassFragmentDensityMapOffsetEndInfoQCOM(Structure):
    pass
VkSubpassFragmentDensityMapOffsetEndInfoQCOM._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("fragmentDensityOffsetCount", c_uint),
             ("pFragmentDensityOffsets", VkOffset2D)
    ]

class VkPhysicalDeviceLinearColorAttachmentFeaturesNV(Structure):
    pass
VkPhysicalDeviceLinearColorAttachmentFeaturesNV._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("linearColorAttachment", c_uint)
    ]

class VkDeviceOrHostAddressKHR(Structure):
    pass
VkDeviceOrHostAddressKHR._fields_ = [
             ("deviceAddress", c_ulong),
             ("hostAddress", c_void_p)
    ]

class VkAccelerationStructureBuildRangeInfoKHR(Structure):
    pass
VkAccelerationStructureBuildRangeInfoKHR._fields_ = [
             ("primitiveCount", c_uint),
             ("primitiveOffset", c_uint),
             ("firstc_uintertex", c_uint),
             ("transformOffset", c_uint)
    ]

class VkAccelerationStructureGeometryTrianglesDataKHR(Structure):
    pass
VkAccelerationStructureGeometryTrianglesDataKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("vertexFormat", c_int),
             ("vertexData", VkDeviceOrHostAddressConstKHR),
             ("vertexStride", c_ulong),
             ("maxc_uintertex", c_uint),
             ("indexType", c_int),
             ("indexData", VkDeviceOrHostAddressConstKHR),
             ("transformData", VkDeviceOrHostAddressConstKHR)
    ]

class VkAccelerationStructureGeometryAabbsDataKHR(Structure):
    pass
VkAccelerationStructureGeometryAabbsDataKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("data", VkDeviceOrHostAddressConstKHR),
             ("stride", c_ulong)
    ]

class VkAccelerationStructureGeometryInstancesDataKHR(Structure):
    pass
VkAccelerationStructureGeometryInstancesDataKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("arrayOfPointers", c_uint),
             ("data", VkDeviceOrHostAddressConstKHR)
    ]

class VkAccelerationStructureGeometryDataKHR(Structure):
    pass
VkAccelerationStructureGeometryDataKHR._fields_ = [
             ("triangles", VkAccelerationStructureGeometryTrianglesDataKHR),
             ("aabbs", VkAccelerationStructureGeometryAabbsDataKHR),
             ("instances", VkAccelerationStructureGeometryInstancesDataKHR)
    ]

class VkAccelerationStructureGeometryKHR(Structure):
    pass
VkAccelerationStructureGeometryKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("geometryType", c_int),
             ("geometry", VkAccelerationStructureGeometryDataKHR),
             ("flags", c_uint)
    ]

class VkAccelerationStructureBuildGeometryInfoKHR(Structure):
    pass
VkAccelerationStructureBuildGeometryInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("type", c_int),
             ("flags", c_uint),
             ("mode", c_int),
             ("srcAccelerationStructure", VkAccelerationStructureKHR_T),
             ("dstAccelerationStructure", VkAccelerationStructureKHR_T),
             ("geometryCount", c_uint),
             ("pGeometries", VkAccelerationStructureGeometryKHR),
             ("ppGeometries", VkAccelerationStructureGeometryKHR),
             ("scratchData", VkDeviceOrHostAddressKHR)
    ]

class VkAccelerationStructureCreateInfoKHR(Structure):
    pass
VkAccelerationStructureCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("createFlags", c_uint),
             ("buffer", VkBuffer_T),
             ("offset", c_ulong),
             ("size", c_ulong),
             ("type", c_int),
             ("deviceAddress", c_ulong)
    ]

class VkWriteDescriptorSetAccelerationStructureKHR(Structure):
    pass
VkWriteDescriptorSetAccelerationStructureKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("accelerationStructureCount", c_uint),
             ("pAccelerationStructures", POINTER(VkAccelerationStructureKHR_T))
    ]

class VkPhysicalDeviceAccelerationStructureFeaturesKHR(Structure):
    pass
VkPhysicalDeviceAccelerationStructureFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("accelerationStructure", c_uint),
             ("accelerationStructureCaptureReplay", c_uint),
             ("accelerationStructureIndirectBuild", c_uint),
             ("accelerationStructureHostCommands", c_uint),
             ("descriptorBindingAccelerationStructureUpdateAfterBind", c_uint)
    ]

class VkPhysicalDeviceAccelerationStructurePropertiesKHR(Structure):
    pass
VkPhysicalDeviceAccelerationStructurePropertiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxGeometryCount", c_ulong),
             ("maxInstanceCount", c_ulong),
             ("maxPrimitiveCount", c_ulong),
             ("maxPerStageDescriptorAccelerationStructures", c_uint),
             ("maxPerStageDescriptorUpdateAfterBindAccelerationStructures", c_uint),
             ("maxDescriptorSetAccelerationStructures", c_uint),
             ("maxDescriptorSetUpdateAfterBindAccelerationStructures", c_uint),
             ("minAccelerationStructureScratchOffsetAlignment", c_uint)
    ]

class VkAccelerationStructureDeviceAddressInfoKHR(Structure):
    pass
VkAccelerationStructureDeviceAddressInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("accelerationStructure", VkAccelerationStructureKHR_T)
    ]

class VkAccelerationStructureVersionInfoKHR(Structure):
    pass
VkAccelerationStructureVersionInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("pPOINTER(c_ubyte)ersionData", POINTER(c_ubyte))
    ]

class VkCopyAccelerationStructureToMemoryInfoKHR(Structure):
    pass
VkCopyAccelerationStructureToMemoryInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("src", VkAccelerationStructureKHR_T),
             ("dst", VkDeviceOrHostAddressKHR),
             ("mode", c_int)
    ]

class VkCopyMemoryToAccelerationStructureInfoKHR(Structure):
    pass
VkCopyMemoryToAccelerationStructureInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("src", VkDeviceOrHostAddressConstKHR),
             ("dst", VkAccelerationStructureKHR_T),
             ("mode", c_int)
    ]

class VkCopyAccelerationStructureInfoKHR(Structure):
    pass
VkCopyAccelerationStructureInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("src", VkAccelerationStructureKHR_T),
             ("dst", VkAccelerationStructureKHR_T),
             ("mode", c_int)
    ]

class VkAccelerationStructureBuildSizesInfoKHR(Structure):
    pass
VkAccelerationStructureBuildSizesInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("accelerationStructureSize", c_ulong),
             ("updateScratchSize", c_ulong),
             ("buildScratchSize", c_ulong)
    ]

class VkRayTracingShaderGroupCreateInfoKHR(Structure):
    pass
VkRayTracingShaderGroupCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("type", c_int),
             ("generalShader", c_uint),
             ("closestHitShader", c_uint),
             ("anyHitShader", c_uint),
             ("intersectionShader", c_uint),
             ("pShaderGroupCaptureReplayHandle", c_void_p)
    ]

class VkRayTracingPipelineInterfaceCreateInfoKHR(Structure):
    pass
VkRayTracingPipelineInterfaceCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("maxPipelineRayPayloadSize", c_uint),
             ("maxPipelineRayHitAttributeSize", c_uint)
    ]

class VkRayTracingPipelineCreateInfoKHR(Structure):
    pass
VkRayTracingPipelineCreateInfoKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("flags", c_uint),
             ("stageCount", c_uint),
             ("pStages", VkPipelineShaderStageCreateInfo),
             ("groupCount", c_uint),
             ("pGroups", VkRayTracingShaderGroupCreateInfoKHR),
             ("maxPipelineRayRecursionDepth", c_uint),
             ("pLibraryInfo", VkPipelineLibraryCreateInfoKHR),
             ("pLibraryInterface", VkRayTracingPipelineInterfaceCreateInfoKHR),
             ("pDynamicState", VkPipelineDynamicStateCreateInfo),
             ("layout", VkPipelineLayout_T),
             ("basePipelineHandle", VkPipeline_T),
             ("basePipelineIndex", c_int)
    ]

class VkPhysicalDeviceRayTracingPipelineFeaturesKHR(Structure):
    pass
VkPhysicalDeviceRayTracingPipelineFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("rayTracingPipeline", c_uint),
             ("rayTracingPipelineShaderGroupHandleCaptureReplay", c_uint),
             ("rayTracingPipelineShaderGroupHandleCaptureReplayMixed", c_uint),
             ("rayTracingPipelineTraceRaysIndirect", c_uint),
             ("rayTraversalPrimitiveCulling", c_uint)
    ]

class VkPhysicalDeviceRayTracingPipelinePropertiesKHR(Structure):
    pass
VkPhysicalDeviceRayTracingPipelinePropertiesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("shaderGroupHandleSize", c_uint),
             ("maxRayRecursionDepth", c_uint),
             ("maxShaderGroupStride", c_uint),
             ("shaderGroupBaseAlignment", c_uint),
             ("shaderGroupHandleCaptureReplaySize", c_uint),
             ("maxRayDispatchInvocationCount", c_uint),
             ("shaderGroupHandleAlignment", c_uint),
             ("maxRayHitAttributeSize", c_uint)
    ]

class VkStridedDeviceAddressRegionKHR(Structure):
    pass
VkStridedDeviceAddressRegionKHR._fields_ = [
             ("deviceAddress", c_ulong),
             ("stride", c_ulong),
             ("size", c_ulong)
    ]

class VkTraceRaysIndirectCommandKHR(Structure):
    pass
VkTraceRaysIndirectCommandKHR._fields_ = [
             ("width", c_uint),
             ("height", c_uint),
             ("depth", c_uint)
    ]

class VkPhysicalDeviceRayQueryFeaturesKHR(Structure):
    pass
VkPhysicalDeviceRayQueryFeaturesKHR._fields_ = [
             ("sType", c_int),
             ("pNext", c_void_p),
             ("rayQuery", c_uint)
    ]

def vkCreateInstance(indict):
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkInstanceCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pInstance" in indict.keys():
         pInstance = indict["pInstance"]
    else: 
         pInstance = pointer(VkInstance_T())
    print(jvulkanLib.vkCreateInstance)
    retval = jvulkanLib.vkCreateInstance(pCreateInfo, pAllocator, pInstance)
    return {"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pInstance" : pInstance,"retval" : retval}
def vkDestroyInstance(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyInstance)
    retval = jvulkanLib.vkDestroyInstance(instance, pAllocator)
    return {"instance" : instance,"pAllocator" : pAllocator,"retval" : retval}
def vkEnumeratePhysicalDevices(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "pPhysicalDeviceCount" in indict.keys():
         pPhysicalDeviceCount = indict["pPhysicalDeviceCount"]
    else: 
         pPhysicalDeviceCount = pointer(c_uint())
    if "pPhysicalDevices" in indict.keys():
         pPhysicalDevices = indict["pPhysicalDevices"]
    else: 
         pPhysicalDevices = pointer(VkPhysicalDevice_T())
    print(jvulkanLib.vkEnumeratePhysicalDevices)
    retval = jvulkanLib.vkEnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices)
    return {"instance" : instance,"pPhysicalDeviceCount" : pPhysicalDeviceCount,"pPhysicalDevices" : pPhysicalDevices,"retval" : retval}
def vkGetPhysicalDeviceFeatures(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pFeatures" in indict.keys():
         pFeatures = indict["pFeatures"]
    else: 
         pFeatures = VkPhysicalDeviceFeatures()
    print(jvulkanLib.vkGetPhysicalDeviceFeatures)
    retval = jvulkanLib.vkGetPhysicalDeviceFeatures(physicalDevice, pFeatures)
    return {"physicalDevice" : physicalDevice,"pFeatures" : pFeatures,"retval" : retval}
def vkGetPhysicalDeviceFormatProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "format" in indict.keys():
         format = indict["format"]
    else: 
         format = c_int()
    if "pFormatProperties" in indict.keys():
         pFormatProperties = indict["pFormatProperties"]
    else: 
         pFormatProperties = VkFormatProperties()
    print(jvulkanLib.vkGetPhysicalDeviceFormatProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceFormatProperties(physicalDevice, format, pFormatProperties)
    return {"physicalDevice" : physicalDevice,"format" : format,"pFormatProperties" : pFormatProperties,"retval" : retval}
def vkGetPhysicalDeviceImageFormatProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "format" in indict.keys():
         format = indict["format"]
    else: 
         format = c_int()
    if "type" in indict.keys():
         type = indict["type"]
    else: 
         type = c_int()
    if "tiling" in indict.keys():
         tiling = indict["tiling"]
    else: 
         tiling = c_int()
    if "usage" in indict.keys():
         usage = indict["usage"]
    else: 
         usage = c_uint()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    if "pImageFormatProperties" in indict.keys():
         pImageFormatProperties = indict["pImageFormatProperties"]
    else: 
         pImageFormatProperties = VkImageFormatProperties()
    print(jvulkanLib.vkGetPhysicalDeviceImageFormatProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceImageFormatProperties(physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties)
    return {"physicalDevice" : physicalDevice,"format" : format,"type" : type,"tiling" : tiling,"usage" : usage,"flags" : flags,"pImageFormatProperties" : pImageFormatProperties,"retval" : retval}
def vkGetPhysicalDeviceProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkPhysicalDeviceProperties()
    print(jvulkanLib.vkGetPhysicalDeviceProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceProperties(physicalDevice, pProperties)
    return {"physicalDevice" : physicalDevice,"pProperties" : pProperties,"retval" : retval}
def vkGetPhysicalDeviceQueueFamilyProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pQueueFamilyPropertyCount" in indict.keys():
         pQueueFamilyPropertyCount = indict["pQueueFamilyPropertyCount"]
    else: 
         pQueueFamilyPropertyCount = pointer(c_uint())
    if "pQueueFamilyProperties" in indict.keys():
         pQueueFamilyProperties = indict["pQueueFamilyProperties"]
    else: 
         pQueueFamilyProperties = VkQueueFamilyProperties()
    print(jvulkanLib.vkGetPhysicalDeviceQueueFamilyProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties)
    return {"physicalDevice" : physicalDevice,"pQueueFamilyPropertyCount" : pQueueFamilyPropertyCount,"pQueueFamilyProperties" : pQueueFamilyProperties,"retval" : retval}
def vkGetPhysicalDeviceMemoryProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pMemoryProperties" in indict.keys():
         pMemoryProperties = indict["pMemoryProperties"]
    else: 
         pMemoryProperties = VkPhysicalDeviceMemoryProperties()
    print(jvulkanLib.vkGetPhysicalDeviceMemoryProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceMemoryProperties(physicalDevice, pMemoryProperties)
    return {"physicalDevice" : physicalDevice,"pMemoryProperties" : pMemoryProperties,"retval" : retval}
def vkGetInstanceProcAddr(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "pName" in indict.keys():
         pName = indict["pName"]
    else: 
         pName = c_char_p()
    print(jvulkanLib.vkGetInstanceProcAddr)
    retval = jvulkanLib.vkGetInstanceProcAddr(instance, pName)
    return {"instance" : instance,"pName" : pName,"retval" : retval}
def vkGetDeviceProcAddr(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pName" in indict.keys():
         pName = indict["pName"]
    else: 
         pName = c_char_p()
    print(jvulkanLib.vkGetDeviceProcAddr)
    retval = jvulkanLib.vkGetDeviceProcAddr(device, pName)
    return {"device" : device,"pName" : pName,"retval" : retval}
def vkCreateDevice(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDeviceCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pDevice" in indict.keys():
         pDevice = indict["pDevice"]
    else: 
         pDevice = pointer(VkDevice_T())
    print(jvulkanLib.vkCreateDevice)
    retval = jvulkanLib.vkCreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice)
    return {"physicalDevice" : physicalDevice,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pDevice" : pDevice,"retval" : retval}
def vkDestroyDevice(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyDevice)
    retval = jvulkanLib.vkDestroyDevice(device, pAllocator)
    return {"device" : device,"pAllocator" : pAllocator,"retval" : retval}
def vkEnumerateInstanceExtensionProperties(indict):
    if "pLayerName" in indict.keys():
         pLayerName = indict["pLayerName"]
    else: 
         pLayerName = c_char_p()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkExtensionProperties()
    print(jvulkanLib.vkEnumerateInstanceExtensionProperties)
    retval = jvulkanLib.vkEnumerateInstanceExtensionProperties(pLayerName, pPropertyCount, pProperties)
    return {"pLayerName" : pLayerName,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkEnumerateDeviceExtensionProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pLayerName" in indict.keys():
         pLayerName = indict["pLayerName"]
    else: 
         pLayerName = c_char_p()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkExtensionProperties()
    print(jvulkanLib.vkEnumerateDeviceExtensionProperties)
    retval = jvulkanLib.vkEnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"pLayerName" : pLayerName,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkEnumerateInstanceLayerProperties(indict):
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkLayerProperties()
    print(jvulkanLib.vkEnumerateInstanceLayerProperties)
    retval = jvulkanLib.vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties)
    return {"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkEnumerateDeviceLayerProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkLayerProperties()
    print(jvulkanLib.vkEnumerateDeviceLayerProperties)
    retval = jvulkanLib.vkEnumerateDeviceLayerProperties(physicalDevice, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkGetDeviceQueue(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "queueFamilyIndex" in indict.keys():
         queueFamilyIndex = indict["queueFamilyIndex"]
    else: 
         queueFamilyIndex = c_uint()
    if "queueIndex" in indict.keys():
         queueIndex = indict["queueIndex"]
    else: 
         queueIndex = c_uint()
    if "pQueue" in indict.keys():
         pQueue = indict["pQueue"]
    else: 
         pQueue = pointer(VkQueue_T())
    print(jvulkanLib.vkGetDeviceQueue)
    retval = jvulkanLib.vkGetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue)
    return {"device" : device,"queueFamilyIndex" : queueFamilyIndex,"queueIndex" : queueIndex,"pQueue" : pQueue,"retval" : retval}
def vkQueueSubmit(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "submitCount" in indict.keys():
         submitCount = indict["submitCount"]
    else: 
         submitCount = c_uint()
    if "pSubmits" in indict.keys():
         pSubmits = indict["pSubmits"]
    else: 
         pSubmits = VkSubmitInfo()
    if "fence" in indict.keys():
         fence = indict["fence"]
    else: 
         fence = VkFence_T()
    print(jvulkanLib.vkQueueSubmit)
    retval = jvulkanLib.vkQueueSubmit(queue, submitCount, pSubmits, fence)
    return {"queue" : queue,"submitCount" : submitCount,"pSubmits" : pSubmits,"fence" : fence,"retval" : retval}
def vkQueueWaitIdle(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    print(jvulkanLib.vkQueueWaitIdle)
    retval = jvulkanLib.vkQueueWaitIdle(queue)
    return {"queue" : queue,"retval" : retval}
def vkDeviceWaitIdle(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    print(jvulkanLib.vkDeviceWaitIdle)
    retval = jvulkanLib.vkDeviceWaitIdle(device)
    return {"device" : device,"retval" : retval}
def vkAllocateMemory(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pAllocateInfo" in indict.keys():
         pAllocateInfo = indict["pAllocateInfo"]
    else: 
         pAllocateInfo = VkMemoryAllocateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pMemory" in indict.keys():
         pMemory = indict["pMemory"]
    else: 
         pMemory = pointer(VkDeviceMemory_T())
    print(jvulkanLib.vkAllocateMemory)
    retval = jvulkanLib.vkAllocateMemory(device, pAllocateInfo, pAllocator, pMemory)
    return {"device" : device,"pAllocateInfo" : pAllocateInfo,"pAllocator" : pAllocator,"pMemory" : pMemory,"retval" : retval}
def vkFreeMemory(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "memory" in indict.keys():
         memory = indict["memory"]
    else: 
         memory = VkDeviceMemory_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkFreeMemory)
    retval = jvulkanLib.vkFreeMemory(device, memory, pAllocator)
    return {"device" : device,"memory" : memory,"pAllocator" : pAllocator,"retval" : retval}
def vkMapMemory(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "memory" in indict.keys():
         memory = indict["memory"]
    else: 
         memory = VkDeviceMemory_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "size" in indict.keys():
         size = indict["size"]
    else: 
         size = c_ulong()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    if "ppData" in indict.keys():
         ppData = indict["ppData"]
    else: 
         ppData = POINTER(c_void_p)()
    print(jvulkanLib.vkMapMemory)
    retval = jvulkanLib.vkMapMemory(device, memory, offset, size, flags, ppData)
    return {"device" : device,"memory" : memory,"offset" : offset,"size" : size,"flags" : flags,"ppData" : ppData,"retval" : retval}
def vkUnmapMemory(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "memory" in indict.keys():
         memory = indict["memory"]
    else: 
         memory = VkDeviceMemory_T()
    print(jvulkanLib.vkUnmapMemory)
    retval = jvulkanLib.vkUnmapMemory(device, memory)
    return {"device" : device,"memory" : memory,"retval" : retval}
def vkFlushMappedMemoryRanges(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "memoryRangeCount" in indict.keys():
         memoryRangeCount = indict["memoryRangeCount"]
    else: 
         memoryRangeCount = c_uint()
    if "pMemoryRanges" in indict.keys():
         pMemoryRanges = indict["pMemoryRanges"]
    else: 
         pMemoryRanges = VkMappedMemoryRange()
    print(jvulkanLib.vkFlushMappedMemoryRanges)
    retval = jvulkanLib.vkFlushMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges)
    return {"device" : device,"memoryRangeCount" : memoryRangeCount,"pMemoryRanges" : pMemoryRanges,"retval" : retval}
def vkInvalidateMappedMemoryRanges(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "memoryRangeCount" in indict.keys():
         memoryRangeCount = indict["memoryRangeCount"]
    else: 
         memoryRangeCount = c_uint()
    if "pMemoryRanges" in indict.keys():
         pMemoryRanges = indict["pMemoryRanges"]
    else: 
         pMemoryRanges = VkMappedMemoryRange()
    print(jvulkanLib.vkInvalidateMappedMemoryRanges)
    retval = jvulkanLib.vkInvalidateMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges)
    return {"device" : device,"memoryRangeCount" : memoryRangeCount,"pMemoryRanges" : pMemoryRanges,"retval" : retval}
def vkGetDeviceMemoryCommitment(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "memory" in indict.keys():
         memory = indict["memory"]
    else: 
         memory = VkDeviceMemory_T()
    if "pCommittedMemoryInBytes" in indict.keys():
         pCommittedMemoryInBytes = indict["pCommittedMemoryInBytes"]
    else: 
         pCommittedMemoryInBytes = pointer(c_ulong())
    print(jvulkanLib.vkGetDeviceMemoryCommitment)
    retval = jvulkanLib.vkGetDeviceMemoryCommitment(device, memory, pCommittedMemoryInBytes)
    return {"device" : device,"memory" : memory,"pCommittedMemoryInBytes" : pCommittedMemoryInBytes,"retval" : retval}
def vkBindBufferMemory(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "memory" in indict.keys():
         memory = indict["memory"]
    else: 
         memory = VkDeviceMemory_T()
    if "memoryOffset" in indict.keys():
         memoryOffset = indict["memoryOffset"]
    else: 
         memoryOffset = c_ulong()
    print(jvulkanLib.vkBindBufferMemory)
    retval = jvulkanLib.vkBindBufferMemory(device, buffer, memory, memoryOffset)
    return {"device" : device,"buffer" : buffer,"memory" : memory,"memoryOffset" : memoryOffset,"retval" : retval}
def vkBindImageMemory(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "image" in indict.keys():
         image = indict["image"]
    else: 
         image = VkImage_T()
    if "memory" in indict.keys():
         memory = indict["memory"]
    else: 
         memory = VkDeviceMemory_T()
    if "memoryOffset" in indict.keys():
         memoryOffset = indict["memoryOffset"]
    else: 
         memoryOffset = c_ulong()
    print(jvulkanLib.vkBindImageMemory)
    retval = jvulkanLib.vkBindImageMemory(device, image, memory, memoryOffset)
    return {"device" : device,"image" : image,"memory" : memory,"memoryOffset" : memoryOffset,"retval" : retval}
def vkGetBufferMemoryRequirements(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements()
    print(jvulkanLib.vkGetBufferMemoryRequirements)
    retval = jvulkanLib.vkGetBufferMemoryRequirements(device, buffer, pMemoryRequirements)
    return {"device" : device,"buffer" : buffer,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetImageMemoryRequirements(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "image" in indict.keys():
         image = indict["image"]
    else: 
         image = VkImage_T()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements()
    print(jvulkanLib.vkGetImageMemoryRequirements)
    retval = jvulkanLib.vkGetImageMemoryRequirements(device, image, pMemoryRequirements)
    return {"device" : device,"image" : image,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetImageSparseMemoryRequirements(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "image" in indict.keys():
         image = indict["image"]
    else: 
         image = VkImage_T()
    if "pSparseMemoryRequirementCount" in indict.keys():
         pSparseMemoryRequirementCount = indict["pSparseMemoryRequirementCount"]
    else: 
         pSparseMemoryRequirementCount = pointer(c_uint())
    if "pSparseMemoryRequirements" in indict.keys():
         pSparseMemoryRequirements = indict["pSparseMemoryRequirements"]
    else: 
         pSparseMemoryRequirements = VkSparseImageMemoryRequirements()
    print(jvulkanLib.vkGetImageSparseMemoryRequirements)
    retval = jvulkanLib.vkGetImageSparseMemoryRequirements(device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements)
    return {"device" : device,"image" : image,"pSparseMemoryRequirementCount" : pSparseMemoryRequirementCount,"pSparseMemoryRequirements" : pSparseMemoryRequirements,"retval" : retval}
def vkGetPhysicalDeviceSparseImageFormatProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "format" in indict.keys():
         format = indict["format"]
    else: 
         format = c_int()
    if "type" in indict.keys():
         type = indict["type"]
    else: 
         type = c_int()
    if "samples" in indict.keys():
         samples = indict["samples"]
    else: 
         samples = c_int()
    if "usage" in indict.keys():
         usage = indict["usage"]
    else: 
         usage = c_uint()
    if "tiling" in indict.keys():
         tiling = indict["tiling"]
    else: 
         tiling = c_int()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkSparseImageFormatProperties()
    print(jvulkanLib.vkGetPhysicalDeviceSparseImageFormatProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceSparseImageFormatProperties(physicalDevice, format, type, samples, usage, tiling, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"format" : format,"type" : type,"samples" : samples,"usage" : usage,"tiling" : tiling,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkQueueBindSparse(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "bindInfoCount" in indict.keys():
         bindInfoCount = indict["bindInfoCount"]
    else: 
         bindInfoCount = c_uint()
    if "pBindInfo" in indict.keys():
         pBindInfo = indict["pBindInfo"]
    else: 
         pBindInfo = VkBindSparseInfo()
    if "fence" in indict.keys():
         fence = indict["fence"]
    else: 
         fence = VkFence_T()
    print(jvulkanLib.vkQueueBindSparse)
    retval = jvulkanLib.vkQueueBindSparse(queue, bindInfoCount, pBindInfo, fence)
    return {"queue" : queue,"bindInfoCount" : bindInfoCount,"pBindInfo" : pBindInfo,"fence" : fence,"retval" : retval}
def vkCreateFence(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkFenceCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pFence" in indict.keys():
         pFence = indict["pFence"]
    else: 
         pFence = pointer(VkFence_T())
    print(jvulkanLib.vkCreateFence)
    retval = jvulkanLib.vkCreateFence(device, pCreateInfo, pAllocator, pFence)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pFence" : pFence,"retval" : retval}
def vkDestroyFence(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "fence" in indict.keys():
         fence = indict["fence"]
    else: 
         fence = VkFence_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyFence)
    retval = jvulkanLib.vkDestroyFence(device, fence, pAllocator)
    return {"device" : device,"fence" : fence,"pAllocator" : pAllocator,"retval" : retval}
def vkResetFences(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "fenceCount" in indict.keys():
         fenceCount = indict["fenceCount"]
    else: 
         fenceCount = c_uint()
    if "pFences" in indict.keys():
         pFences = indict["pFences"]
    else: 
         pFences = pointer(VkFence_T())
    print(jvulkanLib.vkResetFences)
    retval = jvulkanLib.vkResetFences(device, fenceCount, pFences)
    return {"device" : device,"fenceCount" : fenceCount,"pFences" : pFences,"retval" : retval}
def vkGetFenceStatus(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "fence" in indict.keys():
         fence = indict["fence"]
    else: 
         fence = VkFence_T()
    print(jvulkanLib.vkGetFenceStatus)
    retval = jvulkanLib.vkGetFenceStatus(device, fence)
    return {"device" : device,"fence" : fence,"retval" : retval}
def vkWaitForFences(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "fenceCount" in indict.keys():
         fenceCount = indict["fenceCount"]
    else: 
         fenceCount = c_uint()
    if "pFences" in indict.keys():
         pFences = indict["pFences"]
    else: 
         pFences = pointer(VkFence_T())
    if "waitAll" in indict.keys():
         waitAll = indict["waitAll"]
    else: 
         waitAll = c_uint()
    if "timeout" in indict.keys():
         timeout = indict["timeout"]
    else: 
         timeout = c_ulong()
    print(jvulkanLib.vkWaitForFences)
    retval = jvulkanLib.vkWaitForFences(device, fenceCount, pFences, waitAll, timeout)
    return {"device" : device,"fenceCount" : fenceCount,"pFences" : pFences,"waitAll" : waitAll,"timeout" : timeout,"retval" : retval}
def vkCreateSemaphore(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkSemaphoreCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pSemaphore" in indict.keys():
         pSemaphore = indict["pSemaphore"]
    else: 
         pSemaphore = pointer(VkSemaphore_T())
    print(jvulkanLib.vkCreateSemaphore)
    retval = jvulkanLib.vkCreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pSemaphore" : pSemaphore,"retval" : retval}
def vkDestroySemaphore(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "semaphore" in indict.keys():
         semaphore = indict["semaphore"]
    else: 
         semaphore = VkSemaphore_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroySemaphore)
    retval = jvulkanLib.vkDestroySemaphore(device, semaphore, pAllocator)
    return {"device" : device,"semaphore" : semaphore,"pAllocator" : pAllocator,"retval" : retval}
def vkCreateEvent(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkEventCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pEvent" in indict.keys():
         pEvent = indict["pEvent"]
    else: 
         pEvent = pointer(VkEvent_T())
    print(jvulkanLib.vkCreateEvent)
    retval = jvulkanLib.vkCreateEvent(device, pCreateInfo, pAllocator, pEvent)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pEvent" : pEvent,"retval" : retval}
def vkDestroyEvent(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyEvent)
    retval = jvulkanLib.vkDestroyEvent(device, event, pAllocator)
    return {"device" : device,"event" : event,"pAllocator" : pAllocator,"retval" : retval}
def vkGetEventStatus(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    print(jvulkanLib.vkGetEventStatus)
    retval = jvulkanLib.vkGetEventStatus(device, event)
    return {"device" : device,"event" : event,"retval" : retval}
def vkSetEvent(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    print(jvulkanLib.vkSetEvent)
    retval = jvulkanLib.vkSetEvent(device, event)
    return {"device" : device,"event" : event,"retval" : retval}
def vkResetEvent(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    print(jvulkanLib.vkResetEvent)
    retval = jvulkanLib.vkResetEvent(device, event)
    return {"device" : device,"event" : event,"retval" : retval}
def vkCreateQueryPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkQueryPoolCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pQueryPool" in indict.keys():
         pQueryPool = indict["pQueryPool"]
    else: 
         pQueryPool = pointer(VkQueryPool_T())
    print(jvulkanLib.vkCreateQueryPool)
    retval = jvulkanLib.vkCreateQueryPool(device, pCreateInfo, pAllocator, pQueryPool)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pQueryPool" : pQueryPool,"retval" : retval}
def vkDestroyQueryPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyQueryPool)
    retval = jvulkanLib.vkDestroyQueryPool(device, queryPool, pAllocator)
    return {"device" : device,"queryPool" : queryPool,"pAllocator" : pAllocator,"retval" : retval}
def vkGetQueryPoolResults(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "firstQuery" in indict.keys():
         firstQuery = indict["firstQuery"]
    else: 
         firstQuery = c_uint()
    if "queryCount" in indict.keys():
         queryCount = indict["queryCount"]
    else: 
         queryCount = c_uint()
    if "dataSize" in indict.keys():
         dataSize = indict["dataSize"]
    else: 
         dataSize = c_ulong()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_ulong()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    print(jvulkanLib.vkGetQueryPoolResults)
    retval = jvulkanLib.vkGetQueryPoolResults(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags)
    return {"device" : device,"queryPool" : queryPool,"firstQuery" : firstQuery,"queryCount" : queryCount,"dataSize" : dataSize,"pData" : pData,"stride" : stride,"flags" : flags,"retval" : retval}
def vkCreateBuffer(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkBufferCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pBuffer" in indict.keys():
         pBuffer = indict["pBuffer"]
    else: 
         pBuffer = pointer(VkBuffer_T())
    print(jvulkanLib.vkCreateBuffer)
    retval = jvulkanLib.vkCreateBuffer(device, pCreateInfo, pAllocator, pBuffer)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pBuffer" : pBuffer,"retval" : retval}
def vkDestroyBuffer(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyBuffer)
    retval = jvulkanLib.vkDestroyBuffer(device, buffer, pAllocator)
    return {"device" : device,"buffer" : buffer,"pAllocator" : pAllocator,"retval" : retval}
def vkCreateBufferView(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkBufferViewCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pView" in indict.keys():
         pView = indict["pView"]
    else: 
         pView = pointer(VkBufferView_T())
    print(jvulkanLib.vkCreateBufferView)
    retval = jvulkanLib.vkCreateBufferView(device, pCreateInfo, pAllocator, pView)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pView" : pView,"retval" : retval}
def vkDestroyBufferView(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "bufferView" in indict.keys():
         bufferView = indict["bufferView"]
    else: 
         bufferView = VkBufferView_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyBufferView)
    retval = jvulkanLib.vkDestroyBufferView(device, bufferView, pAllocator)
    return {"device" : device,"bufferView" : bufferView,"pAllocator" : pAllocator,"retval" : retval}
def vkCreateImage(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkImageCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pImage" in indict.keys():
         pImage = indict["pImage"]
    else: 
         pImage = pointer(VkImage_T())
    print(jvulkanLib.vkCreateImage)
    retval = jvulkanLib.vkCreateImage(device, pCreateInfo, pAllocator, pImage)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pImage" : pImage,"retval" : retval}
def vkDestroyImage(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "image" in indict.keys():
         image = indict["image"]
    else: 
         image = VkImage_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyImage)
    retval = jvulkanLib.vkDestroyImage(device, image, pAllocator)
    return {"device" : device,"image" : image,"pAllocator" : pAllocator,"retval" : retval}
def vkGetImageSubresourceLayout(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "image" in indict.keys():
         image = indict["image"]
    else: 
         image = VkImage_T()
    if "pSubresource" in indict.keys():
         pSubresource = indict["pSubresource"]
    else: 
         pSubresource = VkImageSubresource()
    if "pLayout" in indict.keys():
         pLayout = indict["pLayout"]
    else: 
         pLayout = VkSubresourceLayout()
    print(jvulkanLib.vkGetImageSubresourceLayout)
    retval = jvulkanLib.vkGetImageSubresourceLayout(device, image, pSubresource, pLayout)
    return {"device" : device,"image" : image,"pSubresource" : pSubresource,"pLayout" : pLayout,"retval" : retval}
def vkCreateImageView(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkImageViewCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pView" in indict.keys():
         pView = indict["pView"]
    else: 
         pView = pointer(VkImageView_T())
    print(jvulkanLib.vkCreateImageView)
    retval = jvulkanLib.vkCreateImageView(device, pCreateInfo, pAllocator, pView)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pView" : pView,"retval" : retval}
def vkDestroyImageView(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "imageView" in indict.keys():
         imageView = indict["imageView"]
    else: 
         imageView = VkImageView_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyImageView)
    retval = jvulkanLib.vkDestroyImageView(device, imageView, pAllocator)
    return {"device" : device,"imageView" : imageView,"pAllocator" : pAllocator,"retval" : retval}
def vkCreateShaderModule(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkShaderModuleCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pShaderModule" in indict.keys():
         pShaderModule = indict["pShaderModule"]
    else: 
         pShaderModule = pointer(VkShaderModule_T())
    print(jvulkanLib.vkCreateShaderModule)
    retval = jvulkanLib.vkCreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pShaderModule" : pShaderModule,"retval" : retval}
def vkDestroyShaderModule(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "shaderModule" in indict.keys():
         shaderModule = indict["shaderModule"]
    else: 
         shaderModule = VkShaderModule_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyShaderModule)
    retval = jvulkanLib.vkDestroyShaderModule(device, shaderModule, pAllocator)
    return {"device" : device,"shaderModule" : shaderModule,"pAllocator" : pAllocator,"retval" : retval}
def vkCreatePipelineCache(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkPipelineCacheCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pPipelineCache" in indict.keys():
         pPipelineCache = indict["pPipelineCache"]
    else: 
         pPipelineCache = pointer(VkPipelineCache_T())
    print(jvulkanLib.vkCreatePipelineCache)
    retval = jvulkanLib.vkCreatePipelineCache(device, pCreateInfo, pAllocator, pPipelineCache)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pPipelineCache" : pPipelineCache,"retval" : retval}
def vkDestroyPipelineCache(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipelineCache" in indict.keys():
         pipelineCache = indict["pipelineCache"]
    else: 
         pipelineCache = VkPipelineCache_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyPipelineCache)
    retval = jvulkanLib.vkDestroyPipelineCache(device, pipelineCache, pAllocator)
    return {"device" : device,"pipelineCache" : pipelineCache,"pAllocator" : pAllocator,"retval" : retval}
def vkGetPipelineCacheData(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipelineCache" in indict.keys():
         pipelineCache = indict["pipelineCache"]
    else: 
         pipelineCache = VkPipelineCache_T()
    if "pDataSize" in indict.keys():
         pDataSize = indict["pDataSize"]
    else: 
         pDataSize = pointer(c_ulong())
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkGetPipelineCacheData)
    retval = jvulkanLib.vkGetPipelineCacheData(device, pipelineCache, pDataSize, pData)
    return {"device" : device,"pipelineCache" : pipelineCache,"pDataSize" : pDataSize,"pData" : pData,"retval" : retval}
def vkMergePipelineCaches(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "dstCache" in indict.keys():
         dstCache = indict["dstCache"]
    else: 
         dstCache = VkPipelineCache_T()
    if "srcCacheCount" in indict.keys():
         srcCacheCount = indict["srcCacheCount"]
    else: 
         srcCacheCount = c_uint()
    if "pSrcCaches" in indict.keys():
         pSrcCaches = indict["pSrcCaches"]
    else: 
         pSrcCaches = pointer(VkPipelineCache_T())
    print(jvulkanLib.vkMergePipelineCaches)
    retval = jvulkanLib.vkMergePipelineCaches(device, dstCache, srcCacheCount, pSrcCaches)
    return {"device" : device,"dstCache" : dstCache,"srcCacheCount" : srcCacheCount,"pSrcCaches" : pSrcCaches,"retval" : retval}
def vkCreateGraphicsPipelines(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipelineCache" in indict.keys():
         pipelineCache = indict["pipelineCache"]
    else: 
         pipelineCache = VkPipelineCache_T()
    if "createInfoCount" in indict.keys():
         createInfoCount = indict["createInfoCount"]
    else: 
         createInfoCount = c_uint()
    if "pCreateInfos" in indict.keys():
         pCreateInfos = indict["pCreateInfos"]
    else: 
         pCreateInfos = VkGraphicsPipelineCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pPipelines" in indict.keys():
         pPipelines = indict["pPipelines"]
    else: 
         pPipelines = pointer(VkPipeline_T())
    print(jvulkanLib.vkCreateGraphicsPipelines)
    retval = jvulkanLib.vkCreateGraphicsPipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines)
    return {"device" : device,"pipelineCache" : pipelineCache,"createInfoCount" : createInfoCount,"pCreateInfos" : pCreateInfos,"pAllocator" : pAllocator,"pPipelines" : pPipelines,"retval" : retval}
def vkCreateComputePipelines(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipelineCache" in indict.keys():
         pipelineCache = indict["pipelineCache"]
    else: 
         pipelineCache = VkPipelineCache_T()
    if "createInfoCount" in indict.keys():
         createInfoCount = indict["createInfoCount"]
    else: 
         createInfoCount = c_uint()
    if "pCreateInfos" in indict.keys():
         pCreateInfos = indict["pCreateInfos"]
    else: 
         pCreateInfos = VkComputePipelineCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pPipelines" in indict.keys():
         pPipelines = indict["pPipelines"]
    else: 
         pPipelines = pointer(VkPipeline_T())
    print(jvulkanLib.vkCreateComputePipelines)
    retval = jvulkanLib.vkCreateComputePipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines)
    return {"device" : device,"pipelineCache" : pipelineCache,"createInfoCount" : createInfoCount,"pCreateInfos" : pCreateInfos,"pAllocator" : pAllocator,"pPipelines" : pPipelines,"retval" : retval}
def vkDestroyPipeline(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipeline" in indict.keys():
         pipeline = indict["pipeline"]
    else: 
         pipeline = VkPipeline_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyPipeline)
    retval = jvulkanLib.vkDestroyPipeline(device, pipeline, pAllocator)
    return {"device" : device,"pipeline" : pipeline,"pAllocator" : pAllocator,"retval" : retval}
def vkCreatePipelineLayout(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkPipelineLayoutCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pPipelineLayout" in indict.keys():
         pPipelineLayout = indict["pPipelineLayout"]
    else: 
         pPipelineLayout = pointer(VkPipelineLayout_T())
    print(jvulkanLib.vkCreatePipelineLayout)
    retval = jvulkanLib.vkCreatePipelineLayout(device, pCreateInfo, pAllocator, pPipelineLayout)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pPipelineLayout" : pPipelineLayout,"retval" : retval}
def vkDestroyPipelineLayout(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipelineLayout" in indict.keys():
         pipelineLayout = indict["pipelineLayout"]
    else: 
         pipelineLayout = VkPipelineLayout_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyPipelineLayout)
    retval = jvulkanLib.vkDestroyPipelineLayout(device, pipelineLayout, pAllocator)
    return {"device" : device,"pipelineLayout" : pipelineLayout,"pAllocator" : pAllocator,"retval" : retval}
def vkCreateSampler(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkSamplerCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pSampler" in indict.keys():
         pSampler = indict["pSampler"]
    else: 
         pSampler = pointer(VkSampler_T())
    print(jvulkanLib.vkCreateSampler)
    retval = jvulkanLib.vkCreateSampler(device, pCreateInfo, pAllocator, pSampler)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pSampler" : pSampler,"retval" : retval}
def vkDestroySampler(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "sampler" in indict.keys():
         sampler = indict["sampler"]
    else: 
         sampler = VkSampler_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroySampler)
    retval = jvulkanLib.vkDestroySampler(device, sampler, pAllocator)
    return {"device" : device,"sampler" : sampler,"pAllocator" : pAllocator,"retval" : retval}
def vkCreateDescriptorSetLayout(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDescriptorSetLayoutCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pSetLayout" in indict.keys():
         pSetLayout = indict["pSetLayout"]
    else: 
         pSetLayout = pointer(VkDescriptorSetLayout_T())
    print(jvulkanLib.vkCreateDescriptorSetLayout)
    retval = jvulkanLib.vkCreateDescriptorSetLayout(device, pCreateInfo, pAllocator, pSetLayout)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pSetLayout" : pSetLayout,"retval" : retval}
def vkDestroyDescriptorSetLayout(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorSetLayout" in indict.keys():
         descriptorSetLayout = indict["descriptorSetLayout"]
    else: 
         descriptorSetLayout = VkDescriptorSetLayout_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyDescriptorSetLayout)
    retval = jvulkanLib.vkDestroyDescriptorSetLayout(device, descriptorSetLayout, pAllocator)
    return {"device" : device,"descriptorSetLayout" : descriptorSetLayout,"pAllocator" : pAllocator,"retval" : retval}
def vkCreateDescriptorPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDescriptorPoolCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pDescriptorPool" in indict.keys():
         pDescriptorPool = indict["pDescriptorPool"]
    else: 
         pDescriptorPool = pointer(VkDescriptorPool_T())
    print(jvulkanLib.vkCreateDescriptorPool)
    retval = jvulkanLib.vkCreateDescriptorPool(device, pCreateInfo, pAllocator, pDescriptorPool)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pDescriptorPool" : pDescriptorPool,"retval" : retval}
def vkDestroyDescriptorPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorPool" in indict.keys():
         descriptorPool = indict["descriptorPool"]
    else: 
         descriptorPool = VkDescriptorPool_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyDescriptorPool)
    retval = jvulkanLib.vkDestroyDescriptorPool(device, descriptorPool, pAllocator)
    return {"device" : device,"descriptorPool" : descriptorPool,"pAllocator" : pAllocator,"retval" : retval}
def vkResetDescriptorPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorPool" in indict.keys():
         descriptorPool = indict["descriptorPool"]
    else: 
         descriptorPool = VkDescriptorPool_T()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    print(jvulkanLib.vkResetDescriptorPool)
    retval = jvulkanLib.vkResetDescriptorPool(device, descriptorPool, flags)
    return {"device" : device,"descriptorPool" : descriptorPool,"flags" : flags,"retval" : retval}
def vkAllocateDescriptorSets(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pAllocateInfo" in indict.keys():
         pAllocateInfo = indict["pAllocateInfo"]
    else: 
         pAllocateInfo = VkDescriptorSetAllocateInfo()
    if "pDescriptorSets" in indict.keys():
         pDescriptorSets = indict["pDescriptorSets"]
    else: 
         pDescriptorSets = pointer(VkDescriptorSet_T())
    print(jvulkanLib.vkAllocateDescriptorSets)
    retval = jvulkanLib.vkAllocateDescriptorSets(device, pAllocateInfo, pDescriptorSets)
    return {"device" : device,"pAllocateInfo" : pAllocateInfo,"pDescriptorSets" : pDescriptorSets,"retval" : retval}
def vkFreeDescriptorSets(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorPool" in indict.keys():
         descriptorPool = indict["descriptorPool"]
    else: 
         descriptorPool = VkDescriptorPool_T()
    if "descriptorSetCount" in indict.keys():
         descriptorSetCount = indict["descriptorSetCount"]
    else: 
         descriptorSetCount = c_uint()
    if "pDescriptorSets" in indict.keys():
         pDescriptorSets = indict["pDescriptorSets"]
    else: 
         pDescriptorSets = pointer(VkDescriptorSet_T())
    print(jvulkanLib.vkFreeDescriptorSets)
    retval = jvulkanLib.vkFreeDescriptorSets(device, descriptorPool, descriptorSetCount, pDescriptorSets)
    return {"device" : device,"descriptorPool" : descriptorPool,"descriptorSetCount" : descriptorSetCount,"pDescriptorSets" : pDescriptorSets,"retval" : retval}
def vkUpdateDescriptorSets(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorWriteCount" in indict.keys():
         descriptorWriteCount = indict["descriptorWriteCount"]
    else: 
         descriptorWriteCount = c_uint()
    if "pDescriptorWrites" in indict.keys():
         pDescriptorWrites = indict["pDescriptorWrites"]
    else: 
         pDescriptorWrites = VkWriteDescriptorSet()
    if "descriptorCopyCount" in indict.keys():
         descriptorCopyCount = indict["descriptorCopyCount"]
    else: 
         descriptorCopyCount = c_uint()
    if "pDescriptorCopies" in indict.keys():
         pDescriptorCopies = indict["pDescriptorCopies"]
    else: 
         pDescriptorCopies = VkCopyDescriptorSet()
    print(jvulkanLib.vkUpdateDescriptorSets)
    retval = jvulkanLib.vkUpdateDescriptorSets(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies)
    return {"device" : device,"descriptorWriteCount" : descriptorWriteCount,"pDescriptorWrites" : pDescriptorWrites,"descriptorCopyCount" : descriptorCopyCount,"pDescriptorCopies" : pDescriptorCopies,"retval" : retval}
def vkCreateFramebuffer(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkFramebufferCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pFramebuffer" in indict.keys():
         pFramebuffer = indict["pFramebuffer"]
    else: 
         pFramebuffer = pointer(VkFramebuffer_T())
    print(jvulkanLib.vkCreateFramebuffer)
    retval = jvulkanLib.vkCreateFramebuffer(device, pCreateInfo, pAllocator, pFramebuffer)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pFramebuffer" : pFramebuffer,"retval" : retval}
def vkDestroyFramebuffer(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "framebuffer" in indict.keys():
         framebuffer = indict["framebuffer"]
    else: 
         framebuffer = VkFramebuffer_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyFramebuffer)
    retval = jvulkanLib.vkDestroyFramebuffer(device, framebuffer, pAllocator)
    return {"device" : device,"framebuffer" : framebuffer,"pAllocator" : pAllocator,"retval" : retval}
def vkCreateRenderPass(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkRenderPassCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pRenderPass" in indict.keys():
         pRenderPass = indict["pRenderPass"]
    else: 
         pRenderPass = pointer(VkRenderPass_T())
    print(jvulkanLib.vkCreateRenderPass)
    retval = jvulkanLib.vkCreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pRenderPass" : pRenderPass,"retval" : retval}
def vkDestroyRenderPass(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "renderPass" in indict.keys():
         renderPass = indict["renderPass"]
    else: 
         renderPass = VkRenderPass_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyRenderPass)
    retval = jvulkanLib.vkDestroyRenderPass(device, renderPass, pAllocator)
    return {"device" : device,"renderPass" : renderPass,"pAllocator" : pAllocator,"retval" : retval}
def vkGetRenderAreaGranularity(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "renderPass" in indict.keys():
         renderPass = indict["renderPass"]
    else: 
         renderPass = VkRenderPass_T()
    if "pGranularity" in indict.keys():
         pGranularity = indict["pGranularity"]
    else: 
         pGranularity = VkExtent2D()
    print(jvulkanLib.vkGetRenderAreaGranularity)
    retval = jvulkanLib.vkGetRenderAreaGranularity(device, renderPass, pGranularity)
    return {"device" : device,"renderPass" : renderPass,"pGranularity" : pGranularity,"retval" : retval}
def vkCreateCommandPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkCommandPoolCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pCommandPool" in indict.keys():
         pCommandPool = indict["pCommandPool"]
    else: 
         pCommandPool = pointer(VkCommandPool_T())
    print(jvulkanLib.vkCreateCommandPool)
    retval = jvulkanLib.vkCreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pCommandPool" : pCommandPool,"retval" : retval}
def vkDestroyCommandPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "commandPool" in indict.keys():
         commandPool = indict["commandPool"]
    else: 
         commandPool = VkCommandPool_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyCommandPool)
    retval = jvulkanLib.vkDestroyCommandPool(device, commandPool, pAllocator)
    return {"device" : device,"commandPool" : commandPool,"pAllocator" : pAllocator,"retval" : retval}
def vkResetCommandPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "commandPool" in indict.keys():
         commandPool = indict["commandPool"]
    else: 
         commandPool = VkCommandPool_T()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    print(jvulkanLib.vkResetCommandPool)
    retval = jvulkanLib.vkResetCommandPool(device, commandPool, flags)
    return {"device" : device,"commandPool" : commandPool,"flags" : flags,"retval" : retval}
def vkAllocateCommandBuffers(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pAllocateInfo" in indict.keys():
         pAllocateInfo = indict["pAllocateInfo"]
    else: 
         pAllocateInfo = VkCommandBufferAllocateInfo()
    if "pCommandBuffers" in indict.keys():
         pCommandBuffers = indict["pCommandBuffers"]
    else: 
         pCommandBuffers = pointer(VkCommandBuffer_T())
    print(jvulkanLib.vkAllocateCommandBuffers)
    retval = jvulkanLib.vkAllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers)
    return {"device" : device,"pAllocateInfo" : pAllocateInfo,"pCommandBuffers" : pCommandBuffers,"retval" : retval}
def vkFreeCommandBuffers(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "commandPool" in indict.keys():
         commandPool = indict["commandPool"]
    else: 
         commandPool = VkCommandPool_T()
    if "commandBufferCount" in indict.keys():
         commandBufferCount = indict["commandBufferCount"]
    else: 
         commandBufferCount = c_uint()
    if "pCommandBuffers" in indict.keys():
         pCommandBuffers = indict["pCommandBuffers"]
    else: 
         pCommandBuffers = pointer(VkCommandBuffer_T())
    print(jvulkanLib.vkFreeCommandBuffers)
    retval = jvulkanLib.vkFreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers)
    return {"device" : device,"commandPool" : commandPool,"commandBufferCount" : commandBufferCount,"pCommandBuffers" : pCommandBuffers,"retval" : retval}
def vkBeginCommandBuffer(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pBeginInfo" in indict.keys():
         pBeginInfo = indict["pBeginInfo"]
    else: 
         pBeginInfo = VkCommandBufferBeginInfo()
    print(jvulkanLib.vkBeginCommandBuffer)
    retval = jvulkanLib.vkBeginCommandBuffer(commandBuffer, pBeginInfo)
    return {"commandBuffer" : commandBuffer,"pBeginInfo" : pBeginInfo,"retval" : retval}
def vkEndCommandBuffer(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    print(jvulkanLib.vkEndCommandBuffer)
    retval = jvulkanLib.vkEndCommandBuffer(commandBuffer)
    return {"commandBuffer" : commandBuffer,"retval" : retval}
def vkResetCommandBuffer(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    print(jvulkanLib.vkResetCommandBuffer)
    retval = jvulkanLib.vkResetCommandBuffer(commandBuffer, flags)
    return {"commandBuffer" : commandBuffer,"flags" : flags,"retval" : retval}
def vkCmdBindPipeline(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pipelineBindPoint" in indict.keys():
         pipelineBindPoint = indict["pipelineBindPoint"]
    else: 
         pipelineBindPoint = c_int()
    if "pipeline" in indict.keys():
         pipeline = indict["pipeline"]
    else: 
         pipeline = VkPipeline_T()
    print(jvulkanLib.vkCmdBindPipeline)
    retval = jvulkanLib.vkCmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline)
    return {"commandBuffer" : commandBuffer,"pipelineBindPoint" : pipelineBindPoint,"pipeline" : pipeline,"retval" : retval}
def vkCmdSetViewport(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstViewport" in indict.keys():
         firstViewport = indict["firstViewport"]
    else: 
         firstViewport = c_uint()
    if "viewportCount" in indict.keys():
         viewportCount = indict["viewportCount"]
    else: 
         viewportCount = c_uint()
    if "pViewports" in indict.keys():
         pViewports = indict["pViewports"]
    else: 
         pViewports = VkViewport()
    print(jvulkanLib.vkCmdSetViewport)
    retval = jvulkanLib.vkCmdSetViewport(commandBuffer, firstViewport, viewportCount, pViewports)
    return {"commandBuffer" : commandBuffer,"firstViewport" : firstViewport,"viewportCount" : viewportCount,"pViewports" : pViewports,"retval" : retval}
def vkCmdSetScissor(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstScissor" in indict.keys():
         firstScissor = indict["firstScissor"]
    else: 
         firstScissor = c_uint()
    if "scissorCount" in indict.keys():
         scissorCount = indict["scissorCount"]
    else: 
         scissorCount = c_uint()
    if "pScissors" in indict.keys():
         pScissors = indict["pScissors"]
    else: 
         pScissors = VkRect2D()
    print(jvulkanLib.vkCmdSetScissor)
    retval = jvulkanLib.vkCmdSetScissor(commandBuffer, firstScissor, scissorCount, pScissors)
    return {"commandBuffer" : commandBuffer,"firstScissor" : firstScissor,"scissorCount" : scissorCount,"pScissors" : pScissors,"retval" : retval}
def vkCmdSetLineWidth(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "lineWidth" in indict.keys():
         lineWidth = indict["lineWidth"]
    else: 
         lineWidth = c_float()
    print(jvulkanLib.vkCmdSetLineWidth)
    retval = jvulkanLib.vkCmdSetLineWidth(commandBuffer, lineWidth)
    return {"commandBuffer" : commandBuffer,"lineWidth" : lineWidth,"retval" : retval}
def vkCmdSetDepthBias(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthBiasConstantFactor" in indict.keys():
         depthBiasConstantFactor = indict["depthBiasConstantFactor"]
    else: 
         depthBiasConstantFactor = c_float()
    if "depthBiasClamp" in indict.keys():
         depthBiasClamp = indict["depthBiasClamp"]
    else: 
         depthBiasClamp = c_float()
    if "depthBiasSlopeFactor" in indict.keys():
         depthBiasSlopeFactor = indict["depthBiasSlopeFactor"]
    else: 
         depthBiasSlopeFactor = c_float()
    print(jvulkanLib.vkCmdSetDepthBias)
    retval = jvulkanLib.vkCmdSetDepthBias(commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor)
    return {"commandBuffer" : commandBuffer,"depthBiasConstantFactor" : depthBiasConstantFactor,"depthBiasClamp" : depthBiasClamp,"depthBiasSlopeFactor" : depthBiasSlopeFactor,"retval" : retval}
def vkCmdSetBlendConstants(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "blendConstants" in indict.keys():
         blendConstants = indict["blendConstants"]
    else: 
         blendConstants = pointer(c_float())
    print(jvulkanLib.vkCmdSetBlendConstants)
    retval = jvulkanLib.vkCmdSetBlendConstants(commandBuffer, blendConstants)
    return {"commandBuffer" : commandBuffer,"blendConstants" : blendConstants,"retval" : retval}
def vkCmdSetDepthBounds(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "minDepthBounds" in indict.keys():
         minDepthBounds = indict["minDepthBounds"]
    else: 
         minDepthBounds = c_float()
    if "maxDepthBounds" in indict.keys():
         maxDepthBounds = indict["maxDepthBounds"]
    else: 
         maxDepthBounds = c_float()
    print(jvulkanLib.vkCmdSetDepthBounds)
    retval = jvulkanLib.vkCmdSetDepthBounds(commandBuffer, minDepthBounds, maxDepthBounds)
    return {"commandBuffer" : commandBuffer,"minDepthBounds" : minDepthBounds,"maxDepthBounds" : maxDepthBounds,"retval" : retval}
def vkCmdSetStencilCompareMask(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "faceMask" in indict.keys():
         faceMask = indict["faceMask"]
    else: 
         faceMask = c_uint()
    if "compareMask" in indict.keys():
         compareMask = indict["compareMask"]
    else: 
         compareMask = c_uint()
    print(jvulkanLib.vkCmdSetStencilCompareMask)
    retval = jvulkanLib.vkCmdSetStencilCompareMask(commandBuffer, faceMask, compareMask)
    return {"commandBuffer" : commandBuffer,"faceMask" : faceMask,"compareMask" : compareMask,"retval" : retval}
def vkCmdSetStencilWriteMask(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "faceMask" in indict.keys():
         faceMask = indict["faceMask"]
    else: 
         faceMask = c_uint()
    if "writeMask" in indict.keys():
         writeMask = indict["writeMask"]
    else: 
         writeMask = c_uint()
    print(jvulkanLib.vkCmdSetStencilWriteMask)
    retval = jvulkanLib.vkCmdSetStencilWriteMask(commandBuffer, faceMask, writeMask)
    return {"commandBuffer" : commandBuffer,"faceMask" : faceMask,"writeMask" : writeMask,"retval" : retval}
def vkCmdSetStencilReference(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "faceMask" in indict.keys():
         faceMask = indict["faceMask"]
    else: 
         faceMask = c_uint()
    if "reference" in indict.keys():
         reference = indict["reference"]
    else: 
         reference = c_uint()
    print(jvulkanLib.vkCmdSetStencilReference)
    retval = jvulkanLib.vkCmdSetStencilReference(commandBuffer, faceMask, reference)
    return {"commandBuffer" : commandBuffer,"faceMask" : faceMask,"reference" : reference,"retval" : retval}
def vkCmdBindDescriptorSets(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pipelineBindPoint" in indict.keys():
         pipelineBindPoint = indict["pipelineBindPoint"]
    else: 
         pipelineBindPoint = c_int()
    if "layout" in indict.keys():
         layout = indict["layout"]
    else: 
         layout = VkPipelineLayout_T()
    if "firstSet" in indict.keys():
         firstSet = indict["firstSet"]
    else: 
         firstSet = c_uint()
    if "descriptorSetCount" in indict.keys():
         descriptorSetCount = indict["descriptorSetCount"]
    else: 
         descriptorSetCount = c_uint()
    if "pDescriptorSets" in indict.keys():
         pDescriptorSets = indict["pDescriptorSets"]
    else: 
         pDescriptorSets = pointer(VkDescriptorSet_T())
    if "dynamicOffsetCount" in indict.keys():
         dynamicOffsetCount = indict["dynamicOffsetCount"]
    else: 
         dynamicOffsetCount = c_uint()
    if "pDynamicOffsets" in indict.keys():
         pDynamicOffsets = indict["pDynamicOffsets"]
    else: 
         pDynamicOffsets = pointer(c_uint())
    print(jvulkanLib.vkCmdBindDescriptorSets)
    retval = jvulkanLib.vkCmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets)
    return {"commandBuffer" : commandBuffer,"pipelineBindPoint" : pipelineBindPoint,"layout" : layout,"firstSet" : firstSet,"descriptorSetCount" : descriptorSetCount,"pDescriptorSets" : pDescriptorSets,"dynamicOffsetCount" : dynamicOffsetCount,"pDynamicOffsets" : pDynamicOffsets,"retval" : retval}
def vkCmdBindIndexBuffer(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "indexType" in indict.keys():
         indexType = indict["indexType"]
    else: 
         indexType = c_int()
    print(jvulkanLib.vkCmdBindIndexBuffer)
    retval = jvulkanLib.vkCmdBindIndexBuffer(commandBuffer, buffer, offset, indexType)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"indexType" : indexType,"retval" : retval}
def vkCmdBindVertexBuffers(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstBinding" in indict.keys():
         firstBinding = indict["firstBinding"]
    else: 
         firstBinding = c_uint()
    if "bindingCount" in indict.keys():
         bindingCount = indict["bindingCount"]
    else: 
         bindingCount = c_uint()
    if "pBuffers" in indict.keys():
         pBuffers = indict["pBuffers"]
    else: 
         pBuffers = pointer(VkBuffer_T())
    if "pOffsets" in indict.keys():
         pOffsets = indict["pOffsets"]
    else: 
         pOffsets = pointer(c_ulong())
    print(jvulkanLib.vkCmdBindVertexBuffers)
    retval = jvulkanLib.vkCmdBindVertexBuffers(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets)
    return {"commandBuffer" : commandBuffer,"firstBinding" : firstBinding,"bindingCount" : bindingCount,"pBuffers" : pBuffers,"pOffsets" : pOffsets,"retval" : retval}
def vkCmdDraw(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "vertexCount" in indict.keys():
         vertexCount = indict["vertexCount"]
    else: 
         vertexCount = c_uint()
    if "instanceCount" in indict.keys():
         instanceCount = indict["instanceCount"]
    else: 
         instanceCount = c_uint()
    if "firstVertex" in indict.keys():
         firstVertex = indict["firstVertex"]
    else: 
         firstVertex = c_uint()
    if "firstInstance" in indict.keys():
         firstInstance = indict["firstInstance"]
    else: 
         firstInstance = c_uint()
    print(jvulkanLib.vkCmdDraw)
    retval = jvulkanLib.vkCmdDraw(commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance)
    return {"commandBuffer" : commandBuffer,"vertexCount" : vertexCount,"instanceCount" : instanceCount,"firstVertex" : firstVertex,"firstInstance" : firstInstance,"retval" : retval}
def vkCmdDrawIndexed(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "indexCount" in indict.keys():
         indexCount = indict["indexCount"]
    else: 
         indexCount = c_uint()
    if "instanceCount" in indict.keys():
         instanceCount = indict["instanceCount"]
    else: 
         instanceCount = c_uint()
    if "firstIndex" in indict.keys():
         firstIndex = indict["firstIndex"]
    else: 
         firstIndex = c_uint()
    if "vertexOffset" in indict.keys():
         vertexOffset = indict["vertexOffset"]
    else: 
         vertexOffset = c_int()
    if "firstInstance" in indict.keys():
         firstInstance = indict["firstInstance"]
    else: 
         firstInstance = c_uint()
    print(jvulkanLib.vkCmdDrawIndexed)
    retval = jvulkanLib.vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance)
    return {"commandBuffer" : commandBuffer,"indexCount" : indexCount,"instanceCount" : instanceCount,"firstIndex" : firstIndex,"vertexOffset" : vertexOffset,"firstInstance" : firstInstance,"retval" : retval}
def vkCmdDrawIndirect(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "drawCount" in indict.keys():
         drawCount = indict["drawCount"]
    else: 
         drawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawIndirect)
    retval = jvulkanLib.vkCmdDrawIndirect(commandBuffer, buffer, offset, drawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"drawCount" : drawCount,"stride" : stride,"retval" : retval}
def vkCmdDrawIndexedIndirect(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "drawCount" in indict.keys():
         drawCount = indict["drawCount"]
    else: 
         drawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawIndexedIndirect)
    retval = jvulkanLib.vkCmdDrawIndexedIndirect(commandBuffer, buffer, offset, drawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"drawCount" : drawCount,"stride" : stride,"retval" : retval}
def vkCmdDispatch(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "groupCountX" in indict.keys():
         groupCountX = indict["groupCountX"]
    else: 
         groupCountX = c_uint()
    if "groupCountY" in indict.keys():
         groupCountY = indict["groupCountY"]
    else: 
         groupCountY = c_uint()
    if "groupCountZ" in indict.keys():
         groupCountZ = indict["groupCountZ"]
    else: 
         groupCountZ = c_uint()
    print(jvulkanLib.vkCmdDispatch)
    retval = jvulkanLib.vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ)
    return {"commandBuffer" : commandBuffer,"groupCountX" : groupCountX,"groupCountY" : groupCountY,"groupCountZ" : groupCountZ,"retval" : retval}
def vkCmdDispatchIndirect(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    print(jvulkanLib.vkCmdDispatchIndirect)
    retval = jvulkanLib.vkCmdDispatchIndirect(commandBuffer, buffer, offset)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"retval" : retval}
def vkCmdCopyBuffer(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "srcBuffer" in indict.keys():
         srcBuffer = indict["srcBuffer"]
    else: 
         srcBuffer = VkBuffer_T()
    if "dstBuffer" in indict.keys():
         dstBuffer = indict["dstBuffer"]
    else: 
         dstBuffer = VkBuffer_T()
    if "regionCount" in indict.keys():
         regionCount = indict["regionCount"]
    else: 
         regionCount = c_uint()
    if "pRegions" in indict.keys():
         pRegions = indict["pRegions"]
    else: 
         pRegions = VkBufferCopy()
    print(jvulkanLib.vkCmdCopyBuffer)
    retval = jvulkanLib.vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions)
    return {"commandBuffer" : commandBuffer,"srcBuffer" : srcBuffer,"dstBuffer" : dstBuffer,"regionCount" : regionCount,"pRegions" : pRegions,"retval" : retval}
def vkCmdCopyImage(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "srcImage" in indict.keys():
         srcImage = indict["srcImage"]
    else: 
         srcImage = VkImage_T()
    if "srcImageLayout" in indict.keys():
         srcImageLayout = indict["srcImageLayout"]
    else: 
         srcImageLayout = c_int()
    if "dstImage" in indict.keys():
         dstImage = indict["dstImage"]
    else: 
         dstImage = VkImage_T()
    if "dstImageLayout" in indict.keys():
         dstImageLayout = indict["dstImageLayout"]
    else: 
         dstImageLayout = c_int()
    if "regionCount" in indict.keys():
         regionCount = indict["regionCount"]
    else: 
         regionCount = c_uint()
    if "pRegions" in indict.keys():
         pRegions = indict["pRegions"]
    else: 
         pRegions = VkImageCopy()
    print(jvulkanLib.vkCmdCopyImage)
    retval = jvulkanLib.vkCmdCopyImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions)
    return {"commandBuffer" : commandBuffer,"srcImage" : srcImage,"srcImageLayout" : srcImageLayout,"dstImage" : dstImage,"dstImageLayout" : dstImageLayout,"regionCount" : regionCount,"pRegions" : pRegions,"retval" : retval}
def vkCmdBlitImage(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "srcImage" in indict.keys():
         srcImage = indict["srcImage"]
    else: 
         srcImage = VkImage_T()
    if "srcImageLayout" in indict.keys():
         srcImageLayout = indict["srcImageLayout"]
    else: 
         srcImageLayout = c_int()
    if "dstImage" in indict.keys():
         dstImage = indict["dstImage"]
    else: 
         dstImage = VkImage_T()
    if "dstImageLayout" in indict.keys():
         dstImageLayout = indict["dstImageLayout"]
    else: 
         dstImageLayout = c_int()
    if "regionCount" in indict.keys():
         regionCount = indict["regionCount"]
    else: 
         regionCount = c_uint()
    if "pRegions" in indict.keys():
         pRegions = indict["pRegions"]
    else: 
         pRegions = VkImageBlit()
    if "filter" in indict.keys():
         filter = indict["filter"]
    else: 
         filter = c_int()
    print(jvulkanLib.vkCmdBlitImage)
    retval = jvulkanLib.vkCmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter)
    return {"commandBuffer" : commandBuffer,"srcImage" : srcImage,"srcImageLayout" : srcImageLayout,"dstImage" : dstImage,"dstImageLayout" : dstImageLayout,"regionCount" : regionCount,"pRegions" : pRegions,"filter" : filter,"retval" : retval}
def vkCmdCopyBufferToImage(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "srcBuffer" in indict.keys():
         srcBuffer = indict["srcBuffer"]
    else: 
         srcBuffer = VkBuffer_T()
    if "dstImage" in indict.keys():
         dstImage = indict["dstImage"]
    else: 
         dstImage = VkImage_T()
    if "dstImageLayout" in indict.keys():
         dstImageLayout = indict["dstImageLayout"]
    else: 
         dstImageLayout = c_int()
    if "regionCount" in indict.keys():
         regionCount = indict["regionCount"]
    else: 
         regionCount = c_uint()
    if "pRegions" in indict.keys():
         pRegions = indict["pRegions"]
    else: 
         pRegions = VkBufferImageCopy()
    print(jvulkanLib.vkCmdCopyBufferToImage)
    retval = jvulkanLib.vkCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions)
    return {"commandBuffer" : commandBuffer,"srcBuffer" : srcBuffer,"dstImage" : dstImage,"dstImageLayout" : dstImageLayout,"regionCount" : regionCount,"pRegions" : pRegions,"retval" : retval}
def vkCmdCopyImageToBuffer(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "srcImage" in indict.keys():
         srcImage = indict["srcImage"]
    else: 
         srcImage = VkImage_T()
    if "srcImageLayout" in indict.keys():
         srcImageLayout = indict["srcImageLayout"]
    else: 
         srcImageLayout = c_int()
    if "dstBuffer" in indict.keys():
         dstBuffer = indict["dstBuffer"]
    else: 
         dstBuffer = VkBuffer_T()
    if "regionCount" in indict.keys():
         regionCount = indict["regionCount"]
    else: 
         regionCount = c_uint()
    if "pRegions" in indict.keys():
         pRegions = indict["pRegions"]
    else: 
         pRegions = VkBufferImageCopy()
    print(jvulkanLib.vkCmdCopyImageToBuffer)
    retval = jvulkanLib.vkCmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions)
    return {"commandBuffer" : commandBuffer,"srcImage" : srcImage,"srcImageLayout" : srcImageLayout,"dstBuffer" : dstBuffer,"regionCount" : regionCount,"pRegions" : pRegions,"retval" : retval}
def vkCmdUpdateBuffer(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "dstBuffer" in indict.keys():
         dstBuffer = indict["dstBuffer"]
    else: 
         dstBuffer = VkBuffer_T()
    if "dstOffset" in indict.keys():
         dstOffset = indict["dstOffset"]
    else: 
         dstOffset = c_ulong()
    if "dataSize" in indict.keys():
         dataSize = indict["dataSize"]
    else: 
         dataSize = c_ulong()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkCmdUpdateBuffer)
    retval = jvulkanLib.vkCmdUpdateBuffer(commandBuffer, dstBuffer, dstOffset, dataSize, pData)
    return {"commandBuffer" : commandBuffer,"dstBuffer" : dstBuffer,"dstOffset" : dstOffset,"dataSize" : dataSize,"pData" : pData,"retval" : retval}
def vkCmdFillBuffer(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "dstBuffer" in indict.keys():
         dstBuffer = indict["dstBuffer"]
    else: 
         dstBuffer = VkBuffer_T()
    if "dstOffset" in indict.keys():
         dstOffset = indict["dstOffset"]
    else: 
         dstOffset = c_ulong()
    if "size" in indict.keys():
         size = indict["size"]
    else: 
         size = c_ulong()
    if "data" in indict.keys():
         data = indict["data"]
    else: 
         data = c_uint()
    print(jvulkanLib.vkCmdFillBuffer)
    retval = jvulkanLib.vkCmdFillBuffer(commandBuffer, dstBuffer, dstOffset, size, data)
    return {"commandBuffer" : commandBuffer,"dstBuffer" : dstBuffer,"dstOffset" : dstOffset,"size" : size,"data" : data,"retval" : retval}
def vkCmdClearColorImage(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "image" in indict.keys():
         image = indict["image"]
    else: 
         image = VkImage_T()
    if "imageLayout" in indict.keys():
         imageLayout = indict["imageLayout"]
    else: 
         imageLayout = c_int()
    if "pColor" in indict.keys():
         pColor = indict["pColor"]
    else: 
         pColor = VkClearColorValue()
    if "rangeCount" in indict.keys():
         rangeCount = indict["rangeCount"]
    else: 
         rangeCount = c_uint()
    if "pRanges" in indict.keys():
         pRanges = indict["pRanges"]
    else: 
         pRanges = VkImageSubresourceRange()
    print(jvulkanLib.vkCmdClearColorImage)
    retval = jvulkanLib.vkCmdClearColorImage(commandBuffer, image, imageLayout, pColor, rangeCount, pRanges)
    return {"commandBuffer" : commandBuffer,"image" : image,"imageLayout" : imageLayout,"pColor" : pColor,"rangeCount" : rangeCount,"pRanges" : pRanges,"retval" : retval}
def vkCmdClearDepthStencilImage(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "image" in indict.keys():
         image = indict["image"]
    else: 
         image = VkImage_T()
    if "imageLayout" in indict.keys():
         imageLayout = indict["imageLayout"]
    else: 
         imageLayout = c_int()
    if "pDepthStencil" in indict.keys():
         pDepthStencil = indict["pDepthStencil"]
    else: 
         pDepthStencil = VkClearDepthStencilValue()
    if "rangeCount" in indict.keys():
         rangeCount = indict["rangeCount"]
    else: 
         rangeCount = c_uint()
    if "pRanges" in indict.keys():
         pRanges = indict["pRanges"]
    else: 
         pRanges = VkImageSubresourceRange()
    print(jvulkanLib.vkCmdClearDepthStencilImage)
    retval = jvulkanLib.vkCmdClearDepthStencilImage(commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges)
    return {"commandBuffer" : commandBuffer,"image" : image,"imageLayout" : imageLayout,"pDepthStencil" : pDepthStencil,"rangeCount" : rangeCount,"pRanges" : pRanges,"retval" : retval}
def vkCmdClearAttachments(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "attachmentCount" in indict.keys():
         attachmentCount = indict["attachmentCount"]
    else: 
         attachmentCount = c_uint()
    if "pAttachments" in indict.keys():
         pAttachments = indict["pAttachments"]
    else: 
         pAttachments = VkClearAttachment()
    if "rectCount" in indict.keys():
         rectCount = indict["rectCount"]
    else: 
         rectCount = c_uint()
    if "pRects" in indict.keys():
         pRects = indict["pRects"]
    else: 
         pRects = VkClearRect()
    print(jvulkanLib.vkCmdClearAttachments)
    retval = jvulkanLib.vkCmdClearAttachments(commandBuffer, attachmentCount, pAttachments, rectCount, pRects)
    return {"commandBuffer" : commandBuffer,"attachmentCount" : attachmentCount,"pAttachments" : pAttachments,"rectCount" : rectCount,"pRects" : pRects,"retval" : retval}
def vkCmdResolveImage(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "srcImage" in indict.keys():
         srcImage = indict["srcImage"]
    else: 
         srcImage = VkImage_T()
    if "srcImageLayout" in indict.keys():
         srcImageLayout = indict["srcImageLayout"]
    else: 
         srcImageLayout = c_int()
    if "dstImage" in indict.keys():
         dstImage = indict["dstImage"]
    else: 
         dstImage = VkImage_T()
    if "dstImageLayout" in indict.keys():
         dstImageLayout = indict["dstImageLayout"]
    else: 
         dstImageLayout = c_int()
    if "regionCount" in indict.keys():
         regionCount = indict["regionCount"]
    else: 
         regionCount = c_uint()
    if "pRegions" in indict.keys():
         pRegions = indict["pRegions"]
    else: 
         pRegions = VkImageResolve()
    print(jvulkanLib.vkCmdResolveImage)
    retval = jvulkanLib.vkCmdResolveImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions)
    return {"commandBuffer" : commandBuffer,"srcImage" : srcImage,"srcImageLayout" : srcImageLayout,"dstImage" : dstImage,"dstImageLayout" : dstImageLayout,"regionCount" : regionCount,"pRegions" : pRegions,"retval" : retval}
def vkCmdSetEvent(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    if "stageMask" in indict.keys():
         stageMask = indict["stageMask"]
    else: 
         stageMask = c_uint()
    print(jvulkanLib.vkCmdSetEvent)
    retval = jvulkanLib.vkCmdSetEvent(commandBuffer, event, stageMask)
    return {"commandBuffer" : commandBuffer,"event" : event,"stageMask" : stageMask,"retval" : retval}
def vkCmdResetEvent(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    if "stageMask" in indict.keys():
         stageMask = indict["stageMask"]
    else: 
         stageMask = c_uint()
    print(jvulkanLib.vkCmdResetEvent)
    retval = jvulkanLib.vkCmdResetEvent(commandBuffer, event, stageMask)
    return {"commandBuffer" : commandBuffer,"event" : event,"stageMask" : stageMask,"retval" : retval}
def vkCmdWaitEvents(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "eventCount" in indict.keys():
         eventCount = indict["eventCount"]
    else: 
         eventCount = c_uint()
    if "pEvents" in indict.keys():
         pEvents = indict["pEvents"]
    else: 
         pEvents = pointer(VkEvent_T())
    if "srcStageMask" in indict.keys():
         srcStageMask = indict["srcStageMask"]
    else: 
         srcStageMask = c_uint()
    if "dstStageMask" in indict.keys():
         dstStageMask = indict["dstStageMask"]
    else: 
         dstStageMask = c_uint()
    if "memoryBarrierCount" in indict.keys():
         memoryBarrierCount = indict["memoryBarrierCount"]
    else: 
         memoryBarrierCount = c_uint()
    if "pMemoryBarriers" in indict.keys():
         pMemoryBarriers = indict["pMemoryBarriers"]
    else: 
         pMemoryBarriers = VkMemoryBarrier()
    if "bufferMemoryBarrierCount" in indict.keys():
         bufferMemoryBarrierCount = indict["bufferMemoryBarrierCount"]
    else: 
         bufferMemoryBarrierCount = c_uint()
    if "pBufferMemoryBarriers" in indict.keys():
         pBufferMemoryBarriers = indict["pBufferMemoryBarriers"]
    else: 
         pBufferMemoryBarriers = VkBufferMemoryBarrier()
    if "imageMemoryBarrierCount" in indict.keys():
         imageMemoryBarrierCount = indict["imageMemoryBarrierCount"]
    else: 
         imageMemoryBarrierCount = c_uint()
    if "pImageMemoryBarriers" in indict.keys():
         pImageMemoryBarriers = indict["pImageMemoryBarriers"]
    else: 
         pImageMemoryBarriers = VkImageMemoryBarrier()
    print(jvulkanLib.vkCmdWaitEvents)
    retval = jvulkanLib.vkCmdWaitEvents(commandBuffer, eventCount, pEvents, srcStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers)
    return {"commandBuffer" : commandBuffer,"eventCount" : eventCount,"pEvents" : pEvents,"srcStageMask" : srcStageMask,"dstStageMask" : dstStageMask,"memoryBarrierCount" : memoryBarrierCount,"pMemoryBarriers" : pMemoryBarriers,"bufferMemoryBarrierCount" : bufferMemoryBarrierCount,"pBufferMemoryBarriers" : pBufferMemoryBarriers,"imageMemoryBarrierCount" : imageMemoryBarrierCount,"pImageMemoryBarriers" : pImageMemoryBarriers,"retval" : retval}
def vkCmdPipelineBarrier(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "srcStageMask" in indict.keys():
         srcStageMask = indict["srcStageMask"]
    else: 
         srcStageMask = c_uint()
    if "dstStageMask" in indict.keys():
         dstStageMask = indict["dstStageMask"]
    else: 
         dstStageMask = c_uint()
    if "dependencyFlags" in indict.keys():
         dependencyFlags = indict["dependencyFlags"]
    else: 
         dependencyFlags = c_uint()
    if "memoryBarrierCount" in indict.keys():
         memoryBarrierCount = indict["memoryBarrierCount"]
    else: 
         memoryBarrierCount = c_uint()
    if "pMemoryBarriers" in indict.keys():
         pMemoryBarriers = indict["pMemoryBarriers"]
    else: 
         pMemoryBarriers = VkMemoryBarrier()
    if "bufferMemoryBarrierCount" in indict.keys():
         bufferMemoryBarrierCount = indict["bufferMemoryBarrierCount"]
    else: 
         bufferMemoryBarrierCount = c_uint()
    if "pBufferMemoryBarriers" in indict.keys():
         pBufferMemoryBarriers = indict["pBufferMemoryBarriers"]
    else: 
         pBufferMemoryBarriers = VkBufferMemoryBarrier()
    if "imageMemoryBarrierCount" in indict.keys():
         imageMemoryBarrierCount = indict["imageMemoryBarrierCount"]
    else: 
         imageMemoryBarrierCount = c_uint()
    if "pImageMemoryBarriers" in indict.keys():
         pImageMemoryBarriers = indict["pImageMemoryBarriers"]
    else: 
         pImageMemoryBarriers = VkImageMemoryBarrier()
    print(jvulkanLib.vkCmdPipelineBarrier)
    retval = jvulkanLib.vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers)
    return {"commandBuffer" : commandBuffer,"srcStageMask" : srcStageMask,"dstStageMask" : dstStageMask,"dependencyFlags" : dependencyFlags,"memoryBarrierCount" : memoryBarrierCount,"pMemoryBarriers" : pMemoryBarriers,"bufferMemoryBarrierCount" : bufferMemoryBarrierCount,"pBufferMemoryBarriers" : pBufferMemoryBarriers,"imageMemoryBarrierCount" : imageMemoryBarrierCount,"pImageMemoryBarriers" : pImageMemoryBarriers,"retval" : retval}
def vkCmdBeginQuery(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "query" in indict.keys():
         query = indict["query"]
    else: 
         query = c_uint()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    print(jvulkanLib.vkCmdBeginQuery)
    retval = jvulkanLib.vkCmdBeginQuery(commandBuffer, queryPool, query, flags)
    return {"commandBuffer" : commandBuffer,"queryPool" : queryPool,"query" : query,"flags" : flags,"retval" : retval}
def vkCmdEndQuery(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "query" in indict.keys():
         query = indict["query"]
    else: 
         query = c_uint()
    print(jvulkanLib.vkCmdEndQuery)
    retval = jvulkanLib.vkCmdEndQuery(commandBuffer, queryPool, query)
    return {"commandBuffer" : commandBuffer,"queryPool" : queryPool,"query" : query,"retval" : retval}
def vkCmdResetQueryPool(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "firstQuery" in indict.keys():
         firstQuery = indict["firstQuery"]
    else: 
         firstQuery = c_uint()
    if "queryCount" in indict.keys():
         queryCount = indict["queryCount"]
    else: 
         queryCount = c_uint()
    print(jvulkanLib.vkCmdResetQueryPool)
    retval = jvulkanLib.vkCmdResetQueryPool(commandBuffer, queryPool, firstQuery, queryCount)
    return {"commandBuffer" : commandBuffer,"queryPool" : queryPool,"firstQuery" : firstQuery,"queryCount" : queryCount,"retval" : retval}
def vkCmdWriteTimestamp(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pipelineStage" in indict.keys():
         pipelineStage = indict["pipelineStage"]
    else: 
         pipelineStage = c_int()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "query" in indict.keys():
         query = indict["query"]
    else: 
         query = c_uint()
    print(jvulkanLib.vkCmdWriteTimestamp)
    retval = jvulkanLib.vkCmdWriteTimestamp(commandBuffer, pipelineStage, queryPool, query)
    return {"commandBuffer" : commandBuffer,"pipelineStage" : pipelineStage,"queryPool" : queryPool,"query" : query,"retval" : retval}
def vkCmdCopyQueryPoolResults(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "firstQuery" in indict.keys():
         firstQuery = indict["firstQuery"]
    else: 
         firstQuery = c_uint()
    if "queryCount" in indict.keys():
         queryCount = indict["queryCount"]
    else: 
         queryCount = c_uint()
    if "dstBuffer" in indict.keys():
         dstBuffer = indict["dstBuffer"]
    else: 
         dstBuffer = VkBuffer_T()
    if "dstOffset" in indict.keys():
         dstOffset = indict["dstOffset"]
    else: 
         dstOffset = c_ulong()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_ulong()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    print(jvulkanLib.vkCmdCopyQueryPoolResults)
    retval = jvulkanLib.vkCmdCopyQueryPoolResults(commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags)
    return {"commandBuffer" : commandBuffer,"queryPool" : queryPool,"firstQuery" : firstQuery,"queryCount" : queryCount,"dstBuffer" : dstBuffer,"dstOffset" : dstOffset,"stride" : stride,"flags" : flags,"retval" : retval}
def vkCmdPushConstants(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "layout" in indict.keys():
         layout = indict["layout"]
    else: 
         layout = VkPipelineLayout_T()
    if "stageFlags" in indict.keys():
         stageFlags = indict["stageFlags"]
    else: 
         stageFlags = c_uint()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_uint()
    if "size" in indict.keys():
         size = indict["size"]
    else: 
         size = c_uint()
    if "pValues" in indict.keys():
         pValues = indict["pValues"]
    else: 
         pValues = c_void_p()
    print(jvulkanLib.vkCmdPushConstants)
    retval = jvulkanLib.vkCmdPushConstants(commandBuffer, layout, stageFlags, offset, size, pValues)
    return {"commandBuffer" : commandBuffer,"layout" : layout,"stageFlags" : stageFlags,"offset" : offset,"size" : size,"pValues" : pValues,"retval" : retval}
def vkCmdBeginRenderPass(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pRenderPassBegin" in indict.keys():
         pRenderPassBegin = indict["pRenderPassBegin"]
    else: 
         pRenderPassBegin = VkRenderPassBeginInfo()
    if "contents" in indict.keys():
         contents = indict["contents"]
    else: 
         contents = c_int()
    print(jvulkanLib.vkCmdBeginRenderPass)
    retval = jvulkanLib.vkCmdBeginRenderPass(commandBuffer, pRenderPassBegin, contents)
    return {"commandBuffer" : commandBuffer,"pRenderPassBegin" : pRenderPassBegin,"contents" : contents,"retval" : retval}
def vkCmdNextSubpass(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "contents" in indict.keys():
         contents = indict["contents"]
    else: 
         contents = c_int()
    print(jvulkanLib.vkCmdNextSubpass)
    retval = jvulkanLib.vkCmdNextSubpass(commandBuffer, contents)
    return {"commandBuffer" : commandBuffer,"contents" : contents,"retval" : retval}
def vkCmdEndRenderPass(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    print(jvulkanLib.vkCmdEndRenderPass)
    retval = jvulkanLib.vkCmdEndRenderPass(commandBuffer)
    return {"commandBuffer" : commandBuffer,"retval" : retval}
def vkCmdExecuteCommands(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "commandBufferCount" in indict.keys():
         commandBufferCount = indict["commandBufferCount"]
    else: 
         commandBufferCount = c_uint()
    if "pCommandBuffers" in indict.keys():
         pCommandBuffers = indict["pCommandBuffers"]
    else: 
         pCommandBuffers = pointer(VkCommandBuffer_T())
    print(jvulkanLib.vkCmdExecuteCommands)
    retval = jvulkanLib.vkCmdExecuteCommands(commandBuffer, commandBufferCount, pCommandBuffers)
    return {"commandBuffer" : commandBuffer,"commandBufferCount" : commandBufferCount,"pCommandBuffers" : pCommandBuffers,"retval" : retval}
def vkEnumerateInstanceVersion(indict):
    if "pApiVersion" in indict.keys():
         pApiVersion = indict["pApiVersion"]
    else: 
         pApiVersion = pointer(c_uint())
    print(jvulkanLib.vkEnumerateInstanceVersion)
    retval = jvulkanLib.vkEnumerateInstanceVersion(pApiVersion)
    return {"pApiVersion" : pApiVersion,"retval" : retval}
def vkBindBufferMemory2(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "bindInfoCount" in indict.keys():
         bindInfoCount = indict["bindInfoCount"]
    else: 
         bindInfoCount = c_uint()
    if "pBindInfos" in indict.keys():
         pBindInfos = indict["pBindInfos"]
    else: 
         pBindInfos = VkBindBufferMemoryInfo()
    print(jvulkanLib.vkBindBufferMemory2)
    retval = jvulkanLib.vkBindBufferMemory2(device, bindInfoCount, pBindInfos)
    return {"device" : device,"bindInfoCount" : bindInfoCount,"pBindInfos" : pBindInfos,"retval" : retval}
def vkBindImageMemory2(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "bindInfoCount" in indict.keys():
         bindInfoCount = indict["bindInfoCount"]
    else: 
         bindInfoCount = c_uint()
    if "pBindInfos" in indict.keys():
         pBindInfos = indict["pBindInfos"]
    else: 
         pBindInfos = VkBindImageMemoryInfo()
    print(jvulkanLib.vkBindImageMemory2)
    retval = jvulkanLib.vkBindImageMemory2(device, bindInfoCount, pBindInfos)
    return {"device" : device,"bindInfoCount" : bindInfoCount,"pBindInfos" : pBindInfos,"retval" : retval}
def vkGetDeviceGroupPeerMemoryFeatures(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "heapIndex" in indict.keys():
         heapIndex = indict["heapIndex"]
    else: 
         heapIndex = c_uint()
    if "localDeviceIndex" in indict.keys():
         localDeviceIndex = indict["localDeviceIndex"]
    else: 
         localDeviceIndex = c_uint()
    if "remoteDeviceIndex" in indict.keys():
         remoteDeviceIndex = indict["remoteDeviceIndex"]
    else: 
         remoteDeviceIndex = c_uint()
    if "pPeerMemoryFeatures" in indict.keys():
         pPeerMemoryFeatures = indict["pPeerMemoryFeatures"]
    else: 
         pPeerMemoryFeatures = pointer(c_uint())
    print(jvulkanLib.vkGetDeviceGroupPeerMemoryFeatures)
    retval = jvulkanLib.vkGetDeviceGroupPeerMemoryFeatures(device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures)
    return {"device" : device,"heapIndex" : heapIndex,"localDeviceIndex" : localDeviceIndex,"remoteDeviceIndex" : remoteDeviceIndex,"pPeerMemoryFeatures" : pPeerMemoryFeatures,"retval" : retval}
def vkCmdSetDeviceMask(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "deviceMask" in indict.keys():
         deviceMask = indict["deviceMask"]
    else: 
         deviceMask = c_uint()
    print(jvulkanLib.vkCmdSetDeviceMask)
    retval = jvulkanLib.vkCmdSetDeviceMask(commandBuffer, deviceMask)
    return {"commandBuffer" : commandBuffer,"deviceMask" : deviceMask,"retval" : retval}
def vkCmdDispatchBase(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "baseGroupX" in indict.keys():
         baseGroupX = indict["baseGroupX"]
    else: 
         baseGroupX = c_uint()
    if "baseGroupY" in indict.keys():
         baseGroupY = indict["baseGroupY"]
    else: 
         baseGroupY = c_uint()
    if "baseGroupZ" in indict.keys():
         baseGroupZ = indict["baseGroupZ"]
    else: 
         baseGroupZ = c_uint()
    if "groupCountX" in indict.keys():
         groupCountX = indict["groupCountX"]
    else: 
         groupCountX = c_uint()
    if "groupCountY" in indict.keys():
         groupCountY = indict["groupCountY"]
    else: 
         groupCountY = c_uint()
    if "groupCountZ" in indict.keys():
         groupCountZ = indict["groupCountZ"]
    else: 
         groupCountZ = c_uint()
    print(jvulkanLib.vkCmdDispatchBase)
    retval = jvulkanLib.vkCmdDispatchBase(commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ)
    return {"commandBuffer" : commandBuffer,"baseGroupX" : baseGroupX,"baseGroupY" : baseGroupY,"baseGroupZ" : baseGroupZ,"groupCountX" : groupCountX,"groupCountY" : groupCountY,"groupCountZ" : groupCountZ,"retval" : retval}
def vkEnumeratePhysicalDeviceGroups(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "pPhysicalDeviceGroupCount" in indict.keys():
         pPhysicalDeviceGroupCount = indict["pPhysicalDeviceGroupCount"]
    else: 
         pPhysicalDeviceGroupCount = pointer(c_uint())
    if "pPhysicalDeviceGroupProperties" in indict.keys():
         pPhysicalDeviceGroupProperties = indict["pPhysicalDeviceGroupProperties"]
    else: 
         pPhysicalDeviceGroupProperties = VkPhysicalDeviceGroupProperties()
    print(jvulkanLib.vkEnumeratePhysicalDeviceGroups)
    retval = jvulkanLib.vkEnumeratePhysicalDeviceGroups(instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties)
    return {"instance" : instance,"pPhysicalDeviceGroupCount" : pPhysicalDeviceGroupCount,"pPhysicalDeviceGroupProperties" : pPhysicalDeviceGroupProperties,"retval" : retval}
def vkGetImageMemoryRequirements2(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkImageMemoryRequirementsInfo2()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetImageMemoryRequirements2)
    retval = jvulkanLib.vkGetImageMemoryRequirements2(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetBufferMemoryRequirements2(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkBufferMemoryRequirementsInfo2()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetBufferMemoryRequirements2)
    retval = jvulkanLib.vkGetBufferMemoryRequirements2(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetImageSparseMemoryRequirements2(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkImageSparseMemoryRequirementsInfo2()
    if "pSparseMemoryRequirementCount" in indict.keys():
         pSparseMemoryRequirementCount = indict["pSparseMemoryRequirementCount"]
    else: 
         pSparseMemoryRequirementCount = pointer(c_uint())
    if "pSparseMemoryRequirements" in indict.keys():
         pSparseMemoryRequirements = indict["pSparseMemoryRequirements"]
    else: 
         pSparseMemoryRequirements = VkSparseImageMemoryRequirements2()
    print(jvulkanLib.vkGetImageSparseMemoryRequirements2)
    retval = jvulkanLib.vkGetImageSparseMemoryRequirements2(device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pSparseMemoryRequirementCount" : pSparseMemoryRequirementCount,"pSparseMemoryRequirements" : pSparseMemoryRequirements,"retval" : retval}
def vkGetPhysicalDeviceFeatures2(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pFeatures" in indict.keys():
         pFeatures = indict["pFeatures"]
    else: 
         pFeatures = VkPhysicalDeviceFeatures2()
    print(jvulkanLib.vkGetPhysicalDeviceFeatures2)
    retval = jvulkanLib.vkGetPhysicalDeviceFeatures2(physicalDevice, pFeatures)
    return {"physicalDevice" : physicalDevice,"pFeatures" : pFeatures,"retval" : retval}
def vkGetPhysicalDeviceProperties2(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkPhysicalDeviceProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceProperties2)
    retval = jvulkanLib.vkGetPhysicalDeviceProperties2(physicalDevice, pProperties)
    return {"physicalDevice" : physicalDevice,"pProperties" : pProperties,"retval" : retval}
def vkGetPhysicalDeviceFormatProperties2(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "format" in indict.keys():
         format = indict["format"]
    else: 
         format = c_int()
    if "pFormatProperties" in indict.keys():
         pFormatProperties = indict["pFormatProperties"]
    else: 
         pFormatProperties = VkFormatProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceFormatProperties2)
    retval = jvulkanLib.vkGetPhysicalDeviceFormatProperties2(physicalDevice, format, pFormatProperties)
    return {"physicalDevice" : physicalDevice,"format" : format,"pFormatProperties" : pFormatProperties,"retval" : retval}
def vkGetPhysicalDeviceImageFormatProperties2(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pImageFormatInfo" in indict.keys():
         pImageFormatInfo = indict["pImageFormatInfo"]
    else: 
         pImageFormatInfo = VkPhysicalDeviceImageFormatInfo2()
    if "pImageFormatProperties" in indict.keys():
         pImageFormatProperties = indict["pImageFormatProperties"]
    else: 
         pImageFormatProperties = VkImageFormatProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceImageFormatProperties2)
    retval = jvulkanLib.vkGetPhysicalDeviceImageFormatProperties2(physicalDevice, pImageFormatInfo, pImageFormatProperties)
    return {"physicalDevice" : physicalDevice,"pImageFormatInfo" : pImageFormatInfo,"pImageFormatProperties" : pImageFormatProperties,"retval" : retval}
def vkGetPhysicalDeviceQueueFamilyProperties2(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pQueueFamilyPropertyCount" in indict.keys():
         pQueueFamilyPropertyCount = indict["pQueueFamilyPropertyCount"]
    else: 
         pQueueFamilyPropertyCount = pointer(c_uint())
    if "pQueueFamilyProperties" in indict.keys():
         pQueueFamilyProperties = indict["pQueueFamilyProperties"]
    else: 
         pQueueFamilyProperties = VkQueueFamilyProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceQueueFamilyProperties2)
    retval = jvulkanLib.vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties)
    return {"physicalDevice" : physicalDevice,"pQueueFamilyPropertyCount" : pQueueFamilyPropertyCount,"pQueueFamilyProperties" : pQueueFamilyProperties,"retval" : retval}
def vkGetPhysicalDeviceMemoryProperties2(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pMemoryProperties" in indict.keys():
         pMemoryProperties = indict["pMemoryProperties"]
    else: 
         pMemoryProperties = VkPhysicalDeviceMemoryProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceMemoryProperties2)
    retval = jvulkanLib.vkGetPhysicalDeviceMemoryProperties2(physicalDevice, pMemoryProperties)
    return {"physicalDevice" : physicalDevice,"pMemoryProperties" : pMemoryProperties,"retval" : retval}
def vkGetPhysicalDeviceSparseImageFormatProperties2(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pFormatInfo" in indict.keys():
         pFormatInfo = indict["pFormatInfo"]
    else: 
         pFormatInfo = VkPhysicalDeviceSparseImageFormatInfo2()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkSparseImageFormatProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceSparseImageFormatProperties2)
    retval = jvulkanLib.vkGetPhysicalDeviceSparseImageFormatProperties2(physicalDevice, pFormatInfo, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"pFormatInfo" : pFormatInfo,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkTrimCommandPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "commandPool" in indict.keys():
         commandPool = indict["commandPool"]
    else: 
         commandPool = VkCommandPool_T()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    print(jvulkanLib.vkTrimCommandPool)
    retval = jvulkanLib.vkTrimCommandPool(device, commandPool, flags)
    return {"device" : device,"commandPool" : commandPool,"flags" : flags,"retval" : retval}
def vkGetDeviceQueue2(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pQueueInfo" in indict.keys():
         pQueueInfo = indict["pQueueInfo"]
    else: 
         pQueueInfo = VkDeviceQueueInfo2()
    if "pQueue" in indict.keys():
         pQueue = indict["pQueue"]
    else: 
         pQueue = pointer(VkQueue_T())
    print(jvulkanLib.vkGetDeviceQueue2)
    retval = jvulkanLib.vkGetDeviceQueue2(device, pQueueInfo, pQueue)
    return {"device" : device,"pQueueInfo" : pQueueInfo,"pQueue" : pQueue,"retval" : retval}
def vkCreateSamplerYcbcrConversion(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkSamplerYcbcrConversionCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pYcbcrConversion" in indict.keys():
         pYcbcrConversion = indict["pYcbcrConversion"]
    else: 
         pYcbcrConversion = pointer(VkSamplerYcbcrConversion_T())
    print(jvulkanLib.vkCreateSamplerYcbcrConversion)
    retval = jvulkanLib.vkCreateSamplerYcbcrConversion(device, pCreateInfo, pAllocator, pYcbcrConversion)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pYcbcrConversion" : pYcbcrConversion,"retval" : retval}
def vkDestroySamplerYcbcrConversion(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "ycbcrConversion" in indict.keys():
         ycbcrConversion = indict["ycbcrConversion"]
    else: 
         ycbcrConversion = VkSamplerYcbcrConversion_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroySamplerYcbcrConversion)
    retval = jvulkanLib.vkDestroySamplerYcbcrConversion(device, ycbcrConversion, pAllocator)
    return {"device" : device,"ycbcrConversion" : ycbcrConversion,"pAllocator" : pAllocator,"retval" : retval}
def vkCreateDescriptorUpdateTemplate(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDescriptorUpdateTemplateCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pDescriptorUpdateTemplate" in indict.keys():
         pDescriptorUpdateTemplate = indict["pDescriptorUpdateTemplate"]
    else: 
         pDescriptorUpdateTemplate = pointer(VkDescriptorUpdateTemplate_T())
    print(jvulkanLib.vkCreateDescriptorUpdateTemplate)
    retval = jvulkanLib.vkCreateDescriptorUpdateTemplate(device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pDescriptorUpdateTemplate" : pDescriptorUpdateTemplate,"retval" : retval}
def vkDestroyDescriptorUpdateTemplate(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorUpdateTemplate" in indict.keys():
         descriptorUpdateTemplate = indict["descriptorUpdateTemplate"]
    else: 
         descriptorUpdateTemplate = VkDescriptorUpdateTemplate_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyDescriptorUpdateTemplate)
    retval = jvulkanLib.vkDestroyDescriptorUpdateTemplate(device, descriptorUpdateTemplate, pAllocator)
    return {"device" : device,"descriptorUpdateTemplate" : descriptorUpdateTemplate,"pAllocator" : pAllocator,"retval" : retval}
def vkUpdateDescriptorSetWithTemplate(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorSet" in indict.keys():
         descriptorSet = indict["descriptorSet"]
    else: 
         descriptorSet = VkDescriptorSet_T()
    if "descriptorUpdateTemplate" in indict.keys():
         descriptorUpdateTemplate = indict["descriptorUpdateTemplate"]
    else: 
         descriptorUpdateTemplate = VkDescriptorUpdateTemplate_T()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkUpdateDescriptorSetWithTemplate)
    retval = jvulkanLib.vkUpdateDescriptorSetWithTemplate(device, descriptorSet, descriptorUpdateTemplate, pData)
    return {"device" : device,"descriptorSet" : descriptorSet,"descriptorUpdateTemplate" : descriptorUpdateTemplate,"pData" : pData,"retval" : retval}
def vkGetPhysicalDeviceExternalBufferProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pExternalBufferInfo" in indict.keys():
         pExternalBufferInfo = indict["pExternalBufferInfo"]
    else: 
         pExternalBufferInfo = VkPhysicalDeviceExternalBufferInfo()
    if "pExternalBufferProperties" in indict.keys():
         pExternalBufferProperties = indict["pExternalBufferProperties"]
    else: 
         pExternalBufferProperties = VkExternalBufferProperties()
    print(jvulkanLib.vkGetPhysicalDeviceExternalBufferProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceExternalBufferProperties(physicalDevice, pExternalBufferInfo, pExternalBufferProperties)
    return {"physicalDevice" : physicalDevice,"pExternalBufferInfo" : pExternalBufferInfo,"pExternalBufferProperties" : pExternalBufferProperties,"retval" : retval}
def vkGetPhysicalDeviceExternalFenceProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pExternalFenceInfo" in indict.keys():
         pExternalFenceInfo = indict["pExternalFenceInfo"]
    else: 
         pExternalFenceInfo = VkPhysicalDeviceExternalFenceInfo()
    if "pExternalFenceProperties" in indict.keys():
         pExternalFenceProperties = indict["pExternalFenceProperties"]
    else: 
         pExternalFenceProperties = VkExternalFenceProperties()
    print(jvulkanLib.vkGetPhysicalDeviceExternalFenceProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceExternalFenceProperties(physicalDevice, pExternalFenceInfo, pExternalFenceProperties)
    return {"physicalDevice" : physicalDevice,"pExternalFenceInfo" : pExternalFenceInfo,"pExternalFenceProperties" : pExternalFenceProperties,"retval" : retval}
def vkGetPhysicalDeviceExternalSemaphoreProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pExternalSemaphoreInfo" in indict.keys():
         pExternalSemaphoreInfo = indict["pExternalSemaphoreInfo"]
    else: 
         pExternalSemaphoreInfo = VkPhysicalDeviceExternalSemaphoreInfo()
    if "pExternalSemaphoreProperties" in indict.keys():
         pExternalSemaphoreProperties = indict["pExternalSemaphoreProperties"]
    else: 
         pExternalSemaphoreProperties = VkExternalSemaphoreProperties()
    print(jvulkanLib.vkGetPhysicalDeviceExternalSemaphoreProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceExternalSemaphoreProperties(physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties)
    return {"physicalDevice" : physicalDevice,"pExternalSemaphoreInfo" : pExternalSemaphoreInfo,"pExternalSemaphoreProperties" : pExternalSemaphoreProperties,"retval" : retval}
def vkGetDescriptorSetLayoutSupport(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDescriptorSetLayoutCreateInfo()
    if "pSupport" in indict.keys():
         pSupport = indict["pSupport"]
    else: 
         pSupport = VkDescriptorSetLayoutSupport()
    print(jvulkanLib.vkGetDescriptorSetLayoutSupport)
    retval = jvulkanLib.vkGetDescriptorSetLayoutSupport(device, pCreateInfo, pSupport)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pSupport" : pSupport,"retval" : retval}
def vkCmdDrawIndirectCount(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "countBuffer" in indict.keys():
         countBuffer = indict["countBuffer"]
    else: 
         countBuffer = VkBuffer_T()
    if "countBufferOffset" in indict.keys():
         countBufferOffset = indict["countBufferOffset"]
    else: 
         countBufferOffset = c_ulong()
    if "maxDrawCount" in indict.keys():
         maxDrawCount = indict["maxDrawCount"]
    else: 
         maxDrawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawIndirectCount)
    retval = jvulkanLib.vkCmdDrawIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"countBuffer" : countBuffer,"countBufferOffset" : countBufferOffset,"maxDrawCount" : maxDrawCount,"stride" : stride,"retval" : retval}
def vkCmdDrawIndexedIndirectCount(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "countBuffer" in indict.keys():
         countBuffer = indict["countBuffer"]
    else: 
         countBuffer = VkBuffer_T()
    if "countBufferOffset" in indict.keys():
         countBufferOffset = indict["countBufferOffset"]
    else: 
         countBufferOffset = c_ulong()
    if "maxDrawCount" in indict.keys():
         maxDrawCount = indict["maxDrawCount"]
    else: 
         maxDrawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawIndexedIndirectCount)
    retval = jvulkanLib.vkCmdDrawIndexedIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"countBuffer" : countBuffer,"countBufferOffset" : countBufferOffset,"maxDrawCount" : maxDrawCount,"stride" : stride,"retval" : retval}
def vkCreateRenderPass2(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkRenderPassCreateInfo2()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pRenderPass" in indict.keys():
         pRenderPass = indict["pRenderPass"]
    else: 
         pRenderPass = pointer(VkRenderPass_T())
    print(jvulkanLib.vkCreateRenderPass2)
    retval = jvulkanLib.vkCreateRenderPass2(device, pCreateInfo, pAllocator, pRenderPass)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pRenderPass" : pRenderPass,"retval" : retval}
def vkCmdBeginRenderPass2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pRenderPassBegin" in indict.keys():
         pRenderPassBegin = indict["pRenderPassBegin"]
    else: 
         pRenderPassBegin = VkRenderPassBeginInfo()
    if "pSubpassBeginInfo" in indict.keys():
         pSubpassBeginInfo = indict["pSubpassBeginInfo"]
    else: 
         pSubpassBeginInfo = VkSubpassBeginInfo()
    print(jvulkanLib.vkCmdBeginRenderPass2)
    retval = jvulkanLib.vkCmdBeginRenderPass2(commandBuffer, pRenderPassBegin, pSubpassBeginInfo)
    return {"commandBuffer" : commandBuffer,"pRenderPassBegin" : pRenderPassBegin,"pSubpassBeginInfo" : pSubpassBeginInfo,"retval" : retval}
def vkCmdNextSubpass2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pSubpassBeginInfo" in indict.keys():
         pSubpassBeginInfo = indict["pSubpassBeginInfo"]
    else: 
         pSubpassBeginInfo = VkSubpassBeginInfo()
    if "pSubpassEndInfo" in indict.keys():
         pSubpassEndInfo = indict["pSubpassEndInfo"]
    else: 
         pSubpassEndInfo = VkSubpassEndInfo()
    print(jvulkanLib.vkCmdNextSubpass2)
    retval = jvulkanLib.vkCmdNextSubpass2(commandBuffer, pSubpassBeginInfo, pSubpassEndInfo)
    return {"commandBuffer" : commandBuffer,"pSubpassBeginInfo" : pSubpassBeginInfo,"pSubpassEndInfo" : pSubpassEndInfo,"retval" : retval}
def vkCmdEndRenderPass2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pSubpassEndInfo" in indict.keys():
         pSubpassEndInfo = indict["pSubpassEndInfo"]
    else: 
         pSubpassEndInfo = VkSubpassEndInfo()
    print(jvulkanLib.vkCmdEndRenderPass2)
    retval = jvulkanLib.vkCmdEndRenderPass2(commandBuffer, pSubpassEndInfo)
    return {"commandBuffer" : commandBuffer,"pSubpassEndInfo" : pSubpassEndInfo,"retval" : retval}
def vkResetQueryPool(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "firstQuery" in indict.keys():
         firstQuery = indict["firstQuery"]
    else: 
         firstQuery = c_uint()
    if "queryCount" in indict.keys():
         queryCount = indict["queryCount"]
    else: 
         queryCount = c_uint()
    print(jvulkanLib.vkResetQueryPool)
    retval = jvulkanLib.vkResetQueryPool(device, queryPool, firstQuery, queryCount)
    return {"device" : device,"queryPool" : queryPool,"firstQuery" : firstQuery,"queryCount" : queryCount,"retval" : retval}
def vkGetSemaphoreCounterValue(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "semaphore" in indict.keys():
         semaphore = indict["semaphore"]
    else: 
         semaphore = VkSemaphore_T()
    if "pValue" in indict.keys():
         pValue = indict["pValue"]
    else: 
         pValue = pointer(c_ulong())
    print(jvulkanLib.vkGetSemaphoreCounterValue)
    retval = jvulkanLib.vkGetSemaphoreCounterValue(device, semaphore, pValue)
    return {"device" : device,"semaphore" : semaphore,"pValue" : pValue,"retval" : retval}
def vkWaitSemaphores(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pWaitInfo" in indict.keys():
         pWaitInfo = indict["pWaitInfo"]
    else: 
         pWaitInfo = VkSemaphoreWaitInfo()
    if "timeout" in indict.keys():
         timeout = indict["timeout"]
    else: 
         timeout = c_ulong()
    print(jvulkanLib.vkWaitSemaphores)
    retval = jvulkanLib.vkWaitSemaphores(device, pWaitInfo, timeout)
    return {"device" : device,"pWaitInfo" : pWaitInfo,"timeout" : timeout,"retval" : retval}
def vkSignalSemaphore(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pSignalInfo" in indict.keys():
         pSignalInfo = indict["pSignalInfo"]
    else: 
         pSignalInfo = VkSemaphoreSignalInfo()
    print(jvulkanLib.vkSignalSemaphore)
    retval = jvulkanLib.vkSignalSemaphore(device, pSignalInfo)
    return {"device" : device,"pSignalInfo" : pSignalInfo,"retval" : retval}
def vkGetBufferDeviceAddress(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkBufferDeviceAddressInfo()
    print(jvulkanLib.vkGetBufferDeviceAddress)
    retval = jvulkanLib.vkGetBufferDeviceAddress(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkGetBufferOpaqueCaptureAddress(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkBufferDeviceAddressInfo()
    print(jvulkanLib.vkGetBufferOpaqueCaptureAddress)
    retval = jvulkanLib.vkGetBufferOpaqueCaptureAddress(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkGetDeviceMemoryOpaqueCaptureAddress(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkDeviceMemoryOpaqueCaptureAddressInfo()
    print(jvulkanLib.vkGetDeviceMemoryOpaqueCaptureAddress)
    retval = jvulkanLib.vkGetDeviceMemoryOpaqueCaptureAddress(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkGetPhysicalDeviceToolProperties(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pToolCount" in indict.keys():
         pToolCount = indict["pToolCount"]
    else: 
         pToolCount = pointer(c_uint())
    if "pToolProperties" in indict.keys():
         pToolProperties = indict["pToolProperties"]
    else: 
         pToolProperties = VkPhysicalDeviceToolProperties()
    print(jvulkanLib.vkGetPhysicalDeviceToolProperties)
    retval = jvulkanLib.vkGetPhysicalDeviceToolProperties(physicalDevice, pToolCount, pToolProperties)
    return {"physicalDevice" : physicalDevice,"pToolCount" : pToolCount,"pToolProperties" : pToolProperties,"retval" : retval}
def vkCreatePrivateDataSlot(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkPrivateDataSlotCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pPrivateDataSlot" in indict.keys():
         pPrivateDataSlot = indict["pPrivateDataSlot"]
    else: 
         pPrivateDataSlot = pointer(VkPrivateDataSlot_T())
    print(jvulkanLib.vkCreatePrivateDataSlot)
    retval = jvulkanLib.vkCreatePrivateDataSlot(device, pCreateInfo, pAllocator, pPrivateDataSlot)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pPrivateDataSlot" : pPrivateDataSlot,"retval" : retval}
def vkDestroyPrivateDataSlot(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "privateDataSlot" in indict.keys():
         privateDataSlot = indict["privateDataSlot"]
    else: 
         privateDataSlot = VkPrivateDataSlot_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyPrivateDataSlot)
    retval = jvulkanLib.vkDestroyPrivateDataSlot(device, privateDataSlot, pAllocator)
    return {"device" : device,"privateDataSlot" : privateDataSlot,"pAllocator" : pAllocator,"retval" : retval}
def vkSetPrivateData(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "objectType" in indict.keys():
         objectType = indict["objectType"]
    else: 
         objectType = c_int()
    if "objectHandle" in indict.keys():
         objectHandle = indict["objectHandle"]
    else: 
         objectHandle = c_ulong()
    if "privateDataSlot" in indict.keys():
         privateDataSlot = indict["privateDataSlot"]
    else: 
         privateDataSlot = VkPrivateDataSlot_T()
    if "data" in indict.keys():
         data = indict["data"]
    else: 
         data = c_ulong()
    print(jvulkanLib.vkSetPrivateData)
    retval = jvulkanLib.vkSetPrivateData(device, objectType, objectHandle, privateDataSlot, data)
    return {"device" : device,"objectType" : objectType,"objectHandle" : objectHandle,"privateDataSlot" : privateDataSlot,"data" : data,"retval" : retval}
def vkGetPrivateData(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "objectType" in indict.keys():
         objectType = indict["objectType"]
    else: 
         objectType = c_int()
    if "objectHandle" in indict.keys():
         objectHandle = indict["objectHandle"]
    else: 
         objectHandle = c_ulong()
    if "privateDataSlot" in indict.keys():
         privateDataSlot = indict["privateDataSlot"]
    else: 
         privateDataSlot = VkPrivateDataSlot_T()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = pointer(c_ulong())
    print(jvulkanLib.vkGetPrivateData)
    retval = jvulkanLib.vkGetPrivateData(device, objectType, objectHandle, privateDataSlot, pData)
    return {"device" : device,"objectType" : objectType,"objectHandle" : objectHandle,"privateDataSlot" : privateDataSlot,"pData" : pData,"retval" : retval}
def vkCmdSetEvent2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    if "pDependencyInfo" in indict.keys():
         pDependencyInfo = indict["pDependencyInfo"]
    else: 
         pDependencyInfo = VkDependencyInfo()
    print(jvulkanLib.vkCmdSetEvent2)
    retval = jvulkanLib.vkCmdSetEvent2(commandBuffer, event, pDependencyInfo)
    return {"commandBuffer" : commandBuffer,"event" : event,"pDependencyInfo" : pDependencyInfo,"retval" : retval}
def vkCmdResetEvent2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    if "stageMask" in indict.keys():
         stageMask = indict["stageMask"]
    else: 
         stageMask = c_ulong()
    print(jvulkanLib.vkCmdResetEvent2)
    retval = jvulkanLib.vkCmdResetEvent2(commandBuffer, event, stageMask)
    return {"commandBuffer" : commandBuffer,"event" : event,"stageMask" : stageMask,"retval" : retval}
def vkCmdWaitEvents2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "eventCount" in indict.keys():
         eventCount = indict["eventCount"]
    else: 
         eventCount = c_uint()
    if "pEvents" in indict.keys():
         pEvents = indict["pEvents"]
    else: 
         pEvents = pointer(VkEvent_T())
    if "pDependencyInfos" in indict.keys():
         pDependencyInfos = indict["pDependencyInfos"]
    else: 
         pDependencyInfos = VkDependencyInfo()
    print(jvulkanLib.vkCmdWaitEvents2)
    retval = jvulkanLib.vkCmdWaitEvents2(commandBuffer, eventCount, pEvents, pDependencyInfos)
    return {"commandBuffer" : commandBuffer,"eventCount" : eventCount,"pEvents" : pEvents,"pDependencyInfos" : pDependencyInfos,"retval" : retval}
def vkCmdPipelineBarrier2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pDependencyInfo" in indict.keys():
         pDependencyInfo = indict["pDependencyInfo"]
    else: 
         pDependencyInfo = VkDependencyInfo()
    print(jvulkanLib.vkCmdPipelineBarrier2)
    retval = jvulkanLib.vkCmdPipelineBarrier2(commandBuffer, pDependencyInfo)
    return {"commandBuffer" : commandBuffer,"pDependencyInfo" : pDependencyInfo,"retval" : retval}
def vkCmdWriteTimestamp2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "stage" in indict.keys():
         stage = indict["stage"]
    else: 
         stage = c_ulong()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "query" in indict.keys():
         query = indict["query"]
    else: 
         query = c_uint()
    print(jvulkanLib.vkCmdWriteTimestamp2)
    retval = jvulkanLib.vkCmdWriteTimestamp2(commandBuffer, stage, queryPool, query)
    return {"commandBuffer" : commandBuffer,"stage" : stage,"queryPool" : queryPool,"query" : query,"retval" : retval}
def vkQueueSubmit2(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "submitCount" in indict.keys():
         submitCount = indict["submitCount"]
    else: 
         submitCount = c_uint()
    if "pSubmits" in indict.keys():
         pSubmits = indict["pSubmits"]
    else: 
         pSubmits = VkSubmitInfo2()
    if "fence" in indict.keys():
         fence = indict["fence"]
    else: 
         fence = VkFence_T()
    print(jvulkanLib.vkQueueSubmit2)
    retval = jvulkanLib.vkQueueSubmit2(queue, submitCount, pSubmits, fence)
    return {"queue" : queue,"submitCount" : submitCount,"pSubmits" : pSubmits,"fence" : fence,"retval" : retval}
def vkCmdCopyBuffer2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pCopyBufferInfo" in indict.keys():
         pCopyBufferInfo = indict["pCopyBufferInfo"]
    else: 
         pCopyBufferInfo = VkCopyBufferInfo2()
    print(jvulkanLib.vkCmdCopyBuffer2)
    retval = jvulkanLib.vkCmdCopyBuffer2(commandBuffer, pCopyBufferInfo)
    return {"commandBuffer" : commandBuffer,"pCopyBufferInfo" : pCopyBufferInfo,"retval" : retval}
def vkCmdCopyImage2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pCopyImageInfo" in indict.keys():
         pCopyImageInfo = indict["pCopyImageInfo"]
    else: 
         pCopyImageInfo = VkCopyImageInfo2()
    print(jvulkanLib.vkCmdCopyImage2)
    retval = jvulkanLib.vkCmdCopyImage2(commandBuffer, pCopyImageInfo)
    return {"commandBuffer" : commandBuffer,"pCopyImageInfo" : pCopyImageInfo,"retval" : retval}
def vkCmdCopyBufferToImage2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pCopyBufferToImageInfo" in indict.keys():
         pCopyBufferToImageInfo = indict["pCopyBufferToImageInfo"]
    else: 
         pCopyBufferToImageInfo = VkCopyBufferToImageInfo2()
    print(jvulkanLib.vkCmdCopyBufferToImage2)
    retval = jvulkanLib.vkCmdCopyBufferToImage2(commandBuffer, pCopyBufferToImageInfo)
    return {"commandBuffer" : commandBuffer,"pCopyBufferToImageInfo" : pCopyBufferToImageInfo,"retval" : retval}
def vkCmdCopyImageToBuffer2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pCopyImageToBufferInfo" in indict.keys():
         pCopyImageToBufferInfo = indict["pCopyImageToBufferInfo"]
    else: 
         pCopyImageToBufferInfo = VkCopyImageToBufferInfo2()
    print(jvulkanLib.vkCmdCopyImageToBuffer2)
    retval = jvulkanLib.vkCmdCopyImageToBuffer2(commandBuffer, pCopyImageToBufferInfo)
    return {"commandBuffer" : commandBuffer,"pCopyImageToBufferInfo" : pCopyImageToBufferInfo,"retval" : retval}
def vkCmdBlitImage2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pBlitImageInfo" in indict.keys():
         pBlitImageInfo = indict["pBlitImageInfo"]
    else: 
         pBlitImageInfo = VkBlitImageInfo2()
    print(jvulkanLib.vkCmdBlitImage2)
    retval = jvulkanLib.vkCmdBlitImage2(commandBuffer, pBlitImageInfo)
    return {"commandBuffer" : commandBuffer,"pBlitImageInfo" : pBlitImageInfo,"retval" : retval}
def vkCmdResolveImage2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pResolveImageInfo" in indict.keys():
         pResolveImageInfo = indict["pResolveImageInfo"]
    else: 
         pResolveImageInfo = VkResolveImageInfo2()
    print(jvulkanLib.vkCmdResolveImage2)
    retval = jvulkanLib.vkCmdResolveImage2(commandBuffer, pResolveImageInfo)
    return {"commandBuffer" : commandBuffer,"pResolveImageInfo" : pResolveImageInfo,"retval" : retval}
def vkCmdBeginRendering(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pRenderingInfo" in indict.keys():
         pRenderingInfo = indict["pRenderingInfo"]
    else: 
         pRenderingInfo = VkRenderingInfo()
    print(jvulkanLib.vkCmdBeginRendering)
    retval = jvulkanLib.vkCmdBeginRendering(commandBuffer, pRenderingInfo)
    return {"commandBuffer" : commandBuffer,"pRenderingInfo" : pRenderingInfo,"retval" : retval}
def vkCmdEndRendering(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    print(jvulkanLib.vkCmdEndRendering)
    retval = jvulkanLib.vkCmdEndRendering(commandBuffer)
    return {"commandBuffer" : commandBuffer,"retval" : retval}
def vkCmdSetCullMode(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "cullMode" in indict.keys():
         cullMode = indict["cullMode"]
    else: 
         cullMode = c_uint()
    print(jvulkanLib.vkCmdSetCullMode)
    retval = jvulkanLib.vkCmdSetCullMode(commandBuffer, cullMode)
    return {"commandBuffer" : commandBuffer,"cullMode" : cullMode,"retval" : retval}
def vkCmdSetFrontFace(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "frontFace" in indict.keys():
         frontFace = indict["frontFace"]
    else: 
         frontFace = c_int()
    print(jvulkanLib.vkCmdSetFrontFace)
    retval = jvulkanLib.vkCmdSetFrontFace(commandBuffer, frontFace)
    return {"commandBuffer" : commandBuffer,"frontFace" : frontFace,"retval" : retval}
def vkCmdSetPrimitiveTopology(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "primitiveTopology" in indict.keys():
         primitiveTopology = indict["primitiveTopology"]
    else: 
         primitiveTopology = c_int()
    print(jvulkanLib.vkCmdSetPrimitiveTopology)
    retval = jvulkanLib.vkCmdSetPrimitiveTopology(commandBuffer, primitiveTopology)
    return {"commandBuffer" : commandBuffer,"primitiveTopology" : primitiveTopology,"retval" : retval}
def vkCmdSetViewportWithCount(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "viewportCount" in indict.keys():
         viewportCount = indict["viewportCount"]
    else: 
         viewportCount = c_uint()
    if "pViewports" in indict.keys():
         pViewports = indict["pViewports"]
    else: 
         pViewports = VkViewport()
    print(jvulkanLib.vkCmdSetViewportWithCount)
    retval = jvulkanLib.vkCmdSetViewportWithCount(commandBuffer, viewportCount, pViewports)
    return {"commandBuffer" : commandBuffer,"viewportCount" : viewportCount,"pViewports" : pViewports,"retval" : retval}
def vkCmdSetScissorWithCount(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "scissorCount" in indict.keys():
         scissorCount = indict["scissorCount"]
    else: 
         scissorCount = c_uint()
    if "pScissors" in indict.keys():
         pScissors = indict["pScissors"]
    else: 
         pScissors = VkRect2D()
    print(jvulkanLib.vkCmdSetScissorWithCount)
    retval = jvulkanLib.vkCmdSetScissorWithCount(commandBuffer, scissorCount, pScissors)
    return {"commandBuffer" : commandBuffer,"scissorCount" : scissorCount,"pScissors" : pScissors,"retval" : retval}
def vkCmdBindVertexBuffers2(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstBinding" in indict.keys():
         firstBinding = indict["firstBinding"]
    else: 
         firstBinding = c_uint()
    if "bindingCount" in indict.keys():
         bindingCount = indict["bindingCount"]
    else: 
         bindingCount = c_uint()
    if "pBuffers" in indict.keys():
         pBuffers = indict["pBuffers"]
    else: 
         pBuffers = pointer(VkBuffer_T())
    if "pOffsets" in indict.keys():
         pOffsets = indict["pOffsets"]
    else: 
         pOffsets = pointer(c_ulong())
    if "pSizes" in indict.keys():
         pSizes = indict["pSizes"]
    else: 
         pSizes = pointer(c_ulong())
    if "pStrides" in indict.keys():
         pStrides = indict["pStrides"]
    else: 
         pStrides = pointer(c_ulong())
    print(jvulkanLib.vkCmdBindVertexBuffers2)
    retval = jvulkanLib.vkCmdBindVertexBuffers2(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes, pStrides)
    return {"commandBuffer" : commandBuffer,"firstBinding" : firstBinding,"bindingCount" : bindingCount,"pBuffers" : pBuffers,"pOffsets" : pOffsets,"pSizes" : pSizes,"pStrides" : pStrides,"retval" : retval}
def vkCmdSetDepthTestEnable(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthTestEnable" in indict.keys():
         depthTestEnable = indict["depthTestEnable"]
    else: 
         depthTestEnable = c_uint()
    print(jvulkanLib.vkCmdSetDepthTestEnable)
    retval = jvulkanLib.vkCmdSetDepthTestEnable(commandBuffer, depthTestEnable)
    return {"commandBuffer" : commandBuffer,"depthTestEnable" : depthTestEnable,"retval" : retval}
def vkCmdSetDepthWriteEnable(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthWriteEnable" in indict.keys():
         depthWriteEnable = indict["depthWriteEnable"]
    else: 
         depthWriteEnable = c_uint()
    print(jvulkanLib.vkCmdSetDepthWriteEnable)
    retval = jvulkanLib.vkCmdSetDepthWriteEnable(commandBuffer, depthWriteEnable)
    return {"commandBuffer" : commandBuffer,"depthWriteEnable" : depthWriteEnable,"retval" : retval}
def vkCmdSetDepthCompareOp(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthCompareOp" in indict.keys():
         depthCompareOp = indict["depthCompareOp"]
    else: 
         depthCompareOp = c_int()
    print(jvulkanLib.vkCmdSetDepthCompareOp)
    retval = jvulkanLib.vkCmdSetDepthCompareOp(commandBuffer, depthCompareOp)
    return {"commandBuffer" : commandBuffer,"depthCompareOp" : depthCompareOp,"retval" : retval}
def vkCmdSetDepthBoundsTestEnable(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthBoundsTestEnable" in indict.keys():
         depthBoundsTestEnable = indict["depthBoundsTestEnable"]
    else: 
         depthBoundsTestEnable = c_uint()
    print(jvulkanLib.vkCmdSetDepthBoundsTestEnable)
    retval = jvulkanLib.vkCmdSetDepthBoundsTestEnable(commandBuffer, depthBoundsTestEnable)
    return {"commandBuffer" : commandBuffer,"depthBoundsTestEnable" : depthBoundsTestEnable,"retval" : retval}
def vkCmdSetStencilTestEnable(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "stencilTestEnable" in indict.keys():
         stencilTestEnable = indict["stencilTestEnable"]
    else: 
         stencilTestEnable = c_uint()
    print(jvulkanLib.vkCmdSetStencilTestEnable)
    retval = jvulkanLib.vkCmdSetStencilTestEnable(commandBuffer, stencilTestEnable)
    return {"commandBuffer" : commandBuffer,"stencilTestEnable" : stencilTestEnable,"retval" : retval}
def vkCmdSetStencilOp(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "faceMask" in indict.keys():
         faceMask = indict["faceMask"]
    else: 
         faceMask = c_uint()
    if "failOp" in indict.keys():
         failOp = indict["failOp"]
    else: 
         failOp = c_int()
    if "passOp" in indict.keys():
         passOp = indict["passOp"]
    else: 
         passOp = c_int()
    if "depthFailOp" in indict.keys():
         depthFailOp = indict["depthFailOp"]
    else: 
         depthFailOp = c_int()
    if "compareOp" in indict.keys():
         compareOp = indict["compareOp"]
    else: 
         compareOp = c_int()
    print(jvulkanLib.vkCmdSetStencilOp)
    retval = jvulkanLib.vkCmdSetStencilOp(commandBuffer, faceMask, failOp, passOp, depthFailOp, compareOp)
    return {"commandBuffer" : commandBuffer,"faceMask" : faceMask,"failOp" : failOp,"passOp" : passOp,"depthFailOp" : depthFailOp,"compareOp" : compareOp,"retval" : retval}
def vkCmdSetRasterizerDiscardEnable(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "rasterizerDiscardEnable" in indict.keys():
         rasterizerDiscardEnable = indict["rasterizerDiscardEnable"]
    else: 
         rasterizerDiscardEnable = c_uint()
    print(jvulkanLib.vkCmdSetRasterizerDiscardEnable)
    retval = jvulkanLib.vkCmdSetRasterizerDiscardEnable(commandBuffer, rasterizerDiscardEnable)
    return {"commandBuffer" : commandBuffer,"rasterizerDiscardEnable" : rasterizerDiscardEnable,"retval" : retval}
def vkCmdSetDepthBiasEnable(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthBiasEnable" in indict.keys():
         depthBiasEnable = indict["depthBiasEnable"]
    else: 
         depthBiasEnable = c_uint()
    print(jvulkanLib.vkCmdSetDepthBiasEnable)
    retval = jvulkanLib.vkCmdSetDepthBiasEnable(commandBuffer, depthBiasEnable)
    return {"commandBuffer" : commandBuffer,"depthBiasEnable" : depthBiasEnable,"retval" : retval}
def vkCmdSetPrimitiveRestartEnable(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "primitiveRestartEnable" in indict.keys():
         primitiveRestartEnable = indict["primitiveRestartEnable"]
    else: 
         primitiveRestartEnable = c_uint()
    print(jvulkanLib.vkCmdSetPrimitiveRestartEnable)
    retval = jvulkanLib.vkCmdSetPrimitiveRestartEnable(commandBuffer, primitiveRestartEnable)
    return {"commandBuffer" : commandBuffer,"primitiveRestartEnable" : primitiveRestartEnable,"retval" : retval}
def vkGetDeviceBufferMemoryRequirements(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkDeviceBufferMemoryRequirements()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetDeviceBufferMemoryRequirements)
    retval = jvulkanLib.vkGetDeviceBufferMemoryRequirements(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetDeviceImageMemoryRequirements(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkDeviceImageMemoryRequirements()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetDeviceImageMemoryRequirements)
    retval = jvulkanLib.vkGetDeviceImageMemoryRequirements(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetDeviceImageSparseMemoryRequirements(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkDeviceImageMemoryRequirements()
    if "pSparseMemoryRequirementCount" in indict.keys():
         pSparseMemoryRequirementCount = indict["pSparseMemoryRequirementCount"]
    else: 
         pSparseMemoryRequirementCount = pointer(c_uint())
    if "pSparseMemoryRequirements" in indict.keys():
         pSparseMemoryRequirements = indict["pSparseMemoryRequirements"]
    else: 
         pSparseMemoryRequirements = VkSparseImageMemoryRequirements2()
    print(jvulkanLib.vkGetDeviceImageSparseMemoryRequirements)
    retval = jvulkanLib.vkGetDeviceImageSparseMemoryRequirements(device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pSparseMemoryRequirementCount" : pSparseMemoryRequirementCount,"pSparseMemoryRequirements" : pSparseMemoryRequirements,"retval" : retval}
def vkDestroySurfaceKHR(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "surface" in indict.keys():
         surface = indict["surface"]
    else: 
         surface = VkSurfaceKHR_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroySurfaceKHR)
    retval = jvulkanLib.vkDestroySurfaceKHR(instance, surface, pAllocator)
    return {"instance" : instance,"surface" : surface,"pAllocator" : pAllocator,"retval" : retval}
def vkGetPhysicalDeviceSurfaceSupportKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "queueFamilyIndex" in indict.keys():
         queueFamilyIndex = indict["queueFamilyIndex"]
    else: 
         queueFamilyIndex = c_uint()
    if "surface" in indict.keys():
         surface = indict["surface"]
    else: 
         surface = VkSurfaceKHR_T()
    if "pSupported" in indict.keys():
         pSupported = indict["pSupported"]
    else: 
         pSupported = pointer(c_uint())
    print(jvulkanLib.vkGetPhysicalDeviceSurfaceSupportKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface, pSupported)
    return {"physicalDevice" : physicalDevice,"queueFamilyIndex" : queueFamilyIndex,"surface" : surface,"pSupported" : pSupported,"retval" : retval}
def vkGetPhysicalDeviceSurfaceCapabilitiesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "surface" in indict.keys():
         surface = indict["surface"]
    else: 
         surface = VkSurfaceKHR_T()
    if "pSurfaceCapabilities" in indict.keys():
         pSurfaceCapabilities = indict["pSurfaceCapabilities"]
    else: 
         pSurfaceCapabilities = VkSurfaceCapabilitiesKHR()
    print(jvulkanLib.vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, pSurfaceCapabilities)
    return {"physicalDevice" : physicalDevice,"surface" : surface,"pSurfaceCapabilities" : pSurfaceCapabilities,"retval" : retval}
def vkGetPhysicalDeviceSurfaceFormatsKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "surface" in indict.keys():
         surface = indict["surface"]
    else: 
         surface = VkSurfaceKHR_T()
    if "pSurfaceFormatCount" in indict.keys():
         pSurfaceFormatCount = indict["pSurfaceFormatCount"]
    else: 
         pSurfaceFormatCount = pointer(c_uint())
    if "pSurfaceFormats" in indict.keys():
         pSurfaceFormats = indict["pSurfaceFormats"]
    else: 
         pSurfaceFormats = VkSurfaceFormatKHR()
    print(jvulkanLib.vkGetPhysicalDeviceSurfaceFormatsKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats)
    return {"physicalDevice" : physicalDevice,"surface" : surface,"pSurfaceFormatCount" : pSurfaceFormatCount,"pSurfaceFormats" : pSurfaceFormats,"retval" : retval}
def vkGetPhysicalDeviceSurfacePresentModesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "surface" in indict.keys():
         surface = indict["surface"]
    else: 
         surface = VkSurfaceKHR_T()
    if "pPresentModeCount" in indict.keys():
         pPresentModeCount = indict["pPresentModeCount"]
    else: 
         pPresentModeCount = pointer(c_uint())
    if "pPresentModes" in indict.keys():
         pPresentModes = indict["pPresentModes"]
    else: 
         pPresentModes = pointer(c_int())
    print(jvulkanLib.vkGetPhysicalDeviceSurfacePresentModesKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pPresentModeCount, pPresentModes)
    return {"physicalDevice" : physicalDevice,"surface" : surface,"pPresentModeCount" : pPresentModeCount,"pPresentModes" : pPresentModes,"retval" : retval}
def vkCreateSwapchainKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkSwapchainCreateInfoKHR()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pSwapchain" in indict.keys():
         pSwapchain = indict["pSwapchain"]
    else: 
         pSwapchain = pointer(VkSwapchainKHR_T())
    print(jvulkanLib.vkCreateSwapchainKHR)
    retval = jvulkanLib.vkCreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pSwapchain" : pSwapchain,"retval" : retval}
def vkDestroySwapchainKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchain" in indict.keys():
         swapchain = indict["swapchain"]
    else: 
         swapchain = VkSwapchainKHR_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroySwapchainKHR)
    retval = jvulkanLib.vkDestroySwapchainKHR(device, swapchain, pAllocator)
    return {"device" : device,"swapchain" : swapchain,"pAllocator" : pAllocator,"retval" : retval}
def vkGetSwapchainImagesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchain" in indict.keys():
         swapchain = indict["swapchain"]
    else: 
         swapchain = VkSwapchainKHR_T()
    if "pSwapchainImageCount" in indict.keys():
         pSwapchainImageCount = indict["pSwapchainImageCount"]
    else: 
         pSwapchainImageCount = pointer(c_uint())
    if "pSwapchainImages" in indict.keys():
         pSwapchainImages = indict["pSwapchainImages"]
    else: 
         pSwapchainImages = pointer(VkImage_T())
    print(jvulkanLib.vkGetSwapchainImagesKHR)
    retval = jvulkanLib.vkGetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, pSwapchainImages)
    return {"device" : device,"swapchain" : swapchain,"pSwapchainImageCount" : pSwapchainImageCount,"pSwapchainImages" : pSwapchainImages,"retval" : retval}
def vkAcquireNextImageKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchain" in indict.keys():
         swapchain = indict["swapchain"]
    else: 
         swapchain = VkSwapchainKHR_T()
    if "timeout" in indict.keys():
         timeout = indict["timeout"]
    else: 
         timeout = c_ulong()
    if "semaphore" in indict.keys():
         semaphore = indict["semaphore"]
    else: 
         semaphore = VkSemaphore_T()
    if "fence" in indict.keys():
         fence = indict["fence"]
    else: 
         fence = VkFence_T()
    if "pImageIndex" in indict.keys():
         pImageIndex = indict["pImageIndex"]
    else: 
         pImageIndex = pointer(c_uint())
    print(jvulkanLib.vkAcquireNextImageKHR)
    retval = jvulkanLib.vkAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pImageIndex)
    return {"device" : device,"swapchain" : swapchain,"timeout" : timeout,"semaphore" : semaphore,"fence" : fence,"pImageIndex" : pImageIndex,"retval" : retval}
def vkQueuePresentKHR(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "pPresentInfo" in indict.keys():
         pPresentInfo = indict["pPresentInfo"]
    else: 
         pPresentInfo = VkPresentInfoKHR()
    print(jvulkanLib.vkQueuePresentKHR)
    retval = jvulkanLib.vkQueuePresentKHR(queue, pPresentInfo)
    return {"queue" : queue,"pPresentInfo" : pPresentInfo,"retval" : retval}
def vkGetDeviceGroupPresentCapabilitiesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pDeviceGroupPresentCapabilities" in indict.keys():
         pDeviceGroupPresentCapabilities = indict["pDeviceGroupPresentCapabilities"]
    else: 
         pDeviceGroupPresentCapabilities = VkDeviceGroupPresentCapabilitiesKHR()
    print(jvulkanLib.vkGetDeviceGroupPresentCapabilitiesKHR)
    retval = jvulkanLib.vkGetDeviceGroupPresentCapabilitiesKHR(device, pDeviceGroupPresentCapabilities)
    return {"device" : device,"pDeviceGroupPresentCapabilities" : pDeviceGroupPresentCapabilities,"retval" : retval}
def vkGetDeviceGroupSurfacePresentModesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "surface" in indict.keys():
         surface = indict["surface"]
    else: 
         surface = VkSurfaceKHR_T()
    if "pModes" in indict.keys():
         pModes = indict["pModes"]
    else: 
         pModes = pointer(c_uint())
    print(jvulkanLib.vkGetDeviceGroupSurfacePresentModesKHR)
    retval = jvulkanLib.vkGetDeviceGroupSurfacePresentModesKHR(device, surface, pModes)
    return {"device" : device,"surface" : surface,"pModes" : pModes,"retval" : retval}
def vkGetPhysicalDevicePresentRectanglesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "surface" in indict.keys():
         surface = indict["surface"]
    else: 
         surface = VkSurfaceKHR_T()
    if "pRectCount" in indict.keys():
         pRectCount = indict["pRectCount"]
    else: 
         pRectCount = pointer(c_uint())
    if "pRects" in indict.keys():
         pRects = indict["pRects"]
    else: 
         pRects = VkRect2D()
    print(jvulkanLib.vkGetPhysicalDevicePresentRectanglesKHR)
    retval = jvulkanLib.vkGetPhysicalDevicePresentRectanglesKHR(physicalDevice, surface, pRectCount, pRects)
    return {"physicalDevice" : physicalDevice,"surface" : surface,"pRectCount" : pRectCount,"pRects" : pRects,"retval" : retval}
def vkAcquireNextImage2KHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pAcquireInfo" in indict.keys():
         pAcquireInfo = indict["pAcquireInfo"]
    else: 
         pAcquireInfo = VkAcquireNextImageInfoKHR()
    if "pImageIndex" in indict.keys():
         pImageIndex = indict["pImageIndex"]
    else: 
         pImageIndex = pointer(c_uint())
    print(jvulkanLib.vkAcquireNextImage2KHR)
    retval = jvulkanLib.vkAcquireNextImage2KHR(device, pAcquireInfo, pImageIndex)
    return {"device" : device,"pAcquireInfo" : pAcquireInfo,"pImageIndex" : pImageIndex,"retval" : retval}
def vkGetPhysicalDeviceDisplayPropertiesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkDisplayPropertiesKHR()
    print(jvulkanLib.vkGetPhysicalDeviceDisplayPropertiesKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkGetPhysicalDeviceDisplayPlanePropertiesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkDisplayPlanePropertiesKHR()
    print(jvulkanLib.vkGetPhysicalDeviceDisplayPlanePropertiesKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkGetDisplayPlaneSupportedDisplaysKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "planeIndex" in indict.keys():
         planeIndex = indict["planeIndex"]
    else: 
         planeIndex = c_uint()
    if "pDisplayCount" in indict.keys():
         pDisplayCount = indict["pDisplayCount"]
    else: 
         pDisplayCount = pointer(c_uint())
    if "pDisplays" in indict.keys():
         pDisplays = indict["pDisplays"]
    else: 
         pDisplays = pointer(VkDisplayKHR_T())
    print(jvulkanLib.vkGetDisplayPlaneSupportedDisplaysKHR)
    retval = jvulkanLib.vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, pDisplayCount, pDisplays)
    return {"physicalDevice" : physicalDevice,"planeIndex" : planeIndex,"pDisplayCount" : pDisplayCount,"pDisplays" : pDisplays,"retval" : retval}
def vkGetDisplayModePropertiesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "display" in indict.keys():
         display = indict["display"]
    else: 
         display = VkDisplayKHR_T()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkDisplayModePropertiesKHR()
    print(jvulkanLib.vkGetDisplayModePropertiesKHR)
    retval = jvulkanLib.vkGetDisplayModePropertiesKHR(physicalDevice, display, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"display" : display,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkCreateDisplayModeKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "display" in indict.keys():
         display = indict["display"]
    else: 
         display = VkDisplayKHR_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDisplayModeCreateInfoKHR()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pMode" in indict.keys():
         pMode = indict["pMode"]
    else: 
         pMode = pointer(VkDisplayModeKHR_T())
    print(jvulkanLib.vkCreateDisplayModeKHR)
    retval = jvulkanLib.vkCreateDisplayModeKHR(physicalDevice, display, pCreateInfo, pAllocator, pMode)
    return {"physicalDevice" : physicalDevice,"display" : display,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pMode" : pMode,"retval" : retval}
def vkGetDisplayPlaneCapabilitiesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "mode" in indict.keys():
         mode = indict["mode"]
    else: 
         mode = VkDisplayModeKHR_T()
    if "planeIndex" in indict.keys():
         planeIndex = indict["planeIndex"]
    else: 
         planeIndex = c_uint()
    if "pCapabilities" in indict.keys():
         pCapabilities = indict["pCapabilities"]
    else: 
         pCapabilities = VkDisplayPlaneCapabilitiesKHR()
    print(jvulkanLib.vkGetDisplayPlaneCapabilitiesKHR)
    retval = jvulkanLib.vkGetDisplayPlaneCapabilitiesKHR(physicalDevice, mode, planeIndex, pCapabilities)
    return {"physicalDevice" : physicalDevice,"mode" : mode,"planeIndex" : planeIndex,"pCapabilities" : pCapabilities,"retval" : retval}
def vkCreateDisplayPlaneSurfaceKHR(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDisplaySurfaceCreateInfoKHR()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pSurface" in indict.keys():
         pSurface = indict["pSurface"]
    else: 
         pSurface = pointer(VkSurfaceKHR_T())
    print(jvulkanLib.vkCreateDisplayPlaneSurfaceKHR)
    retval = jvulkanLib.vkCreateDisplayPlaneSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface)
    return {"instance" : instance,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pSurface" : pSurface,"retval" : retval}
def vkCreateSharedSwapchainsKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchainCount" in indict.keys():
         swapchainCount = indict["swapchainCount"]
    else: 
         swapchainCount = c_uint()
    if "pCreateInfos" in indict.keys():
         pCreateInfos = indict["pCreateInfos"]
    else: 
         pCreateInfos = VkSwapchainCreateInfoKHR()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pSwapchains" in indict.keys():
         pSwapchains = indict["pSwapchains"]
    else: 
         pSwapchains = pointer(VkSwapchainKHR_T())
    print(jvulkanLib.vkCreateSharedSwapchainsKHR)
    retval = jvulkanLib.vkCreateSharedSwapchainsKHR(device, swapchainCount, pCreateInfos, pAllocator, pSwapchains)
    return {"device" : device,"swapchainCount" : swapchainCount,"pCreateInfos" : pCreateInfos,"pAllocator" : pAllocator,"pSwapchains" : pSwapchains,"retval" : retval}
def vkCmdBeginRenderingKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pRenderingInfo" in indict.keys():
         pRenderingInfo = indict["pRenderingInfo"]
    else: 
         pRenderingInfo = VkRenderingInfo()
    print(jvulkanLib.vkCmdBeginRenderingKHR)
    retval = jvulkanLib.vkCmdBeginRenderingKHR(commandBuffer, pRenderingInfo)
    return {"commandBuffer" : commandBuffer,"pRenderingInfo" : pRenderingInfo,"retval" : retval}
def vkCmdEndRenderingKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    print(jvulkanLib.vkCmdEndRenderingKHR)
    retval = jvulkanLib.vkCmdEndRenderingKHR(commandBuffer)
    return {"commandBuffer" : commandBuffer,"retval" : retval}
def vkGetPhysicalDeviceFeatures2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pFeatures" in indict.keys():
         pFeatures = indict["pFeatures"]
    else: 
         pFeatures = VkPhysicalDeviceFeatures2()
    print(jvulkanLib.vkGetPhysicalDeviceFeatures2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceFeatures2KHR(physicalDevice, pFeatures)
    return {"physicalDevice" : physicalDevice,"pFeatures" : pFeatures,"retval" : retval}
def vkGetPhysicalDeviceProperties2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkPhysicalDeviceProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceProperties2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceProperties2KHR(physicalDevice, pProperties)
    return {"physicalDevice" : physicalDevice,"pProperties" : pProperties,"retval" : retval}
def vkGetPhysicalDeviceFormatProperties2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "format" in indict.keys():
         format = indict["format"]
    else: 
         format = c_int()
    if "pFormatProperties" in indict.keys():
         pFormatProperties = indict["pFormatProperties"]
    else: 
         pFormatProperties = VkFormatProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceFormatProperties2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceFormatProperties2KHR(physicalDevice, format, pFormatProperties)
    return {"physicalDevice" : physicalDevice,"format" : format,"pFormatProperties" : pFormatProperties,"retval" : retval}
def vkGetPhysicalDeviceImageFormatProperties2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pImageFormatInfo" in indict.keys():
         pImageFormatInfo = indict["pImageFormatInfo"]
    else: 
         pImageFormatInfo = VkPhysicalDeviceImageFormatInfo2()
    if "pImageFormatProperties" in indict.keys():
         pImageFormatProperties = indict["pImageFormatProperties"]
    else: 
         pImageFormatProperties = VkImageFormatProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceImageFormatProperties2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceImageFormatProperties2KHR(physicalDevice, pImageFormatInfo, pImageFormatProperties)
    return {"physicalDevice" : physicalDevice,"pImageFormatInfo" : pImageFormatInfo,"pImageFormatProperties" : pImageFormatProperties,"retval" : retval}
def vkGetPhysicalDeviceQueueFamilyProperties2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pQueueFamilyPropertyCount" in indict.keys():
         pQueueFamilyPropertyCount = indict["pQueueFamilyPropertyCount"]
    else: 
         pQueueFamilyPropertyCount = pointer(c_uint())
    if "pQueueFamilyProperties" in indict.keys():
         pQueueFamilyProperties = indict["pQueueFamilyProperties"]
    else: 
         pQueueFamilyProperties = VkQueueFamilyProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceQueueFamilyProperties2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceQueueFamilyProperties2KHR(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties)
    return {"physicalDevice" : physicalDevice,"pQueueFamilyPropertyCount" : pQueueFamilyPropertyCount,"pQueueFamilyProperties" : pQueueFamilyProperties,"retval" : retval}
def vkGetPhysicalDeviceMemoryProperties2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pMemoryProperties" in indict.keys():
         pMemoryProperties = indict["pMemoryProperties"]
    else: 
         pMemoryProperties = VkPhysicalDeviceMemoryProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceMemoryProperties2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceMemoryProperties2KHR(physicalDevice, pMemoryProperties)
    return {"physicalDevice" : physicalDevice,"pMemoryProperties" : pMemoryProperties,"retval" : retval}
def vkGetPhysicalDeviceSparseImageFormatProperties2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pFormatInfo" in indict.keys():
         pFormatInfo = indict["pFormatInfo"]
    else: 
         pFormatInfo = VkPhysicalDeviceSparseImageFormatInfo2()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkSparseImageFormatProperties2()
    print(jvulkanLib.vkGetPhysicalDeviceSparseImageFormatProperties2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceSparseImageFormatProperties2KHR(physicalDevice, pFormatInfo, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"pFormatInfo" : pFormatInfo,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkGetDeviceGroupPeerMemoryFeaturesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "heapIndex" in indict.keys():
         heapIndex = indict["heapIndex"]
    else: 
         heapIndex = c_uint()
    if "localDeviceIndex" in indict.keys():
         localDeviceIndex = indict["localDeviceIndex"]
    else: 
         localDeviceIndex = c_uint()
    if "remoteDeviceIndex" in indict.keys():
         remoteDeviceIndex = indict["remoteDeviceIndex"]
    else: 
         remoteDeviceIndex = c_uint()
    if "pPeerMemoryFeatures" in indict.keys():
         pPeerMemoryFeatures = indict["pPeerMemoryFeatures"]
    else: 
         pPeerMemoryFeatures = pointer(c_uint())
    print(jvulkanLib.vkGetDeviceGroupPeerMemoryFeaturesKHR)
    retval = jvulkanLib.vkGetDeviceGroupPeerMemoryFeaturesKHR(device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures)
    return {"device" : device,"heapIndex" : heapIndex,"localDeviceIndex" : localDeviceIndex,"remoteDeviceIndex" : remoteDeviceIndex,"pPeerMemoryFeatures" : pPeerMemoryFeatures,"retval" : retval}
def vkCmdSetDeviceMaskKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "deviceMask" in indict.keys():
         deviceMask = indict["deviceMask"]
    else: 
         deviceMask = c_uint()
    print(jvulkanLib.vkCmdSetDeviceMaskKHR)
    retval = jvulkanLib.vkCmdSetDeviceMaskKHR(commandBuffer, deviceMask)
    return {"commandBuffer" : commandBuffer,"deviceMask" : deviceMask,"retval" : retval}
def vkCmdDispatchBaseKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "baseGroupX" in indict.keys():
         baseGroupX = indict["baseGroupX"]
    else: 
         baseGroupX = c_uint()
    if "baseGroupY" in indict.keys():
         baseGroupY = indict["baseGroupY"]
    else: 
         baseGroupY = c_uint()
    if "baseGroupZ" in indict.keys():
         baseGroupZ = indict["baseGroupZ"]
    else: 
         baseGroupZ = c_uint()
    if "groupCountX" in indict.keys():
         groupCountX = indict["groupCountX"]
    else: 
         groupCountX = c_uint()
    if "groupCountY" in indict.keys():
         groupCountY = indict["groupCountY"]
    else: 
         groupCountY = c_uint()
    if "groupCountZ" in indict.keys():
         groupCountZ = indict["groupCountZ"]
    else: 
         groupCountZ = c_uint()
    print(jvulkanLib.vkCmdDispatchBaseKHR)
    retval = jvulkanLib.vkCmdDispatchBaseKHR(commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ)
    return {"commandBuffer" : commandBuffer,"baseGroupX" : baseGroupX,"baseGroupY" : baseGroupY,"baseGroupZ" : baseGroupZ,"groupCountX" : groupCountX,"groupCountY" : groupCountY,"groupCountZ" : groupCountZ,"retval" : retval}
def vkTrimCommandPoolKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "commandPool" in indict.keys():
         commandPool = indict["commandPool"]
    else: 
         commandPool = VkCommandPool_T()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    print(jvulkanLib.vkTrimCommandPoolKHR)
    retval = jvulkanLib.vkTrimCommandPoolKHR(device, commandPool, flags)
    return {"device" : device,"commandPool" : commandPool,"flags" : flags,"retval" : retval}
def vkEnumeratePhysicalDeviceGroupsKHR(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "pPhysicalDeviceGroupCount" in indict.keys():
         pPhysicalDeviceGroupCount = indict["pPhysicalDeviceGroupCount"]
    else: 
         pPhysicalDeviceGroupCount = pointer(c_uint())
    if "pPhysicalDeviceGroupProperties" in indict.keys():
         pPhysicalDeviceGroupProperties = indict["pPhysicalDeviceGroupProperties"]
    else: 
         pPhysicalDeviceGroupProperties = VkPhysicalDeviceGroupProperties()
    print(jvulkanLib.vkEnumeratePhysicalDeviceGroupsKHR)
    retval = jvulkanLib.vkEnumeratePhysicalDeviceGroupsKHR(instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties)
    return {"instance" : instance,"pPhysicalDeviceGroupCount" : pPhysicalDeviceGroupCount,"pPhysicalDeviceGroupProperties" : pPhysicalDeviceGroupProperties,"retval" : retval}
def vkGetPhysicalDeviceExternalBufferPropertiesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pExternalBufferInfo" in indict.keys():
         pExternalBufferInfo = indict["pExternalBufferInfo"]
    else: 
         pExternalBufferInfo = VkPhysicalDeviceExternalBufferInfo()
    if "pExternalBufferProperties" in indict.keys():
         pExternalBufferProperties = indict["pExternalBufferProperties"]
    else: 
         pExternalBufferProperties = VkExternalBufferProperties()
    print(jvulkanLib.vkGetPhysicalDeviceExternalBufferPropertiesKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceExternalBufferPropertiesKHR(physicalDevice, pExternalBufferInfo, pExternalBufferProperties)
    return {"physicalDevice" : physicalDevice,"pExternalBufferInfo" : pExternalBufferInfo,"pExternalBufferProperties" : pExternalBufferProperties,"retval" : retval}
def vkGetMemoryFdKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pGetFdInfo" in indict.keys():
         pGetFdInfo = indict["pGetFdInfo"]
    else: 
         pGetFdInfo = VkMemoryGetFdInfoKHR()
    if "pFd" in indict.keys():
         pFd = indict["pFd"]
    else: 
         pFd = pointer(c_int())
    print(jvulkanLib.vkGetMemoryFdKHR)
    retval = jvulkanLib.vkGetMemoryFdKHR(device, pGetFdInfo, pFd)
    return {"device" : device,"pGetFdInfo" : pGetFdInfo,"pFd" : pFd,"retval" : retval}
def vkGetMemoryFdPropertiesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "handleType" in indict.keys():
         handleType = indict["handleType"]
    else: 
         handleType = c_int()
    if "fd" in indict.keys():
         fd = indict["fd"]
    else: 
         fd = c_int()
    if "pMemoryFdProperties" in indict.keys():
         pMemoryFdProperties = indict["pMemoryFdProperties"]
    else: 
         pMemoryFdProperties = VkMemoryFdPropertiesKHR()
    print(jvulkanLib.vkGetMemoryFdPropertiesKHR)
    retval = jvulkanLib.vkGetMemoryFdPropertiesKHR(device, handleType, fd, pMemoryFdProperties)
    return {"device" : device,"handleType" : handleType,"fd" : fd,"pMemoryFdProperties" : pMemoryFdProperties,"retval" : retval}
def vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pExternalSemaphoreInfo" in indict.keys():
         pExternalSemaphoreInfo = indict["pExternalSemaphoreInfo"]
    else: 
         pExternalSemaphoreInfo = VkPhysicalDeviceExternalSemaphoreInfo()
    if "pExternalSemaphoreProperties" in indict.keys():
         pExternalSemaphoreProperties = indict["pExternalSemaphoreProperties"]
    else: 
         pExternalSemaphoreProperties = VkExternalSemaphoreProperties()
    print(jvulkanLib.vkGetPhysicalDeviceExternalSemaphorePropertiesKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties)
    return {"physicalDevice" : physicalDevice,"pExternalSemaphoreInfo" : pExternalSemaphoreInfo,"pExternalSemaphoreProperties" : pExternalSemaphoreProperties,"retval" : retval}
def vkImportSemaphoreFdKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pImportSemaphoreFdInfo" in indict.keys():
         pImportSemaphoreFdInfo = indict["pImportSemaphoreFdInfo"]
    else: 
         pImportSemaphoreFdInfo = VkImportSemaphoreFdInfoKHR()
    print(jvulkanLib.vkImportSemaphoreFdKHR)
    retval = jvulkanLib.vkImportSemaphoreFdKHR(device, pImportSemaphoreFdInfo)
    return {"device" : device,"pImportSemaphoreFdInfo" : pImportSemaphoreFdInfo,"retval" : retval}
def vkGetSemaphoreFdKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pGetFdInfo" in indict.keys():
         pGetFdInfo = indict["pGetFdInfo"]
    else: 
         pGetFdInfo = VkSemaphoreGetFdInfoKHR()
    if "pFd" in indict.keys():
         pFd = indict["pFd"]
    else: 
         pFd = pointer(c_int())
    print(jvulkanLib.vkGetSemaphoreFdKHR)
    retval = jvulkanLib.vkGetSemaphoreFdKHR(device, pGetFdInfo, pFd)
    return {"device" : device,"pGetFdInfo" : pGetFdInfo,"pFd" : pFd,"retval" : retval}
def vkCmdPushDescriptorSetKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pipelineBindPoint" in indict.keys():
         pipelineBindPoint = indict["pipelineBindPoint"]
    else: 
         pipelineBindPoint = c_int()
    if "layout" in indict.keys():
         layout = indict["layout"]
    else: 
         layout = VkPipelineLayout_T()
    if "set" in indict.keys():
         set = indict["set"]
    else: 
         set = c_uint()
    if "descriptorWriteCount" in indict.keys():
         descriptorWriteCount = indict["descriptorWriteCount"]
    else: 
         descriptorWriteCount = c_uint()
    if "pDescriptorWrites" in indict.keys():
         pDescriptorWrites = indict["pDescriptorWrites"]
    else: 
         pDescriptorWrites = VkWriteDescriptorSet()
    print(jvulkanLib.vkCmdPushDescriptorSetKHR)
    retval = jvulkanLib.vkCmdPushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites)
    return {"commandBuffer" : commandBuffer,"pipelineBindPoint" : pipelineBindPoint,"layout" : layout,"set" : set,"descriptorWriteCount" : descriptorWriteCount,"pDescriptorWrites" : pDescriptorWrites,"retval" : retval}
def vkCmdPushDescriptorSetWithTemplateKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "descriptorUpdateTemplate" in indict.keys():
         descriptorUpdateTemplate = indict["descriptorUpdateTemplate"]
    else: 
         descriptorUpdateTemplate = VkDescriptorUpdateTemplate_T()
    if "layout" in indict.keys():
         layout = indict["layout"]
    else: 
         layout = VkPipelineLayout_T()
    if "set" in indict.keys():
         set = indict["set"]
    else: 
         set = c_uint()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkCmdPushDescriptorSetWithTemplateKHR)
    retval = jvulkanLib.vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, descriptorUpdateTemplate, layout, set, pData)
    return {"commandBuffer" : commandBuffer,"descriptorUpdateTemplate" : descriptorUpdateTemplate,"layout" : layout,"set" : set,"pData" : pData,"retval" : retval}
def vkCreateDescriptorUpdateTemplateKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDescriptorUpdateTemplateCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pDescriptorUpdateTemplate" in indict.keys():
         pDescriptorUpdateTemplate = indict["pDescriptorUpdateTemplate"]
    else: 
         pDescriptorUpdateTemplate = pointer(VkDescriptorUpdateTemplate_T())
    print(jvulkanLib.vkCreateDescriptorUpdateTemplateKHR)
    retval = jvulkanLib.vkCreateDescriptorUpdateTemplateKHR(device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pDescriptorUpdateTemplate" : pDescriptorUpdateTemplate,"retval" : retval}
def vkDestroyDescriptorUpdateTemplateKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorUpdateTemplate" in indict.keys():
         descriptorUpdateTemplate = indict["descriptorUpdateTemplate"]
    else: 
         descriptorUpdateTemplate = VkDescriptorUpdateTemplate_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyDescriptorUpdateTemplateKHR)
    retval = jvulkanLib.vkDestroyDescriptorUpdateTemplateKHR(device, descriptorUpdateTemplate, pAllocator)
    return {"device" : device,"descriptorUpdateTemplate" : descriptorUpdateTemplate,"pAllocator" : pAllocator,"retval" : retval}
def vkUpdateDescriptorSetWithTemplateKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorSet" in indict.keys():
         descriptorSet = indict["descriptorSet"]
    else: 
         descriptorSet = VkDescriptorSet_T()
    if "descriptorUpdateTemplate" in indict.keys():
         descriptorUpdateTemplate = indict["descriptorUpdateTemplate"]
    else: 
         descriptorUpdateTemplate = VkDescriptorUpdateTemplate_T()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkUpdateDescriptorSetWithTemplateKHR)
    retval = jvulkanLib.vkUpdateDescriptorSetWithTemplateKHR(device, descriptorSet, descriptorUpdateTemplate, pData)
    return {"device" : device,"descriptorSet" : descriptorSet,"descriptorUpdateTemplate" : descriptorUpdateTemplate,"pData" : pData,"retval" : retval}
def vkCreateRenderPass2KHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkRenderPassCreateInfo2()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pRenderPass" in indict.keys():
         pRenderPass = indict["pRenderPass"]
    else: 
         pRenderPass = pointer(VkRenderPass_T())
    print(jvulkanLib.vkCreateRenderPass2KHR)
    retval = jvulkanLib.vkCreateRenderPass2KHR(device, pCreateInfo, pAllocator, pRenderPass)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pRenderPass" : pRenderPass,"retval" : retval}
def vkCmdBeginRenderPass2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pRenderPassBegin" in indict.keys():
         pRenderPassBegin = indict["pRenderPassBegin"]
    else: 
         pRenderPassBegin = VkRenderPassBeginInfo()
    if "pSubpassBeginInfo" in indict.keys():
         pSubpassBeginInfo = indict["pSubpassBeginInfo"]
    else: 
         pSubpassBeginInfo = VkSubpassBeginInfo()
    print(jvulkanLib.vkCmdBeginRenderPass2KHR)
    retval = jvulkanLib.vkCmdBeginRenderPass2KHR(commandBuffer, pRenderPassBegin, pSubpassBeginInfo)
    return {"commandBuffer" : commandBuffer,"pRenderPassBegin" : pRenderPassBegin,"pSubpassBeginInfo" : pSubpassBeginInfo,"retval" : retval}
def vkCmdNextSubpass2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pSubpassBeginInfo" in indict.keys():
         pSubpassBeginInfo = indict["pSubpassBeginInfo"]
    else: 
         pSubpassBeginInfo = VkSubpassBeginInfo()
    if "pSubpassEndInfo" in indict.keys():
         pSubpassEndInfo = indict["pSubpassEndInfo"]
    else: 
         pSubpassEndInfo = VkSubpassEndInfo()
    print(jvulkanLib.vkCmdNextSubpass2KHR)
    retval = jvulkanLib.vkCmdNextSubpass2KHR(commandBuffer, pSubpassBeginInfo, pSubpassEndInfo)
    return {"commandBuffer" : commandBuffer,"pSubpassBeginInfo" : pSubpassBeginInfo,"pSubpassEndInfo" : pSubpassEndInfo,"retval" : retval}
def vkCmdEndRenderPass2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pSubpassEndInfo" in indict.keys():
         pSubpassEndInfo = indict["pSubpassEndInfo"]
    else: 
         pSubpassEndInfo = VkSubpassEndInfo()
    print(jvulkanLib.vkCmdEndRenderPass2KHR)
    retval = jvulkanLib.vkCmdEndRenderPass2KHR(commandBuffer, pSubpassEndInfo)
    return {"commandBuffer" : commandBuffer,"pSubpassEndInfo" : pSubpassEndInfo,"retval" : retval}
def vkGetSwapchainStatusKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchain" in indict.keys():
         swapchain = indict["swapchain"]
    else: 
         swapchain = VkSwapchainKHR_T()
    print(jvulkanLib.vkGetSwapchainStatusKHR)
    retval = jvulkanLib.vkGetSwapchainStatusKHR(device, swapchain)
    return {"device" : device,"swapchain" : swapchain,"retval" : retval}
def vkGetPhysicalDeviceExternalFencePropertiesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pExternalFenceInfo" in indict.keys():
         pExternalFenceInfo = indict["pExternalFenceInfo"]
    else: 
         pExternalFenceInfo = VkPhysicalDeviceExternalFenceInfo()
    if "pExternalFenceProperties" in indict.keys():
         pExternalFenceProperties = indict["pExternalFenceProperties"]
    else: 
         pExternalFenceProperties = VkExternalFenceProperties()
    print(jvulkanLib.vkGetPhysicalDeviceExternalFencePropertiesKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceExternalFencePropertiesKHR(physicalDevice, pExternalFenceInfo, pExternalFenceProperties)
    return {"physicalDevice" : physicalDevice,"pExternalFenceInfo" : pExternalFenceInfo,"pExternalFenceProperties" : pExternalFenceProperties,"retval" : retval}
def vkImportFenceFdKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pImportFenceFdInfo" in indict.keys():
         pImportFenceFdInfo = indict["pImportFenceFdInfo"]
    else: 
         pImportFenceFdInfo = VkImportFenceFdInfoKHR()
    print(jvulkanLib.vkImportFenceFdKHR)
    retval = jvulkanLib.vkImportFenceFdKHR(device, pImportFenceFdInfo)
    return {"device" : device,"pImportFenceFdInfo" : pImportFenceFdInfo,"retval" : retval}
def vkGetFenceFdKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pGetFdInfo" in indict.keys():
         pGetFdInfo = indict["pGetFdInfo"]
    else: 
         pGetFdInfo = VkFenceGetFdInfoKHR()
    if "pFd" in indict.keys():
         pFd = indict["pFd"]
    else: 
         pFd = pointer(c_int())
    print(jvulkanLib.vkGetFenceFdKHR)
    retval = jvulkanLib.vkGetFenceFdKHR(device, pGetFdInfo, pFd)
    return {"device" : device,"pGetFdInfo" : pGetFdInfo,"pFd" : pFd,"retval" : retval}
def vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "queueFamilyIndex" in indict.keys():
         queueFamilyIndex = indict["queueFamilyIndex"]
    else: 
         queueFamilyIndex = c_uint()
    if "pCounterCount" in indict.keys():
         pCounterCount = indict["pCounterCount"]
    else: 
         pCounterCount = pointer(c_uint())
    if "pCounters" in indict.keys():
         pCounters = indict["pCounters"]
    else: 
         pCounters = VkPerformanceCounterKHR()
    if "pCounterDescriptions" in indict.keys():
         pCounterDescriptions = indict["pCounterDescriptions"]
    else: 
         pCounterDescriptions = VkPerformanceCounterDescriptionKHR()
    print(jvulkanLib.vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)
    retval = jvulkanLib.vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(physicalDevice, queueFamilyIndex, pCounterCount, pCounters, pCounterDescriptions)
    return {"physicalDevice" : physicalDevice,"queueFamilyIndex" : queueFamilyIndex,"pCounterCount" : pCounterCount,"pCounters" : pCounters,"pCounterDescriptions" : pCounterDescriptions,"retval" : retval}
def vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pPerformanceQueryCreateInfo" in indict.keys():
         pPerformanceQueryCreateInfo = indict["pPerformanceQueryCreateInfo"]
    else: 
         pPerformanceQueryCreateInfo = VkQueryPoolPerformanceCreateInfoKHR()
    if "pNumPasses" in indict.keys():
         pNumPasses = indict["pNumPasses"]
    else: 
         pNumPasses = pointer(c_uint())
    print(jvulkanLib.vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(physicalDevice, pPerformanceQueryCreateInfo, pNumPasses)
    return {"physicalDevice" : physicalDevice,"pPerformanceQueryCreateInfo" : pPerformanceQueryCreateInfo,"pNumPasses" : pNumPasses,"retval" : retval}
def vkAcquireProfilingLockKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkAcquireProfilingLockInfoKHR()
    print(jvulkanLib.vkAcquireProfilingLockKHR)
    retval = jvulkanLib.vkAcquireProfilingLockKHR(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkReleaseProfilingLockKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    print(jvulkanLib.vkReleaseProfilingLockKHR)
    retval = jvulkanLib.vkReleaseProfilingLockKHR(device)
    return {"device" : device,"retval" : retval}
def vkGetPhysicalDeviceSurfaceCapabilities2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pSurfaceInfo" in indict.keys():
         pSurfaceInfo = indict["pSurfaceInfo"]
    else: 
         pSurfaceInfo = VkPhysicalDeviceSurfaceInfo2KHR()
    if "pSurfaceCapabilities" in indict.keys():
         pSurfaceCapabilities = indict["pSurfaceCapabilities"]
    else: 
         pSurfaceCapabilities = VkSurfaceCapabilities2KHR()
    print(jvulkanLib.vkGetPhysicalDeviceSurfaceCapabilities2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceSurfaceCapabilities2KHR(physicalDevice, pSurfaceInfo, pSurfaceCapabilities)
    return {"physicalDevice" : physicalDevice,"pSurfaceInfo" : pSurfaceInfo,"pSurfaceCapabilities" : pSurfaceCapabilities,"retval" : retval}
def vkGetPhysicalDeviceSurfaceFormats2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pSurfaceInfo" in indict.keys():
         pSurfaceInfo = indict["pSurfaceInfo"]
    else: 
         pSurfaceInfo = VkPhysicalDeviceSurfaceInfo2KHR()
    if "pSurfaceFormatCount" in indict.keys():
         pSurfaceFormatCount = indict["pSurfaceFormatCount"]
    else: 
         pSurfaceFormatCount = pointer(c_uint())
    if "pSurfaceFormats" in indict.keys():
         pSurfaceFormats = indict["pSurfaceFormats"]
    else: 
         pSurfaceFormats = VkSurfaceFormat2KHR()
    print(jvulkanLib.vkGetPhysicalDeviceSurfaceFormats2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceSurfaceFormats2KHR(physicalDevice, pSurfaceInfo, pSurfaceFormatCount, pSurfaceFormats)
    return {"physicalDevice" : physicalDevice,"pSurfaceInfo" : pSurfaceInfo,"pSurfaceFormatCount" : pSurfaceFormatCount,"pSurfaceFormats" : pSurfaceFormats,"retval" : retval}
def vkGetPhysicalDeviceDisplayProperties2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkDisplayProperties2KHR()
    print(jvulkanLib.vkGetPhysicalDeviceDisplayProperties2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceDisplayProperties2KHR(physicalDevice, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkGetPhysicalDeviceDisplayPlaneProperties2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkDisplayPlaneProperties2KHR()
    print(jvulkanLib.vkGetPhysicalDeviceDisplayPlaneProperties2KHR)
    retval = jvulkanLib.vkGetPhysicalDeviceDisplayPlaneProperties2KHR(physicalDevice, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkGetDisplayModeProperties2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "display" in indict.keys():
         display = indict["display"]
    else: 
         display = VkDisplayKHR_T()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkDisplayModeProperties2KHR()
    print(jvulkanLib.vkGetDisplayModeProperties2KHR)
    retval = jvulkanLib.vkGetDisplayModeProperties2KHR(physicalDevice, display, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"display" : display,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkGetDisplayPlaneCapabilities2KHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pDisplayPlaneInfo" in indict.keys():
         pDisplayPlaneInfo = indict["pDisplayPlaneInfo"]
    else: 
         pDisplayPlaneInfo = VkDisplayPlaneInfo2KHR()
    if "pCapabilities" in indict.keys():
         pCapabilities = indict["pCapabilities"]
    else: 
         pCapabilities = VkDisplayPlaneCapabilities2KHR()
    print(jvulkanLib.vkGetDisplayPlaneCapabilities2KHR)
    retval = jvulkanLib.vkGetDisplayPlaneCapabilities2KHR(physicalDevice, pDisplayPlaneInfo, pCapabilities)
    return {"physicalDevice" : physicalDevice,"pDisplayPlaneInfo" : pDisplayPlaneInfo,"pCapabilities" : pCapabilities,"retval" : retval}
def vkGetImageMemoryRequirements2KHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkImageMemoryRequirementsInfo2()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetImageMemoryRequirements2KHR)
    retval = jvulkanLib.vkGetImageMemoryRequirements2KHR(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetBufferMemoryRequirements2KHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkBufferMemoryRequirementsInfo2()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetBufferMemoryRequirements2KHR)
    retval = jvulkanLib.vkGetBufferMemoryRequirements2KHR(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetImageSparseMemoryRequirements2KHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkImageSparseMemoryRequirementsInfo2()
    if "pSparseMemoryRequirementCount" in indict.keys():
         pSparseMemoryRequirementCount = indict["pSparseMemoryRequirementCount"]
    else: 
         pSparseMemoryRequirementCount = pointer(c_uint())
    if "pSparseMemoryRequirements" in indict.keys():
         pSparseMemoryRequirements = indict["pSparseMemoryRequirements"]
    else: 
         pSparseMemoryRequirements = VkSparseImageMemoryRequirements2()
    print(jvulkanLib.vkGetImageSparseMemoryRequirements2KHR)
    retval = jvulkanLib.vkGetImageSparseMemoryRequirements2KHR(device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pSparseMemoryRequirementCount" : pSparseMemoryRequirementCount,"pSparseMemoryRequirements" : pSparseMemoryRequirements,"retval" : retval}
def vkCreateSamplerYcbcrConversionKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkSamplerYcbcrConversionCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pYcbcrConversion" in indict.keys():
         pYcbcrConversion = indict["pYcbcrConversion"]
    else: 
         pYcbcrConversion = pointer(VkSamplerYcbcrConversion_T())
    print(jvulkanLib.vkCreateSamplerYcbcrConversionKHR)
    retval = jvulkanLib.vkCreateSamplerYcbcrConversionKHR(device, pCreateInfo, pAllocator, pYcbcrConversion)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pYcbcrConversion" : pYcbcrConversion,"retval" : retval}
def vkDestroySamplerYcbcrConversionKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "ycbcrConversion" in indict.keys():
         ycbcrConversion = indict["ycbcrConversion"]
    else: 
         ycbcrConversion = VkSamplerYcbcrConversion_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroySamplerYcbcrConversionKHR)
    retval = jvulkanLib.vkDestroySamplerYcbcrConversionKHR(device, ycbcrConversion, pAllocator)
    return {"device" : device,"ycbcrConversion" : ycbcrConversion,"pAllocator" : pAllocator,"retval" : retval}
def vkBindBufferMemory2KHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "bindInfoCount" in indict.keys():
         bindInfoCount = indict["bindInfoCount"]
    else: 
         bindInfoCount = c_uint()
    if "pBindInfos" in indict.keys():
         pBindInfos = indict["pBindInfos"]
    else: 
         pBindInfos = VkBindBufferMemoryInfo()
    print(jvulkanLib.vkBindBufferMemory2KHR)
    retval = jvulkanLib.vkBindBufferMemory2KHR(device, bindInfoCount, pBindInfos)
    return {"device" : device,"bindInfoCount" : bindInfoCount,"pBindInfos" : pBindInfos,"retval" : retval}
def vkBindImageMemory2KHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "bindInfoCount" in indict.keys():
         bindInfoCount = indict["bindInfoCount"]
    else: 
         bindInfoCount = c_uint()
    if "pBindInfos" in indict.keys():
         pBindInfos = indict["pBindInfos"]
    else: 
         pBindInfos = VkBindImageMemoryInfo()
    print(jvulkanLib.vkBindImageMemory2KHR)
    retval = jvulkanLib.vkBindImageMemory2KHR(device, bindInfoCount, pBindInfos)
    return {"device" : device,"bindInfoCount" : bindInfoCount,"pBindInfos" : pBindInfos,"retval" : retval}
def vkGetDescriptorSetLayoutSupportKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDescriptorSetLayoutCreateInfo()
    if "pSupport" in indict.keys():
         pSupport = indict["pSupport"]
    else: 
         pSupport = VkDescriptorSetLayoutSupport()
    print(jvulkanLib.vkGetDescriptorSetLayoutSupportKHR)
    retval = jvulkanLib.vkGetDescriptorSetLayoutSupportKHR(device, pCreateInfo, pSupport)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pSupport" : pSupport,"retval" : retval}
def vkCmdDrawIndirectCountKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "countBuffer" in indict.keys():
         countBuffer = indict["countBuffer"]
    else: 
         countBuffer = VkBuffer_T()
    if "countBufferOffset" in indict.keys():
         countBufferOffset = indict["countBufferOffset"]
    else: 
         countBufferOffset = c_ulong()
    if "maxDrawCount" in indict.keys():
         maxDrawCount = indict["maxDrawCount"]
    else: 
         maxDrawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawIndirectCountKHR)
    retval = jvulkanLib.vkCmdDrawIndirectCountKHR(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"countBuffer" : countBuffer,"countBufferOffset" : countBufferOffset,"maxDrawCount" : maxDrawCount,"stride" : stride,"retval" : retval}
def vkCmdDrawIndexedIndirectCountKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "countBuffer" in indict.keys():
         countBuffer = indict["countBuffer"]
    else: 
         countBuffer = VkBuffer_T()
    if "countBufferOffset" in indict.keys():
         countBufferOffset = indict["countBufferOffset"]
    else: 
         countBufferOffset = c_ulong()
    if "maxDrawCount" in indict.keys():
         maxDrawCount = indict["maxDrawCount"]
    else: 
         maxDrawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawIndexedIndirectCountKHR)
    retval = jvulkanLib.vkCmdDrawIndexedIndirectCountKHR(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"countBuffer" : countBuffer,"countBufferOffset" : countBufferOffset,"maxDrawCount" : maxDrawCount,"stride" : stride,"retval" : retval}
def vkGetSemaphoreCounterValueKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "semaphore" in indict.keys():
         semaphore = indict["semaphore"]
    else: 
         semaphore = VkSemaphore_T()
    if "pValue" in indict.keys():
         pValue = indict["pValue"]
    else: 
         pValue = pointer(c_ulong())
    print(jvulkanLib.vkGetSemaphoreCounterValueKHR)
    retval = jvulkanLib.vkGetSemaphoreCounterValueKHR(device, semaphore, pValue)
    return {"device" : device,"semaphore" : semaphore,"pValue" : pValue,"retval" : retval}
def vkWaitSemaphoresKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pWaitInfo" in indict.keys():
         pWaitInfo = indict["pWaitInfo"]
    else: 
         pWaitInfo = VkSemaphoreWaitInfo()
    if "timeout" in indict.keys():
         timeout = indict["timeout"]
    else: 
         timeout = c_ulong()
    print(jvulkanLib.vkWaitSemaphoresKHR)
    retval = jvulkanLib.vkWaitSemaphoresKHR(device, pWaitInfo, timeout)
    return {"device" : device,"pWaitInfo" : pWaitInfo,"timeout" : timeout,"retval" : retval}
def vkSignalSemaphoreKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pSignalInfo" in indict.keys():
         pSignalInfo = indict["pSignalInfo"]
    else: 
         pSignalInfo = VkSemaphoreSignalInfo()
    print(jvulkanLib.vkSignalSemaphoreKHR)
    retval = jvulkanLib.vkSignalSemaphoreKHR(device, pSignalInfo)
    return {"device" : device,"pSignalInfo" : pSignalInfo,"retval" : retval}
def vkGetPhysicalDeviceFragmentShadingRatesKHR(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pFragmentShadingRateCount" in indict.keys():
         pFragmentShadingRateCount = indict["pFragmentShadingRateCount"]
    else: 
         pFragmentShadingRateCount = pointer(c_uint())
    if "pFragmentShadingRates" in indict.keys():
         pFragmentShadingRates = indict["pFragmentShadingRates"]
    else: 
         pFragmentShadingRates = VkPhysicalDeviceFragmentShadingRateKHR()
    print(jvulkanLib.vkGetPhysicalDeviceFragmentShadingRatesKHR)
    retval = jvulkanLib.vkGetPhysicalDeviceFragmentShadingRatesKHR(physicalDevice, pFragmentShadingRateCount, pFragmentShadingRates)
    return {"physicalDevice" : physicalDevice,"pFragmentShadingRateCount" : pFragmentShadingRateCount,"pFragmentShadingRates" : pFragmentShadingRates,"retval" : retval}
def vkCmdSetFragmentShadingRateKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pFragmentSize" in indict.keys():
         pFragmentSize = indict["pFragmentSize"]
    else: 
         pFragmentSize = VkExtent2D()
    if "combinerOps" in indict.keys():
         combinerOps = indict["combinerOps"]
    else: 
         combinerOps = pointer(c_int())
    print(jvulkanLib.vkCmdSetFragmentShadingRateKHR)
    retval = jvulkanLib.vkCmdSetFragmentShadingRateKHR(commandBuffer, pFragmentSize, combinerOps)
    return {"commandBuffer" : commandBuffer,"pFragmentSize" : pFragmentSize,"combinerOps" : combinerOps,"retval" : retval}
def vkWaitForPresentKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchain" in indict.keys():
         swapchain = indict["swapchain"]
    else: 
         swapchain = VkSwapchainKHR_T()
    if "presentId" in indict.keys():
         presentId = indict["presentId"]
    else: 
         presentId = c_ulong()
    if "timeout" in indict.keys():
         timeout = indict["timeout"]
    else: 
         timeout = c_ulong()
    print(jvulkanLib.vkWaitForPresentKHR)
    retval = jvulkanLib.vkWaitForPresentKHR(device, swapchain, presentId, timeout)
    return {"device" : device,"swapchain" : swapchain,"presentId" : presentId,"timeout" : timeout,"retval" : retval}
def vkGetBufferDeviceAddressKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkBufferDeviceAddressInfo()
    print(jvulkanLib.vkGetBufferDeviceAddressKHR)
    retval = jvulkanLib.vkGetBufferDeviceAddressKHR(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkGetBufferOpaqueCaptureAddressKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkBufferDeviceAddressInfo()
    print(jvulkanLib.vkGetBufferOpaqueCaptureAddressKHR)
    retval = jvulkanLib.vkGetBufferOpaqueCaptureAddressKHR(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkGetDeviceMemoryOpaqueCaptureAddressKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkDeviceMemoryOpaqueCaptureAddressInfo()
    print(jvulkanLib.vkGetDeviceMemoryOpaqueCaptureAddressKHR)
    retval = jvulkanLib.vkGetDeviceMemoryOpaqueCaptureAddressKHR(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkCreateDeferredOperationKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pDeferredOperation" in indict.keys():
         pDeferredOperation = indict["pDeferredOperation"]
    else: 
         pDeferredOperation = pointer(VkDeferredOperationKHR_T())
    print(jvulkanLib.vkCreateDeferredOperationKHR)
    retval = jvulkanLib.vkCreateDeferredOperationKHR(device, pAllocator, pDeferredOperation)
    return {"device" : device,"pAllocator" : pAllocator,"pDeferredOperation" : pDeferredOperation,"retval" : retval}
def vkDestroyDeferredOperationKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "operation" in indict.keys():
         operation = indict["operation"]
    else: 
         operation = VkDeferredOperationKHR_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyDeferredOperationKHR)
    retval = jvulkanLib.vkDestroyDeferredOperationKHR(device, operation, pAllocator)
    return {"device" : device,"operation" : operation,"pAllocator" : pAllocator,"retval" : retval}
def vkGetDeferredOperationMaxConcurrencyKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "operation" in indict.keys():
         operation = indict["operation"]
    else: 
         operation = VkDeferredOperationKHR_T()
    print(jvulkanLib.vkGetDeferredOperationMaxConcurrencyKHR)
    retval = jvulkanLib.vkGetDeferredOperationMaxConcurrencyKHR(device, operation)
    return {"device" : device,"operation" : operation,"retval" : retval}
def vkGetDeferredOperationResultKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "operation" in indict.keys():
         operation = indict["operation"]
    else: 
         operation = VkDeferredOperationKHR_T()
    print(jvulkanLib.vkGetDeferredOperationResultKHR)
    retval = jvulkanLib.vkGetDeferredOperationResultKHR(device, operation)
    return {"device" : device,"operation" : operation,"retval" : retval}
def vkDeferredOperationJoinKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "operation" in indict.keys():
         operation = indict["operation"]
    else: 
         operation = VkDeferredOperationKHR_T()
    print(jvulkanLib.vkDeferredOperationJoinKHR)
    retval = jvulkanLib.vkDeferredOperationJoinKHR(device, operation)
    return {"device" : device,"operation" : operation,"retval" : retval}
def vkGetPipelineExecutablePropertiesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pPipelineInfo" in indict.keys():
         pPipelineInfo = indict["pPipelineInfo"]
    else: 
         pPipelineInfo = VkPipelineInfoKHR()
    if "pExecutableCount" in indict.keys():
         pExecutableCount = indict["pExecutableCount"]
    else: 
         pExecutableCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkPipelineExecutablePropertiesKHR()
    print(jvulkanLib.vkGetPipelineExecutablePropertiesKHR)
    retval = jvulkanLib.vkGetPipelineExecutablePropertiesKHR(device, pPipelineInfo, pExecutableCount, pProperties)
    return {"device" : device,"pPipelineInfo" : pPipelineInfo,"pExecutableCount" : pExecutableCount,"pProperties" : pProperties,"retval" : retval}
def vkGetPipelineExecutableStatisticsKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pExecutableInfo" in indict.keys():
         pExecutableInfo = indict["pExecutableInfo"]
    else: 
         pExecutableInfo = VkPipelineExecutableInfoKHR()
    if "pStatisticCount" in indict.keys():
         pStatisticCount = indict["pStatisticCount"]
    else: 
         pStatisticCount = pointer(c_uint())
    if "pStatistics" in indict.keys():
         pStatistics = indict["pStatistics"]
    else: 
         pStatistics = VkPipelineExecutableStatisticKHR()
    print(jvulkanLib.vkGetPipelineExecutableStatisticsKHR)
    retval = jvulkanLib.vkGetPipelineExecutableStatisticsKHR(device, pExecutableInfo, pStatisticCount, pStatistics)
    return {"device" : device,"pExecutableInfo" : pExecutableInfo,"pStatisticCount" : pStatisticCount,"pStatistics" : pStatistics,"retval" : retval}
def vkGetPipelineExecutableInternalRepresentationsKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pExecutableInfo" in indict.keys():
         pExecutableInfo = indict["pExecutableInfo"]
    else: 
         pExecutableInfo = VkPipelineExecutableInfoKHR()
    if "pInternalRepresentationCount" in indict.keys():
         pInternalRepresentationCount = indict["pInternalRepresentationCount"]
    else: 
         pInternalRepresentationCount = pointer(c_uint())
    if "pInternalRepresentations" in indict.keys():
         pInternalRepresentations = indict["pInternalRepresentations"]
    else: 
         pInternalRepresentations = VkPipelineExecutableInternalRepresentationKHR()
    print(jvulkanLib.vkGetPipelineExecutableInternalRepresentationsKHR)
    retval = jvulkanLib.vkGetPipelineExecutableInternalRepresentationsKHR(device, pExecutableInfo, pInternalRepresentationCount, pInternalRepresentations)
    return {"device" : device,"pExecutableInfo" : pExecutableInfo,"pInternalRepresentationCount" : pInternalRepresentationCount,"pInternalRepresentations" : pInternalRepresentations,"retval" : retval}
def vkCmdSetEvent2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    if "pDependencyInfo" in indict.keys():
         pDependencyInfo = indict["pDependencyInfo"]
    else: 
         pDependencyInfo = VkDependencyInfo()
    print(jvulkanLib.vkCmdSetEvent2KHR)
    retval = jvulkanLib.vkCmdSetEvent2KHR(commandBuffer, event, pDependencyInfo)
    return {"commandBuffer" : commandBuffer,"event" : event,"pDependencyInfo" : pDependencyInfo,"retval" : retval}
def vkCmdResetEvent2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "event" in indict.keys():
         event = indict["event"]
    else: 
         event = VkEvent_T()
    if "stageMask" in indict.keys():
         stageMask = indict["stageMask"]
    else: 
         stageMask = c_ulong()
    print(jvulkanLib.vkCmdResetEvent2KHR)
    retval = jvulkanLib.vkCmdResetEvent2KHR(commandBuffer, event, stageMask)
    return {"commandBuffer" : commandBuffer,"event" : event,"stageMask" : stageMask,"retval" : retval}
def vkCmdWaitEvents2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "eventCount" in indict.keys():
         eventCount = indict["eventCount"]
    else: 
         eventCount = c_uint()
    if "pEvents" in indict.keys():
         pEvents = indict["pEvents"]
    else: 
         pEvents = pointer(VkEvent_T())
    if "pDependencyInfos" in indict.keys():
         pDependencyInfos = indict["pDependencyInfos"]
    else: 
         pDependencyInfos = VkDependencyInfo()
    print(jvulkanLib.vkCmdWaitEvents2KHR)
    retval = jvulkanLib.vkCmdWaitEvents2KHR(commandBuffer, eventCount, pEvents, pDependencyInfos)
    return {"commandBuffer" : commandBuffer,"eventCount" : eventCount,"pEvents" : pEvents,"pDependencyInfos" : pDependencyInfos,"retval" : retval}
def vkCmdPipelineBarrier2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pDependencyInfo" in indict.keys():
         pDependencyInfo = indict["pDependencyInfo"]
    else: 
         pDependencyInfo = VkDependencyInfo()
    print(jvulkanLib.vkCmdPipelineBarrier2KHR)
    retval = jvulkanLib.vkCmdPipelineBarrier2KHR(commandBuffer, pDependencyInfo)
    return {"commandBuffer" : commandBuffer,"pDependencyInfo" : pDependencyInfo,"retval" : retval}
def vkCmdWriteTimestamp2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "stage" in indict.keys():
         stage = indict["stage"]
    else: 
         stage = c_ulong()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "query" in indict.keys():
         query = indict["query"]
    else: 
         query = c_uint()
    print(jvulkanLib.vkCmdWriteTimestamp2KHR)
    retval = jvulkanLib.vkCmdWriteTimestamp2KHR(commandBuffer, stage, queryPool, query)
    return {"commandBuffer" : commandBuffer,"stage" : stage,"queryPool" : queryPool,"query" : query,"retval" : retval}
def vkQueueSubmit2KHR(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "submitCount" in indict.keys():
         submitCount = indict["submitCount"]
    else: 
         submitCount = c_uint()
    if "pSubmits" in indict.keys():
         pSubmits = indict["pSubmits"]
    else: 
         pSubmits = VkSubmitInfo2()
    if "fence" in indict.keys():
         fence = indict["fence"]
    else: 
         fence = VkFence_T()
    print(jvulkanLib.vkQueueSubmit2KHR)
    retval = jvulkanLib.vkQueueSubmit2KHR(queue, submitCount, pSubmits, fence)
    return {"queue" : queue,"submitCount" : submitCount,"pSubmits" : pSubmits,"fence" : fence,"retval" : retval}
def vkCmdWriteBufferMarker2AMD(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "stage" in indict.keys():
         stage = indict["stage"]
    else: 
         stage = c_ulong()
    if "dstBuffer" in indict.keys():
         dstBuffer = indict["dstBuffer"]
    else: 
         dstBuffer = VkBuffer_T()
    if "dstOffset" in indict.keys():
         dstOffset = indict["dstOffset"]
    else: 
         dstOffset = c_ulong()
    if "marker" in indict.keys():
         marker = indict["marker"]
    else: 
         marker = c_uint()
    print(jvulkanLib.vkCmdWriteBufferMarker2AMD)
    retval = jvulkanLib.vkCmdWriteBufferMarker2AMD(commandBuffer, stage, dstBuffer, dstOffset, marker)
    return {"commandBuffer" : commandBuffer,"stage" : stage,"dstBuffer" : dstBuffer,"dstOffset" : dstOffset,"marker" : marker,"retval" : retval}
def vkGetQueueCheckpointData2NV(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "pCheckpointDataCount" in indict.keys():
         pCheckpointDataCount = indict["pCheckpointDataCount"]
    else: 
         pCheckpointDataCount = pointer(c_uint())
    if "pCheckpointData" in indict.keys():
         pCheckpointData = indict["pCheckpointData"]
    else: 
         pCheckpointData = VkCheckpointData2NV()
    print(jvulkanLib.vkGetQueueCheckpointData2NV)
    retval = jvulkanLib.vkGetQueueCheckpointData2NV(queue, pCheckpointDataCount, pCheckpointData)
    return {"queue" : queue,"pCheckpointDataCount" : pCheckpointDataCount,"pCheckpointData" : pCheckpointData,"retval" : retval}
def vkCmdCopyBuffer2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pCopyBufferInfo" in indict.keys():
         pCopyBufferInfo = indict["pCopyBufferInfo"]
    else: 
         pCopyBufferInfo = VkCopyBufferInfo2()
    print(jvulkanLib.vkCmdCopyBuffer2KHR)
    retval = jvulkanLib.vkCmdCopyBuffer2KHR(commandBuffer, pCopyBufferInfo)
    return {"commandBuffer" : commandBuffer,"pCopyBufferInfo" : pCopyBufferInfo,"retval" : retval}
def vkCmdCopyImage2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pCopyImageInfo" in indict.keys():
         pCopyImageInfo = indict["pCopyImageInfo"]
    else: 
         pCopyImageInfo = VkCopyImageInfo2()
    print(jvulkanLib.vkCmdCopyImage2KHR)
    retval = jvulkanLib.vkCmdCopyImage2KHR(commandBuffer, pCopyImageInfo)
    return {"commandBuffer" : commandBuffer,"pCopyImageInfo" : pCopyImageInfo,"retval" : retval}
def vkCmdCopyBufferToImage2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pCopyBufferToImageInfo" in indict.keys():
         pCopyBufferToImageInfo = indict["pCopyBufferToImageInfo"]
    else: 
         pCopyBufferToImageInfo = VkCopyBufferToImageInfo2()
    print(jvulkanLib.vkCmdCopyBufferToImage2KHR)
    retval = jvulkanLib.vkCmdCopyBufferToImage2KHR(commandBuffer, pCopyBufferToImageInfo)
    return {"commandBuffer" : commandBuffer,"pCopyBufferToImageInfo" : pCopyBufferToImageInfo,"retval" : retval}
def vkCmdCopyImageToBuffer2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pCopyImageToBufferInfo" in indict.keys():
         pCopyImageToBufferInfo = indict["pCopyImageToBufferInfo"]
    else: 
         pCopyImageToBufferInfo = VkCopyImageToBufferInfo2()
    print(jvulkanLib.vkCmdCopyImageToBuffer2KHR)
    retval = jvulkanLib.vkCmdCopyImageToBuffer2KHR(commandBuffer, pCopyImageToBufferInfo)
    return {"commandBuffer" : commandBuffer,"pCopyImageToBufferInfo" : pCopyImageToBufferInfo,"retval" : retval}
def vkCmdBlitImage2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pBlitImageInfo" in indict.keys():
         pBlitImageInfo = indict["pBlitImageInfo"]
    else: 
         pBlitImageInfo = VkBlitImageInfo2()
    print(jvulkanLib.vkCmdBlitImage2KHR)
    retval = jvulkanLib.vkCmdBlitImage2KHR(commandBuffer, pBlitImageInfo)
    return {"commandBuffer" : commandBuffer,"pBlitImageInfo" : pBlitImageInfo,"retval" : retval}
def vkCmdResolveImage2KHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pResolveImageInfo" in indict.keys():
         pResolveImageInfo = indict["pResolveImageInfo"]
    else: 
         pResolveImageInfo = VkResolveImageInfo2()
    print(jvulkanLib.vkCmdResolveImage2KHR)
    retval = jvulkanLib.vkCmdResolveImage2KHR(commandBuffer, pResolveImageInfo)
    return {"commandBuffer" : commandBuffer,"pResolveImageInfo" : pResolveImageInfo,"retval" : retval}
def vkGetDeviceBufferMemoryRequirementsKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkDeviceBufferMemoryRequirements()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetDeviceBufferMemoryRequirementsKHR)
    retval = jvulkanLib.vkGetDeviceBufferMemoryRequirementsKHR(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetDeviceImageMemoryRequirementsKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkDeviceImageMemoryRequirements()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetDeviceImageMemoryRequirementsKHR)
    retval = jvulkanLib.vkGetDeviceImageMemoryRequirementsKHR(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkGetDeviceImageSparseMemoryRequirementsKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkDeviceImageMemoryRequirements()
    if "pSparseMemoryRequirementCount" in indict.keys():
         pSparseMemoryRequirementCount = indict["pSparseMemoryRequirementCount"]
    else: 
         pSparseMemoryRequirementCount = pointer(c_uint())
    if "pSparseMemoryRequirements" in indict.keys():
         pSparseMemoryRequirements = indict["pSparseMemoryRequirements"]
    else: 
         pSparseMemoryRequirements = VkSparseImageMemoryRequirements2()
    print(jvulkanLib.vkGetDeviceImageSparseMemoryRequirementsKHR)
    retval = jvulkanLib.vkGetDeviceImageSparseMemoryRequirementsKHR(device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pSparseMemoryRequirementCount" : pSparseMemoryRequirementCount,"pSparseMemoryRequirements" : pSparseMemoryRequirements,"retval" : retval}
def vkCreateDebugReportCallbackEXT(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDebugReportCallbackCreateInfoEXT()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pCallback" in indict.keys():
         pCallback = indict["pCallback"]
    else: 
         pCallback = pointer(VkDebugReportCallbackEXT_T())
    print(jvulkanLib.vkCreateDebugReportCallbackEXT)
    retval = jvulkanLib.vkCreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator, pCallback)
    return {"instance" : instance,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pCallback" : pCallback,"retval" : retval}
def vkDestroyDebugReportCallbackEXT(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "callback" in indict.keys():
         callback = indict["callback"]
    else: 
         callback = VkDebugReportCallbackEXT_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyDebugReportCallbackEXT)
    retval = jvulkanLib.vkDestroyDebugReportCallbackEXT(instance, callback, pAllocator)
    return {"instance" : instance,"callback" : callback,"pAllocator" : pAllocator,"retval" : retval}
def vkDebugReportMessageEXT(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    if "objectType" in indict.keys():
         objectType = indict["objectType"]
    else: 
         objectType = c_int()
    if "object" in indict.keys():
         object = indict["object"]
    else: 
         object = c_ulong()
    if "location" in indict.keys():
         location = indict["location"]
    else: 
         location = c_ulong()
    if "messageCode" in indict.keys():
         messageCode = indict["messageCode"]
    else: 
         messageCode = c_int()
    if "pLayerPrefix" in indict.keys():
         pLayerPrefix = indict["pLayerPrefix"]
    else: 
         pLayerPrefix = c_char_p()
    if "pMessage" in indict.keys():
         pMessage = indict["pMessage"]
    else: 
         pMessage = c_char_p()
    print(jvulkanLib.vkDebugReportMessageEXT)
    retval = jvulkanLib.vkDebugReportMessageEXT(instance, flags, objectType, object, location, messageCode, pLayerPrefix, pMessage)
    return {"instance" : instance,"flags" : flags,"objectType" : objectType,"object" : object,"location" : location,"messageCode" : messageCode,"pLayerPrefix" : pLayerPrefix,"pMessage" : pMessage,"retval" : retval}
def vkDebugMarkerSetObjectTagEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pTagInfo" in indict.keys():
         pTagInfo = indict["pTagInfo"]
    else: 
         pTagInfo = VkDebugMarkerObjectTagInfoEXT()
    print(jvulkanLib.vkDebugMarkerSetObjectTagEXT)
    retval = jvulkanLib.vkDebugMarkerSetObjectTagEXT(device, pTagInfo)
    return {"device" : device,"pTagInfo" : pTagInfo,"retval" : retval}
def vkDebugMarkerSetObjectNameEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pNameInfo" in indict.keys():
         pNameInfo = indict["pNameInfo"]
    else: 
         pNameInfo = VkDebugMarkerObjectNameInfoEXT()
    print(jvulkanLib.vkDebugMarkerSetObjectNameEXT)
    retval = jvulkanLib.vkDebugMarkerSetObjectNameEXT(device, pNameInfo)
    return {"device" : device,"pNameInfo" : pNameInfo,"retval" : retval}
def vkCmdDebugMarkerBeginEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pMarkerInfo" in indict.keys():
         pMarkerInfo = indict["pMarkerInfo"]
    else: 
         pMarkerInfo = VkDebugMarkerMarkerInfoEXT()
    print(jvulkanLib.vkCmdDebugMarkerBeginEXT)
    retval = jvulkanLib.vkCmdDebugMarkerBeginEXT(commandBuffer, pMarkerInfo)
    return {"commandBuffer" : commandBuffer,"pMarkerInfo" : pMarkerInfo,"retval" : retval}
def vkCmdDebugMarkerEndEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    print(jvulkanLib.vkCmdDebugMarkerEndEXT)
    retval = jvulkanLib.vkCmdDebugMarkerEndEXT(commandBuffer)
    return {"commandBuffer" : commandBuffer,"retval" : retval}
def vkCmdDebugMarkerInsertEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pMarkerInfo" in indict.keys():
         pMarkerInfo = indict["pMarkerInfo"]
    else: 
         pMarkerInfo = VkDebugMarkerMarkerInfoEXT()
    print(jvulkanLib.vkCmdDebugMarkerInsertEXT)
    retval = jvulkanLib.vkCmdDebugMarkerInsertEXT(commandBuffer, pMarkerInfo)
    return {"commandBuffer" : commandBuffer,"pMarkerInfo" : pMarkerInfo,"retval" : retval}
def vkCmdBindTransformFeedbackBuffersEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstBinding" in indict.keys():
         firstBinding = indict["firstBinding"]
    else: 
         firstBinding = c_uint()
    if "bindingCount" in indict.keys():
         bindingCount = indict["bindingCount"]
    else: 
         bindingCount = c_uint()
    if "pBuffers" in indict.keys():
         pBuffers = indict["pBuffers"]
    else: 
         pBuffers = pointer(VkBuffer_T())
    if "pOffsets" in indict.keys():
         pOffsets = indict["pOffsets"]
    else: 
         pOffsets = pointer(c_ulong())
    if "pSizes" in indict.keys():
         pSizes = indict["pSizes"]
    else: 
         pSizes = pointer(c_ulong())
    print(jvulkanLib.vkCmdBindTransformFeedbackBuffersEXT)
    retval = jvulkanLib.vkCmdBindTransformFeedbackBuffersEXT(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes)
    return {"commandBuffer" : commandBuffer,"firstBinding" : firstBinding,"bindingCount" : bindingCount,"pBuffers" : pBuffers,"pOffsets" : pOffsets,"pSizes" : pSizes,"retval" : retval}
def vkCmdBeginTransformFeedbackEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstCounterBuffer" in indict.keys():
         firstCounterBuffer = indict["firstCounterBuffer"]
    else: 
         firstCounterBuffer = c_uint()
    if "counterBufferCount" in indict.keys():
         counterBufferCount = indict["counterBufferCount"]
    else: 
         counterBufferCount = c_uint()
    if "pCounterBuffers" in indict.keys():
         pCounterBuffers = indict["pCounterBuffers"]
    else: 
         pCounterBuffers = pointer(VkBuffer_T())
    if "pCounterBufferOffsets" in indict.keys():
         pCounterBufferOffsets = indict["pCounterBufferOffsets"]
    else: 
         pCounterBufferOffsets = pointer(c_ulong())
    print(jvulkanLib.vkCmdBeginTransformFeedbackEXT)
    retval = jvulkanLib.vkCmdBeginTransformFeedbackEXT(commandBuffer, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets)
    return {"commandBuffer" : commandBuffer,"firstCounterBuffer" : firstCounterBuffer,"counterBufferCount" : counterBufferCount,"pCounterBuffers" : pCounterBuffers,"pCounterBufferOffsets" : pCounterBufferOffsets,"retval" : retval}
def vkCmdEndTransformFeedbackEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstCounterBuffer" in indict.keys():
         firstCounterBuffer = indict["firstCounterBuffer"]
    else: 
         firstCounterBuffer = c_uint()
    if "counterBufferCount" in indict.keys():
         counterBufferCount = indict["counterBufferCount"]
    else: 
         counterBufferCount = c_uint()
    if "pCounterBuffers" in indict.keys():
         pCounterBuffers = indict["pCounterBuffers"]
    else: 
         pCounterBuffers = pointer(VkBuffer_T())
    if "pCounterBufferOffsets" in indict.keys():
         pCounterBufferOffsets = indict["pCounterBufferOffsets"]
    else: 
         pCounterBufferOffsets = pointer(c_ulong())
    print(jvulkanLib.vkCmdEndTransformFeedbackEXT)
    retval = jvulkanLib.vkCmdEndTransformFeedbackEXT(commandBuffer, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets)
    return {"commandBuffer" : commandBuffer,"firstCounterBuffer" : firstCounterBuffer,"counterBufferCount" : counterBufferCount,"pCounterBuffers" : pCounterBuffers,"pCounterBufferOffsets" : pCounterBufferOffsets,"retval" : retval}
def vkCmdBeginQueryIndexedEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "query" in indict.keys():
         query = indict["query"]
    else: 
         query = c_uint()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    if "index" in indict.keys():
         index = indict["index"]
    else: 
         index = c_uint()
    print(jvulkanLib.vkCmdBeginQueryIndexedEXT)
    retval = jvulkanLib.vkCmdBeginQueryIndexedEXT(commandBuffer, queryPool, query, flags, index)
    return {"commandBuffer" : commandBuffer,"queryPool" : queryPool,"query" : query,"flags" : flags,"index" : index,"retval" : retval}
def vkCmdEndQueryIndexedEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "query" in indict.keys():
         query = indict["query"]
    else: 
         query = c_uint()
    if "index" in indict.keys():
         index = indict["index"]
    else: 
         index = c_uint()
    print(jvulkanLib.vkCmdEndQueryIndexedEXT)
    retval = jvulkanLib.vkCmdEndQueryIndexedEXT(commandBuffer, queryPool, query, index)
    return {"commandBuffer" : commandBuffer,"queryPool" : queryPool,"query" : query,"index" : index,"retval" : retval}
def vkCmdDrawIndirectByteCountEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "instanceCount" in indict.keys():
         instanceCount = indict["instanceCount"]
    else: 
         instanceCount = c_uint()
    if "firstInstance" in indict.keys():
         firstInstance = indict["firstInstance"]
    else: 
         firstInstance = c_uint()
    if "counterBuffer" in indict.keys():
         counterBuffer = indict["counterBuffer"]
    else: 
         counterBuffer = VkBuffer_T()
    if "counterBufferOffset" in indict.keys():
         counterBufferOffset = indict["counterBufferOffset"]
    else: 
         counterBufferOffset = c_ulong()
    if "counterOffset" in indict.keys():
         counterOffset = indict["counterOffset"]
    else: 
         counterOffset = c_uint()
    if "vertexStride" in indict.keys():
         vertexStride = indict["vertexStride"]
    else: 
         vertexStride = c_uint()
    print(jvulkanLib.vkCmdDrawIndirectByteCountEXT)
    retval = jvulkanLib.vkCmdDrawIndirectByteCountEXT(commandBuffer, instanceCount, firstInstance, counterBuffer, counterBufferOffset, counterOffset, vertexStride)
    return {"commandBuffer" : commandBuffer,"instanceCount" : instanceCount,"firstInstance" : firstInstance,"counterBuffer" : counterBuffer,"counterBufferOffset" : counterBufferOffset,"counterOffset" : counterOffset,"vertexStride" : vertexStride,"retval" : retval}
def vkCreateCuModuleNVX(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkCuModuleCreateInfoNVX()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pModule" in indict.keys():
         pModule = indict["pModule"]
    else: 
         pModule = pointer(VkCuModuleNVX_T())
    print(jvulkanLib.vkCreateCuModuleNVX)
    retval = jvulkanLib.vkCreateCuModuleNVX(device, pCreateInfo, pAllocator, pModule)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pModule" : pModule,"retval" : retval}
def vkCreateCuFunctionNVX(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkCuFunctionCreateInfoNVX()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pFunction" in indict.keys():
         pFunction = indict["pFunction"]
    else: 
         pFunction = pointer(VkCuFunctionNVX_T())
    print(jvulkanLib.vkCreateCuFunctionNVX)
    retval = jvulkanLib.vkCreateCuFunctionNVX(device, pCreateInfo, pAllocator, pFunction)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pFunction" : pFunction,"retval" : retval}
def vkDestroyCuModuleNVX(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "module" in indict.keys():
         module = indict["module"]
    else: 
         module = VkCuModuleNVX_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyCuModuleNVX)
    retval = jvulkanLib.vkDestroyCuModuleNVX(device, module, pAllocator)
    return {"device" : device,"module" : module,"pAllocator" : pAllocator,"retval" : retval}
def vkDestroyCuFunctionNVX(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "function" in indict.keys():
         function = indict["function"]
    else: 
         function = VkCuFunctionNVX_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyCuFunctionNVX)
    retval = jvulkanLib.vkDestroyCuFunctionNVX(device, function, pAllocator)
    return {"device" : device,"function" : function,"pAllocator" : pAllocator,"retval" : retval}
def vkCmdCuLaunchKernelNVX(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pLaunchInfo" in indict.keys():
         pLaunchInfo = indict["pLaunchInfo"]
    else: 
         pLaunchInfo = VkCuLaunchInfoNVX()
    print(jvulkanLib.vkCmdCuLaunchKernelNVX)
    retval = jvulkanLib.vkCmdCuLaunchKernelNVX(commandBuffer, pLaunchInfo)
    return {"commandBuffer" : commandBuffer,"pLaunchInfo" : pLaunchInfo,"retval" : retval}
def vkGetImageViewHandleNVX(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkImageViewHandleInfoNVX()
    print(jvulkanLib.vkGetImageViewHandleNVX)
    retval = jvulkanLib.vkGetImageViewHandleNVX(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkGetImageViewAddressNVX(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "imageView" in indict.keys():
         imageView = indict["imageView"]
    else: 
         imageView = VkImageView_T()
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkImageViewAddressPropertiesNVX()
    print(jvulkanLib.vkGetImageViewAddressNVX)
    retval = jvulkanLib.vkGetImageViewAddressNVX(device, imageView, pProperties)
    return {"device" : device,"imageView" : imageView,"pProperties" : pProperties,"retval" : retval}
def vkCmdDrawIndirectCountAMD(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "countBuffer" in indict.keys():
         countBuffer = indict["countBuffer"]
    else: 
         countBuffer = VkBuffer_T()
    if "countBufferOffset" in indict.keys():
         countBufferOffset = indict["countBufferOffset"]
    else: 
         countBufferOffset = c_ulong()
    if "maxDrawCount" in indict.keys():
         maxDrawCount = indict["maxDrawCount"]
    else: 
         maxDrawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawIndirectCountAMD)
    retval = jvulkanLib.vkCmdDrawIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"countBuffer" : countBuffer,"countBufferOffset" : countBufferOffset,"maxDrawCount" : maxDrawCount,"stride" : stride,"retval" : retval}
def vkCmdDrawIndexedIndirectCountAMD(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "countBuffer" in indict.keys():
         countBuffer = indict["countBuffer"]
    else: 
         countBuffer = VkBuffer_T()
    if "countBufferOffset" in indict.keys():
         countBufferOffset = indict["countBufferOffset"]
    else: 
         countBufferOffset = c_ulong()
    if "maxDrawCount" in indict.keys():
         maxDrawCount = indict["maxDrawCount"]
    else: 
         maxDrawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawIndexedIndirectCountAMD)
    retval = jvulkanLib.vkCmdDrawIndexedIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"countBuffer" : countBuffer,"countBufferOffset" : countBufferOffset,"maxDrawCount" : maxDrawCount,"stride" : stride,"retval" : retval}
def vkGetShaderInfoAMD(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipeline" in indict.keys():
         pipeline = indict["pipeline"]
    else: 
         pipeline = VkPipeline_T()
    if "shaderStage" in indict.keys():
         shaderStage = indict["shaderStage"]
    else: 
         shaderStage = c_int()
    if "infoType" in indict.keys():
         infoType = indict["infoType"]
    else: 
         infoType = c_int()
    if "pInfoSize" in indict.keys():
         pInfoSize = indict["pInfoSize"]
    else: 
         pInfoSize = pointer(c_ulong())
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = c_void_p()
    print(jvulkanLib.vkGetShaderInfoAMD)
    retval = jvulkanLib.vkGetShaderInfoAMD(device, pipeline, shaderStage, infoType, pInfoSize, pInfo)
    return {"device" : device,"pipeline" : pipeline,"shaderStage" : shaderStage,"infoType" : infoType,"pInfoSize" : pInfoSize,"pInfo" : pInfo,"retval" : retval}
def vkGetPhysicalDeviceExternalImageFormatPropertiesNV(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "format" in indict.keys():
         format = indict["format"]
    else: 
         format = c_int()
    if "type" in indict.keys():
         type = indict["type"]
    else: 
         type = c_int()
    if "tiling" in indict.keys():
         tiling = indict["tiling"]
    else: 
         tiling = c_int()
    if "usage" in indict.keys():
         usage = indict["usage"]
    else: 
         usage = c_uint()
    if "flags" in indict.keys():
         flags = indict["flags"]
    else: 
         flags = c_uint()
    if "externalHandleType" in indict.keys():
         externalHandleType = indict["externalHandleType"]
    else: 
         externalHandleType = c_uint()
    if "pExternalImageFormatProperties" in indict.keys():
         pExternalImageFormatProperties = indict["pExternalImageFormatProperties"]
    else: 
         pExternalImageFormatProperties = VkExternalImageFormatPropertiesNV()
    print(jvulkanLib.vkGetPhysicalDeviceExternalImageFormatPropertiesNV)
    retval = jvulkanLib.vkGetPhysicalDeviceExternalImageFormatPropertiesNV(physicalDevice, format, type, tiling, usage, flags, externalHandleType, pExternalImageFormatProperties)
    return {"physicalDevice" : physicalDevice,"format" : format,"type" : type,"tiling" : tiling,"usage" : usage,"flags" : flags,"externalHandleType" : externalHandleType,"pExternalImageFormatProperties" : pExternalImageFormatProperties,"retval" : retval}
def vkCmdBeginConditionalRenderingEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pConditionalRenderingBegin" in indict.keys():
         pConditionalRenderingBegin = indict["pConditionalRenderingBegin"]
    else: 
         pConditionalRenderingBegin = VkConditionalRenderingBeginInfoEXT()
    print(jvulkanLib.vkCmdBeginConditionalRenderingEXT)
    retval = jvulkanLib.vkCmdBeginConditionalRenderingEXT(commandBuffer, pConditionalRenderingBegin)
    return {"commandBuffer" : commandBuffer,"pConditionalRenderingBegin" : pConditionalRenderingBegin,"retval" : retval}
def vkCmdEndConditionalRenderingEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    print(jvulkanLib.vkCmdEndConditionalRenderingEXT)
    retval = jvulkanLib.vkCmdEndConditionalRenderingEXT(commandBuffer)
    return {"commandBuffer" : commandBuffer,"retval" : retval}
def vkCmdSetViewportWScalingNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstViewport" in indict.keys():
         firstViewport = indict["firstViewport"]
    else: 
         firstViewport = c_uint()
    if "viewportCount" in indict.keys():
         viewportCount = indict["viewportCount"]
    else: 
         viewportCount = c_uint()
    if "pViewportWScalings" in indict.keys():
         pViewportWScalings = indict["pViewportWScalings"]
    else: 
         pViewportWScalings = VkViewportWScalingNV()
    print(jvulkanLib.vkCmdSetViewportWScalingNV)
    retval = jvulkanLib.vkCmdSetViewportWScalingNV(commandBuffer, firstViewport, viewportCount, pViewportWScalings)
    return {"commandBuffer" : commandBuffer,"firstViewport" : firstViewport,"viewportCount" : viewportCount,"pViewportWScalings" : pViewportWScalings,"retval" : retval}
def vkReleaseDisplayEXT(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "display" in indict.keys():
         display = indict["display"]
    else: 
         display = VkDisplayKHR_T()
    print(jvulkanLib.vkReleaseDisplayEXT)
    retval = jvulkanLib.vkReleaseDisplayEXT(physicalDevice, display)
    return {"physicalDevice" : physicalDevice,"display" : display,"retval" : retval}
def vkGetPhysicalDeviceSurfaceCapabilities2EXT(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "surface" in indict.keys():
         surface = indict["surface"]
    else: 
         surface = VkSurfaceKHR_T()
    if "pSurfaceCapabilities" in indict.keys():
         pSurfaceCapabilities = indict["pSurfaceCapabilities"]
    else: 
         pSurfaceCapabilities = VkSurfaceCapabilities2EXT()
    print(jvulkanLib.vkGetPhysicalDeviceSurfaceCapabilities2EXT)
    retval = jvulkanLib.vkGetPhysicalDeviceSurfaceCapabilities2EXT(physicalDevice, surface, pSurfaceCapabilities)
    return {"physicalDevice" : physicalDevice,"surface" : surface,"pSurfaceCapabilities" : pSurfaceCapabilities,"retval" : retval}
def vkDisplayPowerControlEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "display" in indict.keys():
         display = indict["display"]
    else: 
         display = VkDisplayKHR_T()
    if "pDisplayPowerInfo" in indict.keys():
         pDisplayPowerInfo = indict["pDisplayPowerInfo"]
    else: 
         pDisplayPowerInfo = VkDisplayPowerInfoEXT()
    print(jvulkanLib.vkDisplayPowerControlEXT)
    retval = jvulkanLib.vkDisplayPowerControlEXT(device, display, pDisplayPowerInfo)
    return {"device" : device,"display" : display,"pDisplayPowerInfo" : pDisplayPowerInfo,"retval" : retval}
def vkRegisterDeviceEventEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pDeviceEventInfo" in indict.keys():
         pDeviceEventInfo = indict["pDeviceEventInfo"]
    else: 
         pDeviceEventInfo = VkDeviceEventInfoEXT()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pFence" in indict.keys():
         pFence = indict["pFence"]
    else: 
         pFence = pointer(VkFence_T())
    print(jvulkanLib.vkRegisterDeviceEventEXT)
    retval = jvulkanLib.vkRegisterDeviceEventEXT(device, pDeviceEventInfo, pAllocator, pFence)
    return {"device" : device,"pDeviceEventInfo" : pDeviceEventInfo,"pAllocator" : pAllocator,"pFence" : pFence,"retval" : retval}
def vkRegisterDisplayEventEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "display" in indict.keys():
         display = indict["display"]
    else: 
         display = VkDisplayKHR_T()
    if "pDisplayEventInfo" in indict.keys():
         pDisplayEventInfo = indict["pDisplayEventInfo"]
    else: 
         pDisplayEventInfo = VkDisplayEventInfoEXT()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pFence" in indict.keys():
         pFence = indict["pFence"]
    else: 
         pFence = pointer(VkFence_T())
    print(jvulkanLib.vkRegisterDisplayEventEXT)
    retval = jvulkanLib.vkRegisterDisplayEventEXT(device, display, pDisplayEventInfo, pAllocator, pFence)
    return {"device" : device,"display" : display,"pDisplayEventInfo" : pDisplayEventInfo,"pAllocator" : pAllocator,"pFence" : pFence,"retval" : retval}
def vkGetSwapchainCounterEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchain" in indict.keys():
         swapchain = indict["swapchain"]
    else: 
         swapchain = VkSwapchainKHR_T()
    if "counter" in indict.keys():
         counter = indict["counter"]
    else: 
         counter = c_int()
    if "pCounterValue" in indict.keys():
         pCounterValue = indict["pCounterValue"]
    else: 
         pCounterValue = pointer(c_ulong())
    print(jvulkanLib.vkGetSwapchainCounterEXT)
    retval = jvulkanLib.vkGetSwapchainCounterEXT(device, swapchain, counter, pCounterValue)
    return {"device" : device,"swapchain" : swapchain,"counter" : counter,"pCounterValue" : pCounterValue,"retval" : retval}
def vkGetRefreshCycleDurationGOOGLE(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchain" in indict.keys():
         swapchain = indict["swapchain"]
    else: 
         swapchain = VkSwapchainKHR_T()
    if "pDisplayTimingProperties" in indict.keys():
         pDisplayTimingProperties = indict["pDisplayTimingProperties"]
    else: 
         pDisplayTimingProperties = VkRefreshCycleDurationGOOGLE()
    print(jvulkanLib.vkGetRefreshCycleDurationGOOGLE)
    retval = jvulkanLib.vkGetRefreshCycleDurationGOOGLE(device, swapchain, pDisplayTimingProperties)
    return {"device" : device,"swapchain" : swapchain,"pDisplayTimingProperties" : pDisplayTimingProperties,"retval" : retval}
def vkGetPastPresentationTimingGOOGLE(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchain" in indict.keys():
         swapchain = indict["swapchain"]
    else: 
         swapchain = VkSwapchainKHR_T()
    if "pPresentationTimingCount" in indict.keys():
         pPresentationTimingCount = indict["pPresentationTimingCount"]
    else: 
         pPresentationTimingCount = pointer(c_uint())
    if "pPresentationTimings" in indict.keys():
         pPresentationTimings = indict["pPresentationTimings"]
    else: 
         pPresentationTimings = VkPastPresentationTimingGOOGLE()
    print(jvulkanLib.vkGetPastPresentationTimingGOOGLE)
    retval = jvulkanLib.vkGetPastPresentationTimingGOOGLE(device, swapchain, pPresentationTimingCount, pPresentationTimings)
    return {"device" : device,"swapchain" : swapchain,"pPresentationTimingCount" : pPresentationTimingCount,"pPresentationTimings" : pPresentationTimings,"retval" : retval}
def vkCmdSetDiscardRectangleEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstDiscardRectangle" in indict.keys():
         firstDiscardRectangle = indict["firstDiscardRectangle"]
    else: 
         firstDiscardRectangle = c_uint()
    if "discardRectangleCount" in indict.keys():
         discardRectangleCount = indict["discardRectangleCount"]
    else: 
         discardRectangleCount = c_uint()
    if "pDiscardRectangles" in indict.keys():
         pDiscardRectangles = indict["pDiscardRectangles"]
    else: 
         pDiscardRectangles = VkRect2D()
    print(jvulkanLib.vkCmdSetDiscardRectangleEXT)
    retval = jvulkanLib.vkCmdSetDiscardRectangleEXT(commandBuffer, firstDiscardRectangle, discardRectangleCount, pDiscardRectangles)
    return {"commandBuffer" : commandBuffer,"firstDiscardRectangle" : firstDiscardRectangle,"discardRectangleCount" : discardRectangleCount,"pDiscardRectangles" : pDiscardRectangles,"retval" : retval}
def vkSetHdrMetadataEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapchainCount" in indict.keys():
         swapchainCount = indict["swapchainCount"]
    else: 
         swapchainCount = c_uint()
    if "pSwapchains" in indict.keys():
         pSwapchains = indict["pSwapchains"]
    else: 
         pSwapchains = pointer(VkSwapchainKHR_T())
    if "pMetadata" in indict.keys():
         pMetadata = indict["pMetadata"]
    else: 
         pMetadata = VkHdrMetadataEXT()
    print(jvulkanLib.vkSetHdrMetadataEXT)
    retval = jvulkanLib.vkSetHdrMetadataEXT(device, swapchainCount, pSwapchains, pMetadata)
    return {"device" : device,"swapchainCount" : swapchainCount,"pSwapchains" : pSwapchains,"pMetadata" : pMetadata,"retval" : retval}
def vkSetDebugUtilsObjectNameEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pNameInfo" in indict.keys():
         pNameInfo = indict["pNameInfo"]
    else: 
         pNameInfo = VkDebugUtilsObjectNameInfoEXT()
    print(jvulkanLib.vkSetDebugUtilsObjectNameEXT)
    retval = jvulkanLib.vkSetDebugUtilsObjectNameEXT(device, pNameInfo)
    return {"device" : device,"pNameInfo" : pNameInfo,"retval" : retval}
def vkSetDebugUtilsObjectTagEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pTagInfo" in indict.keys():
         pTagInfo = indict["pTagInfo"]
    else: 
         pTagInfo = VkDebugUtilsObjectTagInfoEXT()
    print(jvulkanLib.vkSetDebugUtilsObjectTagEXT)
    retval = jvulkanLib.vkSetDebugUtilsObjectTagEXT(device, pTagInfo)
    return {"device" : device,"pTagInfo" : pTagInfo,"retval" : retval}
def vkQueueBeginDebugUtilsLabelEXT(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "pLabelInfo" in indict.keys():
         pLabelInfo = indict["pLabelInfo"]
    else: 
         pLabelInfo = VkDebugUtilsLabelEXT()
    print(jvulkanLib.vkQueueBeginDebugUtilsLabelEXT)
    retval = jvulkanLib.vkQueueBeginDebugUtilsLabelEXT(queue, pLabelInfo)
    return {"queue" : queue,"pLabelInfo" : pLabelInfo,"retval" : retval}
def vkQueueEndDebugUtilsLabelEXT(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    print(jvulkanLib.vkQueueEndDebugUtilsLabelEXT)
    retval = jvulkanLib.vkQueueEndDebugUtilsLabelEXT(queue)
    return {"queue" : queue,"retval" : retval}
def vkQueueInsertDebugUtilsLabelEXT(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "pLabelInfo" in indict.keys():
         pLabelInfo = indict["pLabelInfo"]
    else: 
         pLabelInfo = VkDebugUtilsLabelEXT()
    print(jvulkanLib.vkQueueInsertDebugUtilsLabelEXT)
    retval = jvulkanLib.vkQueueInsertDebugUtilsLabelEXT(queue, pLabelInfo)
    return {"queue" : queue,"pLabelInfo" : pLabelInfo,"retval" : retval}
def vkCmdBeginDebugUtilsLabelEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pLabelInfo" in indict.keys():
         pLabelInfo = indict["pLabelInfo"]
    else: 
         pLabelInfo = VkDebugUtilsLabelEXT()
    print(jvulkanLib.vkCmdBeginDebugUtilsLabelEXT)
    retval = jvulkanLib.vkCmdBeginDebugUtilsLabelEXT(commandBuffer, pLabelInfo)
    return {"commandBuffer" : commandBuffer,"pLabelInfo" : pLabelInfo,"retval" : retval}
def vkCmdEndDebugUtilsLabelEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    print(jvulkanLib.vkCmdEndDebugUtilsLabelEXT)
    retval = jvulkanLib.vkCmdEndDebugUtilsLabelEXT(commandBuffer)
    return {"commandBuffer" : commandBuffer,"retval" : retval}
def vkCmdInsertDebugUtilsLabelEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pLabelInfo" in indict.keys():
         pLabelInfo = indict["pLabelInfo"]
    else: 
         pLabelInfo = VkDebugUtilsLabelEXT()
    print(jvulkanLib.vkCmdInsertDebugUtilsLabelEXT)
    retval = jvulkanLib.vkCmdInsertDebugUtilsLabelEXT(commandBuffer, pLabelInfo)
    return {"commandBuffer" : commandBuffer,"pLabelInfo" : pLabelInfo,"retval" : retval}
def vkCreateDebugUtilsMessengerEXT(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkDebugUtilsMessengerCreateInfoEXT()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pMessenger" in indict.keys():
         pMessenger = indict["pMessenger"]
    else: 
         pMessenger = pointer(VkDebugUtilsMessengerEXT_T())
    print(jvulkanLib.vkCreateDebugUtilsMessengerEXT)
    retval = jvulkanLib.vkCreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator, pMessenger)
    return {"instance" : instance,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pMessenger" : pMessenger,"retval" : retval}
def vkDestroyDebugUtilsMessengerEXT(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "messenger" in indict.keys():
         messenger = indict["messenger"]
    else: 
         messenger = VkDebugUtilsMessengerEXT_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyDebugUtilsMessengerEXT)
    retval = jvulkanLib.vkDestroyDebugUtilsMessengerEXT(instance, messenger, pAllocator)
    return {"instance" : instance,"messenger" : messenger,"pAllocator" : pAllocator,"retval" : retval}
def vkSubmitDebugUtilsMessageEXT(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "messageSeverity" in indict.keys():
         messageSeverity = indict["messageSeverity"]
    else: 
         messageSeverity = c_int()
    if "messageTypes" in indict.keys():
         messageTypes = indict["messageTypes"]
    else: 
         messageTypes = c_uint()
    if "pCallbackData" in indict.keys():
         pCallbackData = indict["pCallbackData"]
    else: 
         pCallbackData = VkDebugUtilsMessengerCallbackDataEXT()
    print(jvulkanLib.vkSubmitDebugUtilsMessageEXT)
    retval = jvulkanLib.vkSubmitDebugUtilsMessageEXT(instance, messageSeverity, messageTypes, pCallbackData)
    return {"instance" : instance,"messageSeverity" : messageSeverity,"messageTypes" : messageTypes,"pCallbackData" : pCallbackData,"retval" : retval}
def vkCmdSetSampleLocationsEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pSampleLocationsInfo" in indict.keys():
         pSampleLocationsInfo = indict["pSampleLocationsInfo"]
    else: 
         pSampleLocationsInfo = VkSampleLocationsInfoEXT()
    print(jvulkanLib.vkCmdSetSampleLocationsEXT)
    retval = jvulkanLib.vkCmdSetSampleLocationsEXT(commandBuffer, pSampleLocationsInfo)
    return {"commandBuffer" : commandBuffer,"pSampleLocationsInfo" : pSampleLocationsInfo,"retval" : retval}
def vkGetPhysicalDeviceMultisamplePropertiesEXT(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "samples" in indict.keys():
         samples = indict["samples"]
    else: 
         samples = c_int()
    if "pMultisampleProperties" in indict.keys():
         pMultisampleProperties = indict["pMultisampleProperties"]
    else: 
         pMultisampleProperties = VkMultisamplePropertiesEXT()
    print(jvulkanLib.vkGetPhysicalDeviceMultisamplePropertiesEXT)
    retval = jvulkanLib.vkGetPhysicalDeviceMultisamplePropertiesEXT(physicalDevice, samples, pMultisampleProperties)
    return {"physicalDevice" : physicalDevice,"samples" : samples,"pMultisampleProperties" : pMultisampleProperties,"retval" : retval}
def vkGetImageDrmFormatModifierPropertiesEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "image" in indict.keys():
         image = indict["image"]
    else: 
         image = VkImage_T()
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkImageDrmFormatModifierPropertiesEXT()
    print(jvulkanLib.vkGetImageDrmFormatModifierPropertiesEXT)
    retval = jvulkanLib.vkGetImageDrmFormatModifierPropertiesEXT(device, image, pProperties)
    return {"device" : device,"image" : image,"pProperties" : pProperties,"retval" : retval}
def vkCreateValidationCacheEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkValidationCacheCreateInfoEXT()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pValidationCache" in indict.keys():
         pValidationCache = indict["pValidationCache"]
    else: 
         pValidationCache = pointer(VkValidationCacheEXT_T())
    print(jvulkanLib.vkCreateValidationCacheEXT)
    retval = jvulkanLib.vkCreateValidationCacheEXT(device, pCreateInfo, pAllocator, pValidationCache)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pValidationCache" : pValidationCache,"retval" : retval}
def vkDestroyValidationCacheEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "validationCache" in indict.keys():
         validationCache = indict["validationCache"]
    else: 
         validationCache = VkValidationCacheEXT_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyValidationCacheEXT)
    retval = jvulkanLib.vkDestroyValidationCacheEXT(device, validationCache, pAllocator)
    return {"device" : device,"validationCache" : validationCache,"pAllocator" : pAllocator,"retval" : retval}
def vkMergeValidationCachesEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "dstCache" in indict.keys():
         dstCache = indict["dstCache"]
    else: 
         dstCache = VkValidationCacheEXT_T()
    if "srcCacheCount" in indict.keys():
         srcCacheCount = indict["srcCacheCount"]
    else: 
         srcCacheCount = c_uint()
    if "pSrcCaches" in indict.keys():
         pSrcCaches = indict["pSrcCaches"]
    else: 
         pSrcCaches = pointer(VkValidationCacheEXT_T())
    print(jvulkanLib.vkMergeValidationCachesEXT)
    retval = jvulkanLib.vkMergeValidationCachesEXT(device, dstCache, srcCacheCount, pSrcCaches)
    return {"device" : device,"dstCache" : dstCache,"srcCacheCount" : srcCacheCount,"pSrcCaches" : pSrcCaches,"retval" : retval}
def vkGetValidationCacheDataEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "validationCache" in indict.keys():
         validationCache = indict["validationCache"]
    else: 
         validationCache = VkValidationCacheEXT_T()
    if "pDataSize" in indict.keys():
         pDataSize = indict["pDataSize"]
    else: 
         pDataSize = pointer(c_ulong())
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkGetValidationCacheDataEXT)
    retval = jvulkanLib.vkGetValidationCacheDataEXT(device, validationCache, pDataSize, pData)
    return {"device" : device,"validationCache" : validationCache,"pDataSize" : pDataSize,"pData" : pData,"retval" : retval}
def vkCmdBindShadingRateImageNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "imageView" in indict.keys():
         imageView = indict["imageView"]
    else: 
         imageView = VkImageView_T()
    if "imageLayout" in indict.keys():
         imageLayout = indict["imageLayout"]
    else: 
         imageLayout = c_int()
    print(jvulkanLib.vkCmdBindShadingRateImageNV)
    retval = jvulkanLib.vkCmdBindShadingRateImageNV(commandBuffer, imageView, imageLayout)
    return {"commandBuffer" : commandBuffer,"imageView" : imageView,"imageLayout" : imageLayout,"retval" : retval}
def vkCmdSetViewportShadingRatePaletteNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstViewport" in indict.keys():
         firstViewport = indict["firstViewport"]
    else: 
         firstViewport = c_uint()
    if "viewportCount" in indict.keys():
         viewportCount = indict["viewportCount"]
    else: 
         viewportCount = c_uint()
    if "pShadingRatePalettes" in indict.keys():
         pShadingRatePalettes = indict["pShadingRatePalettes"]
    else: 
         pShadingRatePalettes = VkShadingRatePaletteNV()
    print(jvulkanLib.vkCmdSetViewportShadingRatePaletteNV)
    retval = jvulkanLib.vkCmdSetViewportShadingRatePaletteNV(commandBuffer, firstViewport, viewportCount, pShadingRatePalettes)
    return {"commandBuffer" : commandBuffer,"firstViewport" : firstViewport,"viewportCount" : viewportCount,"pShadingRatePalettes" : pShadingRatePalettes,"retval" : retval}
def vkCmdSetCoarseSampleOrderNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "sampleOrderType" in indict.keys():
         sampleOrderType = indict["sampleOrderType"]
    else: 
         sampleOrderType = c_int()
    if "customSampleOrderCount" in indict.keys():
         customSampleOrderCount = indict["customSampleOrderCount"]
    else: 
         customSampleOrderCount = c_uint()
    if "pCustomSampleOrders" in indict.keys():
         pCustomSampleOrders = indict["pCustomSampleOrders"]
    else: 
         pCustomSampleOrders = VkCoarseSampleOrderCustomNV()
    print(jvulkanLib.vkCmdSetCoarseSampleOrderNV)
    retval = jvulkanLib.vkCmdSetCoarseSampleOrderNV(commandBuffer, sampleOrderType, customSampleOrderCount, pCustomSampleOrders)
    return {"commandBuffer" : commandBuffer,"sampleOrderType" : sampleOrderType,"customSampleOrderCount" : customSampleOrderCount,"pCustomSampleOrders" : pCustomSampleOrders,"retval" : retval}
def vkCreateAccelerationStructureNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkAccelerationStructureCreateInfoNV()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pAccelerationStructure" in indict.keys():
         pAccelerationStructure = indict["pAccelerationStructure"]
    else: 
         pAccelerationStructure = pointer(VkAccelerationStructureNV_T())
    print(jvulkanLib.vkCreateAccelerationStructureNV)
    retval = jvulkanLib.vkCreateAccelerationStructureNV(device, pCreateInfo, pAllocator, pAccelerationStructure)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pAccelerationStructure" : pAccelerationStructure,"retval" : retval}
def vkDestroyAccelerationStructureNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "accelerationStructure" in indict.keys():
         accelerationStructure = indict["accelerationStructure"]
    else: 
         accelerationStructure = VkAccelerationStructureNV_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyAccelerationStructureNV)
    retval = jvulkanLib.vkDestroyAccelerationStructureNV(device, accelerationStructure, pAllocator)
    return {"device" : device,"accelerationStructure" : accelerationStructure,"pAllocator" : pAllocator,"retval" : retval}
def vkGetAccelerationStructureMemoryRequirementsNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkAccelerationStructureMemoryRequirementsInfoNV()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetAccelerationStructureMemoryRequirementsNV)
    retval = jvulkanLib.vkGetAccelerationStructureMemoryRequirementsNV(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkBindAccelerationStructureMemoryNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "bindInfoCount" in indict.keys():
         bindInfoCount = indict["bindInfoCount"]
    else: 
         bindInfoCount = c_uint()
    if "pBindInfos" in indict.keys():
         pBindInfos = indict["pBindInfos"]
    else: 
         pBindInfos = VkBindAccelerationStructureMemoryInfoNV()
    print(jvulkanLib.vkBindAccelerationStructureMemoryNV)
    retval = jvulkanLib.vkBindAccelerationStructureMemoryNV(device, bindInfoCount, pBindInfos)
    return {"device" : device,"bindInfoCount" : bindInfoCount,"pBindInfos" : pBindInfos,"retval" : retval}
def vkCmdBuildAccelerationStructureNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkAccelerationStructureInfoNV()
    if "instanceData" in indict.keys():
         instanceData = indict["instanceData"]
    else: 
         instanceData = VkBuffer_T()
    if "instanceOffset" in indict.keys():
         instanceOffset = indict["instanceOffset"]
    else: 
         instanceOffset = c_ulong()
    if "update" in indict.keys():
         update = indict["update"]
    else: 
         update = c_uint()
    if "dst" in indict.keys():
         dst = indict["dst"]
    else: 
         dst = VkAccelerationStructureNV_T()
    if "src" in indict.keys():
         src = indict["src"]
    else: 
         src = VkAccelerationStructureNV_T()
    if "scratch" in indict.keys():
         scratch = indict["scratch"]
    else: 
         scratch = VkBuffer_T()
    if "scratchOffset" in indict.keys():
         scratchOffset = indict["scratchOffset"]
    else: 
         scratchOffset = c_ulong()
    print(jvulkanLib.vkCmdBuildAccelerationStructureNV)
    retval = jvulkanLib.vkCmdBuildAccelerationStructureNV(commandBuffer, pInfo, instanceData, instanceOffset, update, dst, src, scratch, scratchOffset)
    return {"commandBuffer" : commandBuffer,"pInfo" : pInfo,"instanceData" : instanceData,"instanceOffset" : instanceOffset,"update" : update,"dst" : dst,"src" : src,"scratch" : scratch,"scratchOffset" : scratchOffset,"retval" : retval}
def vkCmdCopyAccelerationStructureNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "dst" in indict.keys():
         dst = indict["dst"]
    else: 
         dst = VkAccelerationStructureNV_T()
    if "src" in indict.keys():
         src = indict["src"]
    else: 
         src = VkAccelerationStructureNV_T()
    if "mode" in indict.keys():
         mode = indict["mode"]
    else: 
         mode = c_int()
    print(jvulkanLib.vkCmdCopyAccelerationStructureNV)
    retval = jvulkanLib.vkCmdCopyAccelerationStructureNV(commandBuffer, dst, src, mode)
    return {"commandBuffer" : commandBuffer,"dst" : dst,"src" : src,"mode" : mode,"retval" : retval}
def vkCmdTraceRaysNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "raygenShaderBindingTableBuffer" in indict.keys():
         raygenShaderBindingTableBuffer = indict["raygenShaderBindingTableBuffer"]
    else: 
         raygenShaderBindingTableBuffer = VkBuffer_T()
    if "raygenShaderBindingOffset" in indict.keys():
         raygenShaderBindingOffset = indict["raygenShaderBindingOffset"]
    else: 
         raygenShaderBindingOffset = c_ulong()
    if "missShaderBindingTableBuffer" in indict.keys():
         missShaderBindingTableBuffer = indict["missShaderBindingTableBuffer"]
    else: 
         missShaderBindingTableBuffer = VkBuffer_T()
    if "missShaderBindingOffset" in indict.keys():
         missShaderBindingOffset = indict["missShaderBindingOffset"]
    else: 
         missShaderBindingOffset = c_ulong()
    if "missShaderBindingStride" in indict.keys():
         missShaderBindingStride = indict["missShaderBindingStride"]
    else: 
         missShaderBindingStride = c_ulong()
    if "hitShaderBindingTableBuffer" in indict.keys():
         hitShaderBindingTableBuffer = indict["hitShaderBindingTableBuffer"]
    else: 
         hitShaderBindingTableBuffer = VkBuffer_T()
    if "hitShaderBindingOffset" in indict.keys():
         hitShaderBindingOffset = indict["hitShaderBindingOffset"]
    else: 
         hitShaderBindingOffset = c_ulong()
    if "hitShaderBindingStride" in indict.keys():
         hitShaderBindingStride = indict["hitShaderBindingStride"]
    else: 
         hitShaderBindingStride = c_ulong()
    if "callableShaderBindingTableBuffer" in indict.keys():
         callableShaderBindingTableBuffer = indict["callableShaderBindingTableBuffer"]
    else: 
         callableShaderBindingTableBuffer = VkBuffer_T()
    if "callableShaderBindingOffset" in indict.keys():
         callableShaderBindingOffset = indict["callableShaderBindingOffset"]
    else: 
         callableShaderBindingOffset = c_ulong()
    if "callableShaderBindingStride" in indict.keys():
         callableShaderBindingStride = indict["callableShaderBindingStride"]
    else: 
         callableShaderBindingStride = c_ulong()
    if "width" in indict.keys():
         width = indict["width"]
    else: 
         width = c_uint()
    if "height" in indict.keys():
         height = indict["height"]
    else: 
         height = c_uint()
    if "depth" in indict.keys():
         depth = indict["depth"]
    else: 
         depth = c_uint()
    print(jvulkanLib.vkCmdTraceRaysNV)
    retval = jvulkanLib.vkCmdTraceRaysNV(commandBuffer, raygenShaderBindingTableBuffer, raygenShaderBindingOffset, missShaderBindingTableBuffer, missShaderBindingOffset, missShaderBindingStride, hitShaderBindingTableBuffer, hitShaderBindingOffset, hitShaderBindingStride, callableShaderBindingTableBuffer, callableShaderBindingOffset, callableShaderBindingStride, width, height, depth)
    return {"commandBuffer" : commandBuffer,"raygenShaderBindingTableBuffer" : raygenShaderBindingTableBuffer,"raygenShaderBindingOffset" : raygenShaderBindingOffset,"missShaderBindingTableBuffer" : missShaderBindingTableBuffer,"missShaderBindingOffset" : missShaderBindingOffset,"missShaderBindingStride" : missShaderBindingStride,"hitShaderBindingTableBuffer" : hitShaderBindingTableBuffer,"hitShaderBindingOffset" : hitShaderBindingOffset,"hitShaderBindingStride" : hitShaderBindingStride,"callableShaderBindingTableBuffer" : callableShaderBindingTableBuffer,"callableShaderBindingOffset" : callableShaderBindingOffset,"callableShaderBindingStride" : callableShaderBindingStride,"width" : width,"height" : height,"depth" : depth,"retval" : retval}
def vkCreateRayTracingPipelinesNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipelineCache" in indict.keys():
         pipelineCache = indict["pipelineCache"]
    else: 
         pipelineCache = VkPipelineCache_T()
    if "createInfoCount" in indict.keys():
         createInfoCount = indict["createInfoCount"]
    else: 
         createInfoCount = c_uint()
    if "pCreateInfos" in indict.keys():
         pCreateInfos = indict["pCreateInfos"]
    else: 
         pCreateInfos = VkRayTracingPipelineCreateInfoNV()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pPipelines" in indict.keys():
         pPipelines = indict["pPipelines"]
    else: 
         pPipelines = pointer(VkPipeline_T())
    print(jvulkanLib.vkCreateRayTracingPipelinesNV)
    retval = jvulkanLib.vkCreateRayTracingPipelinesNV(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines)
    return {"device" : device,"pipelineCache" : pipelineCache,"createInfoCount" : createInfoCount,"pCreateInfos" : pCreateInfos,"pAllocator" : pAllocator,"pPipelines" : pPipelines,"retval" : retval}
def vkGetRayTracingShaderGroupHandlesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipeline" in indict.keys():
         pipeline = indict["pipeline"]
    else: 
         pipeline = VkPipeline_T()
    if "firstGroup" in indict.keys():
         firstGroup = indict["firstGroup"]
    else: 
         firstGroup = c_uint()
    if "groupCount" in indict.keys():
         groupCount = indict["groupCount"]
    else: 
         groupCount = c_uint()
    if "dataSize" in indict.keys():
         dataSize = indict["dataSize"]
    else: 
         dataSize = c_ulong()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkGetRayTracingShaderGroupHandlesKHR)
    retval = jvulkanLib.vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData)
    return {"device" : device,"pipeline" : pipeline,"firstGroup" : firstGroup,"groupCount" : groupCount,"dataSize" : dataSize,"pData" : pData,"retval" : retval}
def vkGetRayTracingShaderGroupHandlesNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipeline" in indict.keys():
         pipeline = indict["pipeline"]
    else: 
         pipeline = VkPipeline_T()
    if "firstGroup" in indict.keys():
         firstGroup = indict["firstGroup"]
    else: 
         firstGroup = c_uint()
    if "groupCount" in indict.keys():
         groupCount = indict["groupCount"]
    else: 
         groupCount = c_uint()
    if "dataSize" in indict.keys():
         dataSize = indict["dataSize"]
    else: 
         dataSize = c_ulong()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkGetRayTracingShaderGroupHandlesNV)
    retval = jvulkanLib.vkGetRayTracingShaderGroupHandlesNV(device, pipeline, firstGroup, groupCount, dataSize, pData)
    return {"device" : device,"pipeline" : pipeline,"firstGroup" : firstGroup,"groupCount" : groupCount,"dataSize" : dataSize,"pData" : pData,"retval" : retval}
def vkGetAccelerationStructureHandleNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "accelerationStructure" in indict.keys():
         accelerationStructure = indict["accelerationStructure"]
    else: 
         accelerationStructure = VkAccelerationStructureNV_T()
    if "dataSize" in indict.keys():
         dataSize = indict["dataSize"]
    else: 
         dataSize = c_ulong()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkGetAccelerationStructureHandleNV)
    retval = jvulkanLib.vkGetAccelerationStructureHandleNV(device, accelerationStructure, dataSize, pData)
    return {"device" : device,"accelerationStructure" : accelerationStructure,"dataSize" : dataSize,"pData" : pData,"retval" : retval}
def vkCmdWriteAccelerationStructuresPropertiesNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "accelerationStructureCount" in indict.keys():
         accelerationStructureCount = indict["accelerationStructureCount"]
    else: 
         accelerationStructureCount = c_uint()
    if "pAccelerationStructures" in indict.keys():
         pAccelerationStructures = indict["pAccelerationStructures"]
    else: 
         pAccelerationStructures = pointer(VkAccelerationStructureNV_T())
    if "queryType" in indict.keys():
         queryType = indict["queryType"]
    else: 
         queryType = c_int()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "firstQuery" in indict.keys():
         firstQuery = indict["firstQuery"]
    else: 
         firstQuery = c_uint()
    print(jvulkanLib.vkCmdWriteAccelerationStructuresPropertiesNV)
    retval = jvulkanLib.vkCmdWriteAccelerationStructuresPropertiesNV(commandBuffer, accelerationStructureCount, pAccelerationStructures, queryType, queryPool, firstQuery)
    return {"commandBuffer" : commandBuffer,"accelerationStructureCount" : accelerationStructureCount,"pAccelerationStructures" : pAccelerationStructures,"queryType" : queryType,"queryPool" : queryPool,"firstQuery" : firstQuery,"retval" : retval}
def vkCompileDeferredNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipeline" in indict.keys():
         pipeline = indict["pipeline"]
    else: 
         pipeline = VkPipeline_T()
    if "shader" in indict.keys():
         shader = indict["shader"]
    else: 
         shader = c_uint()
    print(jvulkanLib.vkCompileDeferredNV)
    retval = jvulkanLib.vkCompileDeferredNV(device, pipeline, shader)
    return {"device" : device,"pipeline" : pipeline,"shader" : shader,"retval" : retval}
def vkGetMemoryHostPointerPropertiesEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "handleType" in indict.keys():
         handleType = indict["handleType"]
    else: 
         handleType = c_int()
    if "pHostPointer" in indict.keys():
         pHostPointer = indict["pHostPointer"]
    else: 
         pHostPointer = c_void_p()
    if "pMemoryHostPointerProperties" in indict.keys():
         pMemoryHostPointerProperties = indict["pMemoryHostPointerProperties"]
    else: 
         pMemoryHostPointerProperties = VkMemoryHostPointerPropertiesEXT()
    print(jvulkanLib.vkGetMemoryHostPointerPropertiesEXT)
    retval = jvulkanLib.vkGetMemoryHostPointerPropertiesEXT(device, handleType, pHostPointer, pMemoryHostPointerProperties)
    return {"device" : device,"handleType" : handleType,"pHostPointer" : pHostPointer,"pMemoryHostPointerProperties" : pMemoryHostPointerProperties,"retval" : retval}
def vkCmdWriteBufferMarkerAMD(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pipelineStage" in indict.keys():
         pipelineStage = indict["pipelineStage"]
    else: 
         pipelineStage = c_int()
    if "dstBuffer" in indict.keys():
         dstBuffer = indict["dstBuffer"]
    else: 
         dstBuffer = VkBuffer_T()
    if "dstOffset" in indict.keys():
         dstOffset = indict["dstOffset"]
    else: 
         dstOffset = c_ulong()
    if "marker" in indict.keys():
         marker = indict["marker"]
    else: 
         marker = c_uint()
    print(jvulkanLib.vkCmdWriteBufferMarkerAMD)
    retval = jvulkanLib.vkCmdWriteBufferMarkerAMD(commandBuffer, pipelineStage, dstBuffer, dstOffset, marker)
    return {"commandBuffer" : commandBuffer,"pipelineStage" : pipelineStage,"dstBuffer" : dstBuffer,"dstOffset" : dstOffset,"marker" : marker,"retval" : retval}
def vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pTimeDomainCount" in indict.keys():
         pTimeDomainCount = indict["pTimeDomainCount"]
    else: 
         pTimeDomainCount = pointer(c_uint())
    if "pTimeDomains" in indict.keys():
         pTimeDomains = indict["pTimeDomains"]
    else: 
         pTimeDomains = pointer(c_int())
    print(jvulkanLib.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT)
    retval = jvulkanLib.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(physicalDevice, pTimeDomainCount, pTimeDomains)
    return {"physicalDevice" : physicalDevice,"pTimeDomainCount" : pTimeDomainCount,"pTimeDomains" : pTimeDomains,"retval" : retval}
def vkGetCalibratedTimestampsEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "timestampCount" in indict.keys():
         timestampCount = indict["timestampCount"]
    else: 
         timestampCount = c_uint()
    if "pTimestampInfos" in indict.keys():
         pTimestampInfos = indict["pTimestampInfos"]
    else: 
         pTimestampInfos = VkCalibratedTimestampInfoEXT()
    if "pTimestamps" in indict.keys():
         pTimestamps = indict["pTimestamps"]
    else: 
         pTimestamps = pointer(c_ulong())
    if "pMaxDeviation" in indict.keys():
         pMaxDeviation = indict["pMaxDeviation"]
    else: 
         pMaxDeviation = pointer(c_ulong())
    print(jvulkanLib.vkGetCalibratedTimestampsEXT)
    retval = jvulkanLib.vkGetCalibratedTimestampsEXT(device, timestampCount, pTimestampInfos, pTimestamps, pMaxDeviation)
    return {"device" : device,"timestampCount" : timestampCount,"pTimestampInfos" : pTimestampInfos,"pTimestamps" : pTimestamps,"pMaxDeviation" : pMaxDeviation,"retval" : retval}
def vkCmdDrawMeshTasksNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "taskCount" in indict.keys():
         taskCount = indict["taskCount"]
    else: 
         taskCount = c_uint()
    if "firstTask" in indict.keys():
         firstTask = indict["firstTask"]
    else: 
         firstTask = c_uint()
    print(jvulkanLib.vkCmdDrawMeshTasksNV)
    retval = jvulkanLib.vkCmdDrawMeshTasksNV(commandBuffer, taskCount, firstTask)
    return {"commandBuffer" : commandBuffer,"taskCount" : taskCount,"firstTask" : firstTask,"retval" : retval}
def vkCmdDrawMeshTasksIndirectNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "drawCount" in indict.keys():
         drawCount = indict["drawCount"]
    else: 
         drawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawMeshTasksIndirectNV)
    retval = jvulkanLib.vkCmdDrawMeshTasksIndirectNV(commandBuffer, buffer, offset, drawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"drawCount" : drawCount,"stride" : stride,"retval" : retval}
def vkCmdDrawMeshTasksIndirectCountNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "buffer" in indict.keys():
         buffer = indict["buffer"]
    else: 
         buffer = VkBuffer_T()
    if "offset" in indict.keys():
         offset = indict["offset"]
    else: 
         offset = c_ulong()
    if "countBuffer" in indict.keys():
         countBuffer = indict["countBuffer"]
    else: 
         countBuffer = VkBuffer_T()
    if "countBufferOffset" in indict.keys():
         countBufferOffset = indict["countBufferOffset"]
    else: 
         countBufferOffset = c_ulong()
    if "maxDrawCount" in indict.keys():
         maxDrawCount = indict["maxDrawCount"]
    else: 
         maxDrawCount = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawMeshTasksIndirectCountNV)
    retval = jvulkanLib.vkCmdDrawMeshTasksIndirectCountNV(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride)
    return {"commandBuffer" : commandBuffer,"buffer" : buffer,"offset" : offset,"countBuffer" : countBuffer,"countBufferOffset" : countBufferOffset,"maxDrawCount" : maxDrawCount,"stride" : stride,"retval" : retval}
def vkCmdSetExclusiveScissorNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstExclusiveScissor" in indict.keys():
         firstExclusiveScissor = indict["firstExclusiveScissor"]
    else: 
         firstExclusiveScissor = c_uint()
    if "exclusiveScissorCount" in indict.keys():
         exclusiveScissorCount = indict["exclusiveScissorCount"]
    else: 
         exclusiveScissorCount = c_uint()
    if "pExclusiveScissors" in indict.keys():
         pExclusiveScissors = indict["pExclusiveScissors"]
    else: 
         pExclusiveScissors = VkRect2D()
    print(jvulkanLib.vkCmdSetExclusiveScissorNV)
    retval = jvulkanLib.vkCmdSetExclusiveScissorNV(commandBuffer, firstExclusiveScissor, exclusiveScissorCount, pExclusiveScissors)
    return {"commandBuffer" : commandBuffer,"firstExclusiveScissor" : firstExclusiveScissor,"exclusiveScissorCount" : exclusiveScissorCount,"pExclusiveScissors" : pExclusiveScissors,"retval" : retval}
def vkCmdSetCheckpointNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pCheckpointMarker" in indict.keys():
         pCheckpointMarker = indict["pCheckpointMarker"]
    else: 
         pCheckpointMarker = c_void_p()
    print(jvulkanLib.vkCmdSetCheckpointNV)
    retval = jvulkanLib.vkCmdSetCheckpointNV(commandBuffer, pCheckpointMarker)
    return {"commandBuffer" : commandBuffer,"pCheckpointMarker" : pCheckpointMarker,"retval" : retval}
def vkGetQueueCheckpointDataNV(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "pCheckpointDataCount" in indict.keys():
         pCheckpointDataCount = indict["pCheckpointDataCount"]
    else: 
         pCheckpointDataCount = pointer(c_uint())
    if "pCheckpointData" in indict.keys():
         pCheckpointData = indict["pCheckpointData"]
    else: 
         pCheckpointData = VkCheckpointDataNV()
    print(jvulkanLib.vkGetQueueCheckpointDataNV)
    retval = jvulkanLib.vkGetQueueCheckpointDataNV(queue, pCheckpointDataCount, pCheckpointData)
    return {"queue" : queue,"pCheckpointDataCount" : pCheckpointDataCount,"pCheckpointData" : pCheckpointData,"retval" : retval}
def vkInitializePerformanceApiINTEL(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInitializeInfo" in indict.keys():
         pInitializeInfo = indict["pInitializeInfo"]
    else: 
         pInitializeInfo = VkInitializePerformanceApiInfoINTEL()
    print(jvulkanLib.vkInitializePerformanceApiINTEL)
    retval = jvulkanLib.vkInitializePerformanceApiINTEL(device, pInitializeInfo)
    return {"device" : device,"pInitializeInfo" : pInitializeInfo,"retval" : retval}
def vkUninitializePerformanceApiINTEL(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    print(jvulkanLib.vkUninitializePerformanceApiINTEL)
    retval = jvulkanLib.vkUninitializePerformanceApiINTEL(device)
    return {"device" : device,"retval" : retval}
def vkCmdSetPerformanceMarkerINTEL(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pMarkerInfo" in indict.keys():
         pMarkerInfo = indict["pMarkerInfo"]
    else: 
         pMarkerInfo = VkPerformanceMarkerInfoINTEL()
    print(jvulkanLib.vkCmdSetPerformanceMarkerINTEL)
    retval = jvulkanLib.vkCmdSetPerformanceMarkerINTEL(commandBuffer, pMarkerInfo)
    return {"commandBuffer" : commandBuffer,"pMarkerInfo" : pMarkerInfo,"retval" : retval}
def vkCmdSetPerformanceStreamMarkerINTEL(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pMarkerInfo" in indict.keys():
         pMarkerInfo = indict["pMarkerInfo"]
    else: 
         pMarkerInfo = VkPerformanceStreamMarkerInfoINTEL()
    print(jvulkanLib.vkCmdSetPerformanceStreamMarkerINTEL)
    retval = jvulkanLib.vkCmdSetPerformanceStreamMarkerINTEL(commandBuffer, pMarkerInfo)
    return {"commandBuffer" : commandBuffer,"pMarkerInfo" : pMarkerInfo,"retval" : retval}
def vkCmdSetPerformanceOverrideINTEL(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pOverrideInfo" in indict.keys():
         pOverrideInfo = indict["pOverrideInfo"]
    else: 
         pOverrideInfo = VkPerformanceOverrideInfoINTEL()
    print(jvulkanLib.vkCmdSetPerformanceOverrideINTEL)
    retval = jvulkanLib.vkCmdSetPerformanceOverrideINTEL(commandBuffer, pOverrideInfo)
    return {"commandBuffer" : commandBuffer,"pOverrideInfo" : pOverrideInfo,"retval" : retval}
def vkAcquirePerformanceConfigurationINTEL(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pAcquireInfo" in indict.keys():
         pAcquireInfo = indict["pAcquireInfo"]
    else: 
         pAcquireInfo = VkPerformanceConfigurationAcquireInfoINTEL()
    if "pConfiguration" in indict.keys():
         pConfiguration = indict["pConfiguration"]
    else: 
         pConfiguration = pointer(VkPerformanceConfigurationINTEL_T())
    print(jvulkanLib.vkAcquirePerformanceConfigurationINTEL)
    retval = jvulkanLib.vkAcquirePerformanceConfigurationINTEL(device, pAcquireInfo, pConfiguration)
    return {"device" : device,"pAcquireInfo" : pAcquireInfo,"pConfiguration" : pConfiguration,"retval" : retval}
def vkReleasePerformanceConfigurationINTEL(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "configuration" in indict.keys():
         configuration = indict["configuration"]
    else: 
         configuration = VkPerformanceConfigurationINTEL_T()
    print(jvulkanLib.vkReleasePerformanceConfigurationINTEL)
    retval = jvulkanLib.vkReleasePerformanceConfigurationINTEL(device, configuration)
    return {"device" : device,"configuration" : configuration,"retval" : retval}
def vkQueueSetPerformanceConfigurationINTEL(indict):
    if "queue" in indict.keys():
         queue = indict["queue"]
    else: 
         queue = VkQueue_T()
    if "configuration" in indict.keys():
         configuration = indict["configuration"]
    else: 
         configuration = VkPerformanceConfigurationINTEL_T()
    print(jvulkanLib.vkQueueSetPerformanceConfigurationINTEL)
    retval = jvulkanLib.vkQueueSetPerformanceConfigurationINTEL(queue, configuration)
    return {"queue" : queue,"configuration" : configuration,"retval" : retval}
def vkGetPerformanceParameterINTEL(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "parameter" in indict.keys():
         parameter = indict["parameter"]
    else: 
         parameter = c_int()
    if "pValue" in indict.keys():
         pValue = indict["pValue"]
    else: 
         pValue = VkPerformanceValueINTEL()
    print(jvulkanLib.vkGetPerformanceParameterINTEL)
    retval = jvulkanLib.vkGetPerformanceParameterINTEL(device, parameter, pValue)
    return {"device" : device,"parameter" : parameter,"pValue" : pValue,"retval" : retval}
def vkSetLocalDimmingAMD(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "swapChain" in indict.keys():
         swapChain = indict["swapChain"]
    else: 
         swapChain = VkSwapchainKHR_T()
    if "localDimmingEnable" in indict.keys():
         localDimmingEnable = indict["localDimmingEnable"]
    else: 
         localDimmingEnable = c_uint()
    print(jvulkanLib.vkSetLocalDimmingAMD)
    retval = jvulkanLib.vkSetLocalDimmingAMD(device, swapChain, localDimmingEnable)
    return {"device" : device,"swapChain" : swapChain,"localDimmingEnable" : localDimmingEnable,"retval" : retval}
def vkGetBufferDeviceAddressEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkBufferDeviceAddressInfo()
    print(jvulkanLib.vkGetBufferDeviceAddressEXT)
    retval = jvulkanLib.vkGetBufferDeviceAddressEXT(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkGetPhysicalDeviceToolPropertiesEXT(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pToolCount" in indict.keys():
         pToolCount = indict["pToolCount"]
    else: 
         pToolCount = pointer(c_uint())
    if "pToolProperties" in indict.keys():
         pToolProperties = indict["pToolProperties"]
    else: 
         pToolProperties = VkPhysicalDeviceToolProperties()
    print(jvulkanLib.vkGetPhysicalDeviceToolPropertiesEXT)
    retval = jvulkanLib.vkGetPhysicalDeviceToolPropertiesEXT(physicalDevice, pToolCount, pToolProperties)
    return {"physicalDevice" : physicalDevice,"pToolCount" : pToolCount,"pToolProperties" : pToolProperties,"retval" : retval}
def vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pPropertyCount" in indict.keys():
         pPropertyCount = indict["pPropertyCount"]
    else: 
         pPropertyCount = pointer(c_uint())
    if "pProperties" in indict.keys():
         pProperties = indict["pProperties"]
    else: 
         pProperties = VkCooperativeMatrixPropertiesNV()
    print(jvulkanLib.vkGetPhysicalDeviceCooperativeMatrixPropertiesNV)
    retval = jvulkanLib.vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(physicalDevice, pPropertyCount, pProperties)
    return {"physicalDevice" : physicalDevice,"pPropertyCount" : pPropertyCount,"pProperties" : pProperties,"retval" : retval}
def vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "pCombinationCount" in indict.keys():
         pCombinationCount = indict["pCombinationCount"]
    else: 
         pCombinationCount = pointer(c_uint())
    if "pCombinations" in indict.keys():
         pCombinations = indict["pCombinations"]
    else: 
         pCombinations = VkFramebufferMixedSamplesCombinationNV()
    print(jvulkanLib.vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV)
    retval = jvulkanLib.vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(physicalDevice, pCombinationCount, pCombinations)
    return {"physicalDevice" : physicalDevice,"pCombinationCount" : pCombinationCount,"pCombinations" : pCombinations,"retval" : retval}
def vkCreateHeadlessSurfaceEXT(indict):
    if "instance" in indict.keys():
         instance = indict["instance"]
    else: 
         instance = VkInstance_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkHeadlessSurfaceCreateInfoEXT()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pSurface" in indict.keys():
         pSurface = indict["pSurface"]
    else: 
         pSurface = pointer(VkSurfaceKHR_T())
    print(jvulkanLib.vkCreateHeadlessSurfaceEXT)
    retval = jvulkanLib.vkCreateHeadlessSurfaceEXT(instance, pCreateInfo, pAllocator, pSurface)
    return {"instance" : instance,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pSurface" : pSurface,"retval" : retval}
def vkCmdSetLineStippleEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "lineStippleFactor" in indict.keys():
         lineStippleFactor = indict["lineStippleFactor"]
    else: 
         lineStippleFactor = c_uint()
    if "lineStipplePattern" in indict.keys():
         lineStipplePattern = indict["lineStipplePattern"]
    else: 
         lineStipplePattern = c_ushort()
    print(jvulkanLib.vkCmdSetLineStippleEXT)
    retval = jvulkanLib.vkCmdSetLineStippleEXT(commandBuffer, lineStippleFactor, lineStipplePattern)
    return {"commandBuffer" : commandBuffer,"lineStippleFactor" : lineStippleFactor,"lineStipplePattern" : lineStipplePattern,"retval" : retval}
def vkResetQueryPoolEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "firstQuery" in indict.keys():
         firstQuery = indict["firstQuery"]
    else: 
         firstQuery = c_uint()
    if "queryCount" in indict.keys():
         queryCount = indict["queryCount"]
    else: 
         queryCount = c_uint()
    print(jvulkanLib.vkResetQueryPoolEXT)
    retval = jvulkanLib.vkResetQueryPoolEXT(device, queryPool, firstQuery, queryCount)
    return {"device" : device,"queryPool" : queryPool,"firstQuery" : firstQuery,"queryCount" : queryCount,"retval" : retval}
def vkCmdSetCullModeEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "cullMode" in indict.keys():
         cullMode = indict["cullMode"]
    else: 
         cullMode = c_uint()
    print(jvulkanLib.vkCmdSetCullModeEXT)
    retval = jvulkanLib.vkCmdSetCullModeEXT(commandBuffer, cullMode)
    return {"commandBuffer" : commandBuffer,"cullMode" : cullMode,"retval" : retval}
def vkCmdSetFrontFaceEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "frontFace" in indict.keys():
         frontFace = indict["frontFace"]
    else: 
         frontFace = c_int()
    print(jvulkanLib.vkCmdSetFrontFaceEXT)
    retval = jvulkanLib.vkCmdSetFrontFaceEXT(commandBuffer, frontFace)
    return {"commandBuffer" : commandBuffer,"frontFace" : frontFace,"retval" : retval}
def vkCmdSetPrimitiveTopologyEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "primitiveTopology" in indict.keys():
         primitiveTopology = indict["primitiveTopology"]
    else: 
         primitiveTopology = c_int()
    print(jvulkanLib.vkCmdSetPrimitiveTopologyEXT)
    retval = jvulkanLib.vkCmdSetPrimitiveTopologyEXT(commandBuffer, primitiveTopology)
    return {"commandBuffer" : commandBuffer,"primitiveTopology" : primitiveTopology,"retval" : retval}
def vkCmdSetViewportWithCountEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "viewportCount" in indict.keys():
         viewportCount = indict["viewportCount"]
    else: 
         viewportCount = c_uint()
    if "pViewports" in indict.keys():
         pViewports = indict["pViewports"]
    else: 
         pViewports = VkViewport()
    print(jvulkanLib.vkCmdSetViewportWithCountEXT)
    retval = jvulkanLib.vkCmdSetViewportWithCountEXT(commandBuffer, viewportCount, pViewports)
    return {"commandBuffer" : commandBuffer,"viewportCount" : viewportCount,"pViewports" : pViewports,"retval" : retval}
def vkCmdSetScissorWithCountEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "scissorCount" in indict.keys():
         scissorCount = indict["scissorCount"]
    else: 
         scissorCount = c_uint()
    if "pScissors" in indict.keys():
         pScissors = indict["pScissors"]
    else: 
         pScissors = VkRect2D()
    print(jvulkanLib.vkCmdSetScissorWithCountEXT)
    retval = jvulkanLib.vkCmdSetScissorWithCountEXT(commandBuffer, scissorCount, pScissors)
    return {"commandBuffer" : commandBuffer,"scissorCount" : scissorCount,"pScissors" : pScissors,"retval" : retval}
def vkCmdBindVertexBuffers2EXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "firstBinding" in indict.keys():
         firstBinding = indict["firstBinding"]
    else: 
         firstBinding = c_uint()
    if "bindingCount" in indict.keys():
         bindingCount = indict["bindingCount"]
    else: 
         bindingCount = c_uint()
    if "pBuffers" in indict.keys():
         pBuffers = indict["pBuffers"]
    else: 
         pBuffers = pointer(VkBuffer_T())
    if "pOffsets" in indict.keys():
         pOffsets = indict["pOffsets"]
    else: 
         pOffsets = pointer(c_ulong())
    if "pSizes" in indict.keys():
         pSizes = indict["pSizes"]
    else: 
         pSizes = pointer(c_ulong())
    if "pStrides" in indict.keys():
         pStrides = indict["pStrides"]
    else: 
         pStrides = pointer(c_ulong())
    print(jvulkanLib.vkCmdBindVertexBuffers2EXT)
    retval = jvulkanLib.vkCmdBindVertexBuffers2EXT(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes, pStrides)
    return {"commandBuffer" : commandBuffer,"firstBinding" : firstBinding,"bindingCount" : bindingCount,"pBuffers" : pBuffers,"pOffsets" : pOffsets,"pSizes" : pSizes,"pStrides" : pStrides,"retval" : retval}
def vkCmdSetDepthTestEnableEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthTestEnable" in indict.keys():
         depthTestEnable = indict["depthTestEnable"]
    else: 
         depthTestEnable = c_uint()
    print(jvulkanLib.vkCmdSetDepthTestEnableEXT)
    retval = jvulkanLib.vkCmdSetDepthTestEnableEXT(commandBuffer, depthTestEnable)
    return {"commandBuffer" : commandBuffer,"depthTestEnable" : depthTestEnable,"retval" : retval}
def vkCmdSetDepthWriteEnableEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthWriteEnable" in indict.keys():
         depthWriteEnable = indict["depthWriteEnable"]
    else: 
         depthWriteEnable = c_uint()
    print(jvulkanLib.vkCmdSetDepthWriteEnableEXT)
    retval = jvulkanLib.vkCmdSetDepthWriteEnableEXT(commandBuffer, depthWriteEnable)
    return {"commandBuffer" : commandBuffer,"depthWriteEnable" : depthWriteEnable,"retval" : retval}
def vkCmdSetDepthCompareOpEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthCompareOp" in indict.keys():
         depthCompareOp = indict["depthCompareOp"]
    else: 
         depthCompareOp = c_int()
    print(jvulkanLib.vkCmdSetDepthCompareOpEXT)
    retval = jvulkanLib.vkCmdSetDepthCompareOpEXT(commandBuffer, depthCompareOp)
    return {"commandBuffer" : commandBuffer,"depthCompareOp" : depthCompareOp,"retval" : retval}
def vkCmdSetDepthBoundsTestEnableEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthBoundsTestEnable" in indict.keys():
         depthBoundsTestEnable = indict["depthBoundsTestEnable"]
    else: 
         depthBoundsTestEnable = c_uint()
    print(jvulkanLib.vkCmdSetDepthBoundsTestEnableEXT)
    retval = jvulkanLib.vkCmdSetDepthBoundsTestEnableEXT(commandBuffer, depthBoundsTestEnable)
    return {"commandBuffer" : commandBuffer,"depthBoundsTestEnable" : depthBoundsTestEnable,"retval" : retval}
def vkCmdSetStencilTestEnableEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "stencilTestEnable" in indict.keys():
         stencilTestEnable = indict["stencilTestEnable"]
    else: 
         stencilTestEnable = c_uint()
    print(jvulkanLib.vkCmdSetStencilTestEnableEXT)
    retval = jvulkanLib.vkCmdSetStencilTestEnableEXT(commandBuffer, stencilTestEnable)
    return {"commandBuffer" : commandBuffer,"stencilTestEnable" : stencilTestEnable,"retval" : retval}
def vkCmdSetStencilOpEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "faceMask" in indict.keys():
         faceMask = indict["faceMask"]
    else: 
         faceMask = c_uint()
    if "failOp" in indict.keys():
         failOp = indict["failOp"]
    else: 
         failOp = c_int()
    if "passOp" in indict.keys():
         passOp = indict["passOp"]
    else: 
         passOp = c_int()
    if "depthFailOp" in indict.keys():
         depthFailOp = indict["depthFailOp"]
    else: 
         depthFailOp = c_int()
    if "compareOp" in indict.keys():
         compareOp = indict["compareOp"]
    else: 
         compareOp = c_int()
    print(jvulkanLib.vkCmdSetStencilOpEXT)
    retval = jvulkanLib.vkCmdSetStencilOpEXT(commandBuffer, faceMask, failOp, passOp, depthFailOp, compareOp)
    return {"commandBuffer" : commandBuffer,"faceMask" : faceMask,"failOp" : failOp,"passOp" : passOp,"depthFailOp" : depthFailOp,"compareOp" : compareOp,"retval" : retval}
def vkGetGeneratedCommandsMemoryRequirementsNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkGeneratedCommandsMemoryRequirementsInfoNV()
    if "pMemoryRequirements" in indict.keys():
         pMemoryRequirements = indict["pMemoryRequirements"]
    else: 
         pMemoryRequirements = VkMemoryRequirements2()
    print(jvulkanLib.vkGetGeneratedCommandsMemoryRequirementsNV)
    retval = jvulkanLib.vkGetGeneratedCommandsMemoryRequirementsNV(device, pInfo, pMemoryRequirements)
    return {"device" : device,"pInfo" : pInfo,"pMemoryRequirements" : pMemoryRequirements,"retval" : retval}
def vkCmdPreprocessGeneratedCommandsNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pGeneratedCommandsInfo" in indict.keys():
         pGeneratedCommandsInfo = indict["pGeneratedCommandsInfo"]
    else: 
         pGeneratedCommandsInfo = VkGeneratedCommandsInfoNV()
    print(jvulkanLib.vkCmdPreprocessGeneratedCommandsNV)
    retval = jvulkanLib.vkCmdPreprocessGeneratedCommandsNV(commandBuffer, pGeneratedCommandsInfo)
    return {"commandBuffer" : commandBuffer,"pGeneratedCommandsInfo" : pGeneratedCommandsInfo,"retval" : retval}
def vkCmdExecuteGeneratedCommandsNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "isPreprocessed" in indict.keys():
         isPreprocessed = indict["isPreprocessed"]
    else: 
         isPreprocessed = c_uint()
    if "pGeneratedCommandsInfo" in indict.keys():
         pGeneratedCommandsInfo = indict["pGeneratedCommandsInfo"]
    else: 
         pGeneratedCommandsInfo = VkGeneratedCommandsInfoNV()
    print(jvulkanLib.vkCmdExecuteGeneratedCommandsNV)
    retval = jvulkanLib.vkCmdExecuteGeneratedCommandsNV(commandBuffer, isPreprocessed, pGeneratedCommandsInfo)
    return {"commandBuffer" : commandBuffer,"isPreprocessed" : isPreprocessed,"pGeneratedCommandsInfo" : pGeneratedCommandsInfo,"retval" : retval}
def vkCmdBindPipelineShaderGroupNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pipelineBindPoint" in indict.keys():
         pipelineBindPoint = indict["pipelineBindPoint"]
    else: 
         pipelineBindPoint = c_int()
    if "pipeline" in indict.keys():
         pipeline = indict["pipeline"]
    else: 
         pipeline = VkPipeline_T()
    if "groupIndex" in indict.keys():
         groupIndex = indict["groupIndex"]
    else: 
         groupIndex = c_uint()
    print(jvulkanLib.vkCmdBindPipelineShaderGroupNV)
    retval = jvulkanLib.vkCmdBindPipelineShaderGroupNV(commandBuffer, pipelineBindPoint, pipeline, groupIndex)
    return {"commandBuffer" : commandBuffer,"pipelineBindPoint" : pipelineBindPoint,"pipeline" : pipeline,"groupIndex" : groupIndex,"retval" : retval}
def vkCreateIndirectCommandsLayoutNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkIndirectCommandsLayoutCreateInfoNV()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pIndirectCommandsLayout" in indict.keys():
         pIndirectCommandsLayout = indict["pIndirectCommandsLayout"]
    else: 
         pIndirectCommandsLayout = pointer(VkIndirectCommandsLayoutNV_T())
    print(jvulkanLib.vkCreateIndirectCommandsLayoutNV)
    retval = jvulkanLib.vkCreateIndirectCommandsLayoutNV(device, pCreateInfo, pAllocator, pIndirectCommandsLayout)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pIndirectCommandsLayout" : pIndirectCommandsLayout,"retval" : retval}
def vkDestroyIndirectCommandsLayoutNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "indirectCommandsLayout" in indict.keys():
         indirectCommandsLayout = indict["indirectCommandsLayout"]
    else: 
         indirectCommandsLayout = VkIndirectCommandsLayoutNV_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyIndirectCommandsLayoutNV)
    retval = jvulkanLib.vkDestroyIndirectCommandsLayoutNV(device, indirectCommandsLayout, pAllocator)
    return {"device" : device,"indirectCommandsLayout" : indirectCommandsLayout,"pAllocator" : pAllocator,"retval" : retval}
def vkAcquireDrmDisplayEXT(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "drmFd" in indict.keys():
         drmFd = indict["drmFd"]
    else: 
         drmFd = c_int()
    if "display" in indict.keys():
         display = indict["display"]
    else: 
         display = VkDisplayKHR_T()
    print(jvulkanLib.vkAcquireDrmDisplayEXT)
    retval = jvulkanLib.vkAcquireDrmDisplayEXT(physicalDevice, drmFd, display)
    return {"physicalDevice" : physicalDevice,"drmFd" : drmFd,"display" : display,"retval" : retval}
def vkGetDrmDisplayEXT(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "drmFd" in indict.keys():
         drmFd = indict["drmFd"]
    else: 
         drmFd = c_int()
    if "connectorId" in indict.keys():
         connectorId = indict["connectorId"]
    else: 
         connectorId = c_uint()
    if "display" in indict.keys():
         display = indict["display"]
    else: 
         display = pointer(VkDisplayKHR_T())
    print(jvulkanLib.vkGetDrmDisplayEXT)
    retval = jvulkanLib.vkGetDrmDisplayEXT(physicalDevice, drmFd, connectorId, display)
    return {"physicalDevice" : physicalDevice,"drmFd" : drmFd,"connectorId" : connectorId,"display" : display,"retval" : retval}
def vkCreatePrivateDataSlotEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkPrivateDataSlotCreateInfo()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pPrivateDataSlot" in indict.keys():
         pPrivateDataSlot = indict["pPrivateDataSlot"]
    else: 
         pPrivateDataSlot = pointer(VkPrivateDataSlot_T())
    print(jvulkanLib.vkCreatePrivateDataSlotEXT)
    retval = jvulkanLib.vkCreatePrivateDataSlotEXT(device, pCreateInfo, pAllocator, pPrivateDataSlot)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pPrivateDataSlot" : pPrivateDataSlot,"retval" : retval}
def vkDestroyPrivateDataSlotEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "privateDataSlot" in indict.keys():
         privateDataSlot = indict["privateDataSlot"]
    else: 
         privateDataSlot = VkPrivateDataSlot_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyPrivateDataSlotEXT)
    retval = jvulkanLib.vkDestroyPrivateDataSlotEXT(device, privateDataSlot, pAllocator)
    return {"device" : device,"privateDataSlot" : privateDataSlot,"pAllocator" : pAllocator,"retval" : retval}
def vkSetPrivateDataEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "objectType" in indict.keys():
         objectType = indict["objectType"]
    else: 
         objectType = c_int()
    if "objectHandle" in indict.keys():
         objectHandle = indict["objectHandle"]
    else: 
         objectHandle = c_ulong()
    if "privateDataSlot" in indict.keys():
         privateDataSlot = indict["privateDataSlot"]
    else: 
         privateDataSlot = VkPrivateDataSlot_T()
    if "data" in indict.keys():
         data = indict["data"]
    else: 
         data = c_ulong()
    print(jvulkanLib.vkSetPrivateDataEXT)
    retval = jvulkanLib.vkSetPrivateDataEXT(device, objectType, objectHandle, privateDataSlot, data)
    return {"device" : device,"objectType" : objectType,"objectHandle" : objectHandle,"privateDataSlot" : privateDataSlot,"data" : data,"retval" : retval}
def vkGetPrivateDataEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "objectType" in indict.keys():
         objectType = indict["objectType"]
    else: 
         objectType = c_int()
    if "objectHandle" in indict.keys():
         objectHandle = indict["objectHandle"]
    else: 
         objectHandle = c_ulong()
    if "privateDataSlot" in indict.keys():
         privateDataSlot = indict["privateDataSlot"]
    else: 
         privateDataSlot = VkPrivateDataSlot_T()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = pointer(c_ulong())
    print(jvulkanLib.vkGetPrivateDataEXT)
    retval = jvulkanLib.vkGetPrivateDataEXT(device, objectType, objectHandle, privateDataSlot, pData)
    return {"device" : device,"objectType" : objectType,"objectHandle" : objectHandle,"privateDataSlot" : privateDataSlot,"pData" : pData,"retval" : retval}
def vkCmdSetFragmentShadingRateEnumNV(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "shadingRate" in indict.keys():
         shadingRate = indict["shadingRate"]
    else: 
         shadingRate = c_int()
    if "combinerOps" in indict.keys():
         combinerOps = indict["combinerOps"]
    else: 
         combinerOps = pointer(c_int())
    print(jvulkanLib.vkCmdSetFragmentShadingRateEnumNV)
    retval = jvulkanLib.vkCmdSetFragmentShadingRateEnumNV(commandBuffer, shadingRate, combinerOps)
    return {"commandBuffer" : commandBuffer,"shadingRate" : shadingRate,"combinerOps" : combinerOps,"retval" : retval}
def vkAcquireWinrtDisplayNV(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "display" in indict.keys():
         display = indict["display"]
    else: 
         display = VkDisplayKHR_T()
    print(jvulkanLib.vkAcquireWinrtDisplayNV)
    retval = jvulkanLib.vkAcquireWinrtDisplayNV(physicalDevice, display)
    return {"physicalDevice" : physicalDevice,"display" : display,"retval" : retval}
def vkGetWinrtDisplayNV(indict):
    if "physicalDevice" in indict.keys():
         physicalDevice = indict["physicalDevice"]
    else: 
         physicalDevice = VkPhysicalDevice_T()
    if "deviceRelativeId" in indict.keys():
         deviceRelativeId = indict["deviceRelativeId"]
    else: 
         deviceRelativeId = c_uint()
    if "pDisplay" in indict.keys():
         pDisplay = indict["pDisplay"]
    else: 
         pDisplay = pointer(VkDisplayKHR_T())
    print(jvulkanLib.vkGetWinrtDisplayNV)
    retval = jvulkanLib.vkGetWinrtDisplayNV(physicalDevice, deviceRelativeId, pDisplay)
    return {"physicalDevice" : physicalDevice,"deviceRelativeId" : deviceRelativeId,"pDisplay" : pDisplay,"retval" : retval}
def vkCmdSetVertexInputEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "vertexBindingDescriptionCount" in indict.keys():
         vertexBindingDescriptionCount = indict["vertexBindingDescriptionCount"]
    else: 
         vertexBindingDescriptionCount = c_uint()
    if "pVertexBindingDescriptions" in indict.keys():
         pVertexBindingDescriptions = indict["pVertexBindingDescriptions"]
    else: 
         pVertexBindingDescriptions = VkVertexInputBindingDescription2EXT()
    if "vertexAttributeDescriptionCount" in indict.keys():
         vertexAttributeDescriptionCount = indict["vertexAttributeDescriptionCount"]
    else: 
         vertexAttributeDescriptionCount = c_uint()
    if "pVertexAttributeDescriptions" in indict.keys():
         pVertexAttributeDescriptions = indict["pVertexAttributeDescriptions"]
    else: 
         pVertexAttributeDescriptions = VkVertexInputAttributeDescription2EXT()
    print(jvulkanLib.vkCmdSetVertexInputEXT)
    retval = jvulkanLib.vkCmdSetVertexInputEXT(commandBuffer, vertexBindingDescriptionCount, pVertexBindingDescriptions, vertexAttributeDescriptionCount, pVertexAttributeDescriptions)
    return {"commandBuffer" : commandBuffer,"vertexBindingDescriptionCount" : vertexBindingDescriptionCount,"pVertexBindingDescriptions" : pVertexBindingDescriptions,"vertexAttributeDescriptionCount" : vertexAttributeDescriptionCount,"pVertexAttributeDescriptions" : pVertexAttributeDescriptions,"retval" : retval}
def vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "renderpass" in indict.keys():
         renderpass = indict["renderpass"]
    else: 
         renderpass = VkRenderPass_T()
    if "pMaxWorkgroupSize" in indict.keys():
         pMaxWorkgroupSize = indict["pMaxWorkgroupSize"]
    else: 
         pMaxWorkgroupSize = VkExtent2D()
    print(jvulkanLib.vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI)
    retval = jvulkanLib.vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI(device, renderpass, pMaxWorkgroupSize)
    return {"device" : device,"renderpass" : renderpass,"pMaxWorkgroupSize" : pMaxWorkgroupSize,"retval" : retval}
def vkCmdSubpassShadingHUAWEI(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    print(jvulkanLib.vkCmdSubpassShadingHUAWEI)
    retval = jvulkanLib.vkCmdSubpassShadingHUAWEI(commandBuffer)
    return {"commandBuffer" : commandBuffer,"retval" : retval}
def vkCmdBindInvocationMaskHUAWEI(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "imageView" in indict.keys():
         imageView = indict["imageView"]
    else: 
         imageView = VkImageView_T()
    if "imageLayout" in indict.keys():
         imageLayout = indict["imageLayout"]
    else: 
         imageLayout = c_int()
    print(jvulkanLib.vkCmdBindInvocationMaskHUAWEI)
    retval = jvulkanLib.vkCmdBindInvocationMaskHUAWEI(commandBuffer, imageView, imageLayout)
    return {"commandBuffer" : commandBuffer,"imageView" : imageView,"imageLayout" : imageLayout,"retval" : retval}
def vkGetMemoryRemoteAddressNV(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pMemoryGetRemoteAddressInfo" in indict.keys():
         pMemoryGetRemoteAddressInfo = indict["pMemoryGetRemoteAddressInfo"]
    else: 
         pMemoryGetRemoteAddressInfo = VkMemoryGetRemoteAddressInfoNV()
    if "pAddress" in indict.keys():
         pAddress = indict["pAddress"]
    else: 
         pAddress = pointer(c_void_p())
    print(jvulkanLib.vkGetMemoryRemoteAddressNV)
    retval = jvulkanLib.vkGetMemoryRemoteAddressNV(device, pMemoryGetRemoteAddressInfo, pAddress)
    return {"device" : device,"pMemoryGetRemoteAddressInfo" : pMemoryGetRemoteAddressInfo,"pAddress" : pAddress,"retval" : retval}
def vkCmdSetPatchControlPointsEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "patchControlPoints" in indict.keys():
         patchControlPoints = indict["patchControlPoints"]
    else: 
         patchControlPoints = c_uint()
    print(jvulkanLib.vkCmdSetPatchControlPointsEXT)
    retval = jvulkanLib.vkCmdSetPatchControlPointsEXT(commandBuffer, patchControlPoints)
    return {"commandBuffer" : commandBuffer,"patchControlPoints" : patchControlPoints,"retval" : retval}
def vkCmdSetRasterizerDiscardEnableEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "rasterizerDiscardEnable" in indict.keys():
         rasterizerDiscardEnable = indict["rasterizerDiscardEnable"]
    else: 
         rasterizerDiscardEnable = c_uint()
    print(jvulkanLib.vkCmdSetRasterizerDiscardEnableEXT)
    retval = jvulkanLib.vkCmdSetRasterizerDiscardEnableEXT(commandBuffer, rasterizerDiscardEnable)
    return {"commandBuffer" : commandBuffer,"rasterizerDiscardEnable" : rasterizerDiscardEnable,"retval" : retval}
def vkCmdSetDepthBiasEnableEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "depthBiasEnable" in indict.keys():
         depthBiasEnable = indict["depthBiasEnable"]
    else: 
         depthBiasEnable = c_uint()
    print(jvulkanLib.vkCmdSetDepthBiasEnableEXT)
    retval = jvulkanLib.vkCmdSetDepthBiasEnableEXT(commandBuffer, depthBiasEnable)
    return {"commandBuffer" : commandBuffer,"depthBiasEnable" : depthBiasEnable,"retval" : retval}
def vkCmdSetLogicOpEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "logicOp" in indict.keys():
         logicOp = indict["logicOp"]
    else: 
         logicOp = c_int()
    print(jvulkanLib.vkCmdSetLogicOpEXT)
    retval = jvulkanLib.vkCmdSetLogicOpEXT(commandBuffer, logicOp)
    return {"commandBuffer" : commandBuffer,"logicOp" : logicOp,"retval" : retval}
def vkCmdSetPrimitiveRestartEnableEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "primitiveRestartEnable" in indict.keys():
         primitiveRestartEnable = indict["primitiveRestartEnable"]
    else: 
         primitiveRestartEnable = c_uint()
    print(jvulkanLib.vkCmdSetPrimitiveRestartEnableEXT)
    retval = jvulkanLib.vkCmdSetPrimitiveRestartEnableEXT(commandBuffer, primitiveRestartEnable)
    return {"commandBuffer" : commandBuffer,"primitiveRestartEnable" : primitiveRestartEnable,"retval" : retval}
def vkCmdSetColorWriteEnableEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "attachmentCount" in indict.keys():
         attachmentCount = indict["attachmentCount"]
    else: 
         attachmentCount = c_uint()
    if "pColorWriteEnables" in indict.keys():
         pColorWriteEnables = indict["pColorWriteEnables"]
    else: 
         pColorWriteEnables = pointer(c_uint())
    print(jvulkanLib.vkCmdSetColorWriteEnableEXT)
    retval = jvulkanLib.vkCmdSetColorWriteEnableEXT(commandBuffer, attachmentCount, pColorWriteEnables)
    return {"commandBuffer" : commandBuffer,"attachmentCount" : attachmentCount,"pColorWriteEnables" : pColorWriteEnables,"retval" : retval}
def vkCmdDrawMultiEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "drawCount" in indict.keys():
         drawCount = indict["drawCount"]
    else: 
         drawCount = c_uint()
    if "pVertexInfo" in indict.keys():
         pVertexInfo = indict["pVertexInfo"]
    else: 
         pVertexInfo = VkMultiDrawInfoEXT()
    if "instanceCount" in indict.keys():
         instanceCount = indict["instanceCount"]
    else: 
         instanceCount = c_uint()
    if "firstInstance" in indict.keys():
         firstInstance = indict["firstInstance"]
    else: 
         firstInstance = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    print(jvulkanLib.vkCmdDrawMultiEXT)
    retval = jvulkanLib.vkCmdDrawMultiEXT(commandBuffer, drawCount, pVertexInfo, instanceCount, firstInstance, stride)
    return {"commandBuffer" : commandBuffer,"drawCount" : drawCount,"pVertexInfo" : pVertexInfo,"instanceCount" : instanceCount,"firstInstance" : firstInstance,"stride" : stride,"retval" : retval}
def vkCmdDrawMultiIndexedEXT(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "drawCount" in indict.keys():
         drawCount = indict["drawCount"]
    else: 
         drawCount = c_uint()
    if "pIndexInfo" in indict.keys():
         pIndexInfo = indict["pIndexInfo"]
    else: 
         pIndexInfo = VkMultiDrawIndexedInfoEXT()
    if "instanceCount" in indict.keys():
         instanceCount = indict["instanceCount"]
    else: 
         instanceCount = c_uint()
    if "firstInstance" in indict.keys():
         firstInstance = indict["firstInstance"]
    else: 
         firstInstance = c_uint()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_uint()
    if "pVertexOffset" in indict.keys():
         pVertexOffset = indict["pVertexOffset"]
    else: 
         pVertexOffset = pointer(c_int())
    print(jvulkanLib.vkCmdDrawMultiIndexedEXT)
    retval = jvulkanLib.vkCmdDrawMultiIndexedEXT(commandBuffer, drawCount, pIndexInfo, instanceCount, firstInstance, stride, pVertexOffset)
    return {"commandBuffer" : commandBuffer,"drawCount" : drawCount,"pIndexInfo" : pIndexInfo,"instanceCount" : instanceCount,"firstInstance" : firstInstance,"stride" : stride,"pVertexOffset" : pVertexOffset,"retval" : retval}
def vkSetDeviceMemoryPriorityEXT(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "memory" in indict.keys():
         memory = indict["memory"]
    else: 
         memory = VkDeviceMemory_T()
    if "priority" in indict.keys():
         priority = indict["priority"]
    else: 
         priority = c_float()
    print(jvulkanLib.vkSetDeviceMemoryPriorityEXT)
    retval = jvulkanLib.vkSetDeviceMemoryPriorityEXT(device, memory, priority)
    return {"device" : device,"memory" : memory,"priority" : priority,"retval" : retval}
def vkGetDescriptorSetLayoutHostMappingInfoVALVE(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pBindingReference" in indict.keys():
         pBindingReference = indict["pBindingReference"]
    else: 
         pBindingReference = VkDescriptorSetBindingReferenceVALVE()
    if "pHostMapping" in indict.keys():
         pHostMapping = indict["pHostMapping"]
    else: 
         pHostMapping = VkDescriptorSetLayoutHostMappingInfoVALVE()
    print(jvulkanLib.vkGetDescriptorSetLayoutHostMappingInfoVALVE)
    retval = jvulkanLib.vkGetDescriptorSetLayoutHostMappingInfoVALVE(device, pBindingReference, pHostMapping)
    return {"device" : device,"pBindingReference" : pBindingReference,"pHostMapping" : pHostMapping,"retval" : retval}
def vkGetDescriptorSetHostMappingVALVE(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "descriptorSet" in indict.keys():
         descriptorSet = indict["descriptorSet"]
    else: 
         descriptorSet = VkDescriptorSet_T()
    if "ppData" in indict.keys():
         ppData = indict["ppData"]
    else: 
         ppData = POINTER(c_void_p)()
    print(jvulkanLib.vkGetDescriptorSetHostMappingVALVE)
    retval = jvulkanLib.vkGetDescriptorSetHostMappingVALVE(device, descriptorSet, ppData)
    return {"device" : device,"descriptorSet" : descriptorSet,"ppData" : ppData,"retval" : retval}
def vkCreateAccelerationStructureKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pCreateInfo" in indict.keys():
         pCreateInfo = indict["pCreateInfo"]
    else: 
         pCreateInfo = VkAccelerationStructureCreateInfoKHR()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pAccelerationStructure" in indict.keys():
         pAccelerationStructure = indict["pAccelerationStructure"]
    else: 
         pAccelerationStructure = pointer(VkAccelerationStructureKHR_T())
    print(jvulkanLib.vkCreateAccelerationStructureKHR)
    retval = jvulkanLib.vkCreateAccelerationStructureKHR(device, pCreateInfo, pAllocator, pAccelerationStructure)
    return {"device" : device,"pCreateInfo" : pCreateInfo,"pAllocator" : pAllocator,"pAccelerationStructure" : pAccelerationStructure,"retval" : retval}
def vkDestroyAccelerationStructureKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "accelerationStructure" in indict.keys():
         accelerationStructure = indict["accelerationStructure"]
    else: 
         accelerationStructure = VkAccelerationStructureKHR_T()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    print(jvulkanLib.vkDestroyAccelerationStructureKHR)
    retval = jvulkanLib.vkDestroyAccelerationStructureKHR(device, accelerationStructure, pAllocator)
    return {"device" : device,"accelerationStructure" : accelerationStructure,"pAllocator" : pAllocator,"retval" : retval}
def vkCmdBuildAccelerationStructuresKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "infoCount" in indict.keys():
         infoCount = indict["infoCount"]
    else: 
         infoCount = c_uint()
    if "pInfos" in indict.keys():
         pInfos = indict["pInfos"]
    else: 
         pInfos = VkAccelerationStructureBuildGeometryInfoKHR()
    if "ppBuildRangeInfos" in indict.keys():
         ppBuildRangeInfos = indict["ppBuildRangeInfos"]
    else: 
         ppBuildRangeInfos = VkAccelerationStructureBuildRangeInfoKHR()
    print(jvulkanLib.vkCmdBuildAccelerationStructuresKHR)
    retval = jvulkanLib.vkCmdBuildAccelerationStructuresKHR(commandBuffer, infoCount, pInfos, ppBuildRangeInfos)
    return {"commandBuffer" : commandBuffer,"infoCount" : infoCount,"pInfos" : pInfos,"ppBuildRangeInfos" : ppBuildRangeInfos,"retval" : retval}
def vkCmdBuildAccelerationStructuresIndirectKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "infoCount" in indict.keys():
         infoCount = indict["infoCount"]
    else: 
         infoCount = c_uint()
    if "pInfos" in indict.keys():
         pInfos = indict["pInfos"]
    else: 
         pInfos = VkAccelerationStructureBuildGeometryInfoKHR()
    if "pIndirectDeviceAddresses" in indict.keys():
         pIndirectDeviceAddresses = indict["pIndirectDeviceAddresses"]
    else: 
         pIndirectDeviceAddresses = pointer(c_ulong())
    if "pIndirectStrides" in indict.keys():
         pIndirectStrides = indict["pIndirectStrides"]
    else: 
         pIndirectStrides = pointer(c_uint())
    if "ppMaxPrimitiveCounts" in indict.keys():
         ppMaxPrimitiveCounts = indict["ppMaxPrimitiveCounts"]
    else: 
         ppMaxPrimitiveCounts = pointer(c_uint())
    print(jvulkanLib.vkCmdBuildAccelerationStructuresIndirectKHR)
    retval = jvulkanLib.vkCmdBuildAccelerationStructuresIndirectKHR(commandBuffer, infoCount, pInfos, pIndirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts)
    return {"commandBuffer" : commandBuffer,"infoCount" : infoCount,"pInfos" : pInfos,"pIndirectDeviceAddresses" : pIndirectDeviceAddresses,"pIndirectStrides" : pIndirectStrides,"ppMaxPrimitiveCounts" : ppMaxPrimitiveCounts,"retval" : retval}
def vkBuildAccelerationStructuresKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "deferredOperation" in indict.keys():
         deferredOperation = indict["deferredOperation"]
    else: 
         deferredOperation = VkDeferredOperationKHR_T()
    if "infoCount" in indict.keys():
         infoCount = indict["infoCount"]
    else: 
         infoCount = c_uint()
    if "pInfos" in indict.keys():
         pInfos = indict["pInfos"]
    else: 
         pInfos = VkAccelerationStructureBuildGeometryInfoKHR()
    if "ppBuildRangeInfos" in indict.keys():
         ppBuildRangeInfos = indict["ppBuildRangeInfos"]
    else: 
         ppBuildRangeInfos = VkAccelerationStructureBuildRangeInfoKHR()
    print(jvulkanLib.vkBuildAccelerationStructuresKHR)
    retval = jvulkanLib.vkBuildAccelerationStructuresKHR(device, deferredOperation, infoCount, pInfos, ppBuildRangeInfos)
    return {"device" : device,"deferredOperation" : deferredOperation,"infoCount" : infoCount,"pInfos" : pInfos,"ppBuildRangeInfos" : ppBuildRangeInfos,"retval" : retval}
def vkCopyAccelerationStructureKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "deferredOperation" in indict.keys():
         deferredOperation = indict["deferredOperation"]
    else: 
         deferredOperation = VkDeferredOperationKHR_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkCopyAccelerationStructureInfoKHR()
    print(jvulkanLib.vkCopyAccelerationStructureKHR)
    retval = jvulkanLib.vkCopyAccelerationStructureKHR(device, deferredOperation, pInfo)
    return {"device" : device,"deferredOperation" : deferredOperation,"pInfo" : pInfo,"retval" : retval}
def vkCopyAccelerationStructureToMemoryKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "deferredOperation" in indict.keys():
         deferredOperation = indict["deferredOperation"]
    else: 
         deferredOperation = VkDeferredOperationKHR_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkCopyAccelerationStructureToMemoryInfoKHR()
    print(jvulkanLib.vkCopyAccelerationStructureToMemoryKHR)
    retval = jvulkanLib.vkCopyAccelerationStructureToMemoryKHR(device, deferredOperation, pInfo)
    return {"device" : device,"deferredOperation" : deferredOperation,"pInfo" : pInfo,"retval" : retval}
def vkCopyMemoryToAccelerationStructureKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "deferredOperation" in indict.keys():
         deferredOperation = indict["deferredOperation"]
    else: 
         deferredOperation = VkDeferredOperationKHR_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkCopyMemoryToAccelerationStructureInfoKHR()
    print(jvulkanLib.vkCopyMemoryToAccelerationStructureKHR)
    retval = jvulkanLib.vkCopyMemoryToAccelerationStructureKHR(device, deferredOperation, pInfo)
    return {"device" : device,"deferredOperation" : deferredOperation,"pInfo" : pInfo,"retval" : retval}
def vkWriteAccelerationStructuresPropertiesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "accelerationStructureCount" in indict.keys():
         accelerationStructureCount = indict["accelerationStructureCount"]
    else: 
         accelerationStructureCount = c_uint()
    if "pAccelerationStructures" in indict.keys():
         pAccelerationStructures = indict["pAccelerationStructures"]
    else: 
         pAccelerationStructures = pointer(VkAccelerationStructureKHR_T())
    if "queryType" in indict.keys():
         queryType = indict["queryType"]
    else: 
         queryType = c_int()
    if "dataSize" in indict.keys():
         dataSize = indict["dataSize"]
    else: 
         dataSize = c_ulong()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    if "stride" in indict.keys():
         stride = indict["stride"]
    else: 
         stride = c_ulong()
    print(jvulkanLib.vkWriteAccelerationStructuresPropertiesKHR)
    retval = jvulkanLib.vkWriteAccelerationStructuresPropertiesKHR(device, accelerationStructureCount, pAccelerationStructures, queryType, dataSize, pData, stride)
    return {"device" : device,"accelerationStructureCount" : accelerationStructureCount,"pAccelerationStructures" : pAccelerationStructures,"queryType" : queryType,"dataSize" : dataSize,"pData" : pData,"stride" : stride,"retval" : retval}
def vkCmdCopyAccelerationStructureKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkCopyAccelerationStructureInfoKHR()
    print(jvulkanLib.vkCmdCopyAccelerationStructureKHR)
    retval = jvulkanLib.vkCmdCopyAccelerationStructureKHR(commandBuffer, pInfo)
    return {"commandBuffer" : commandBuffer,"pInfo" : pInfo,"retval" : retval}
def vkCmdCopyAccelerationStructureToMemoryKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkCopyAccelerationStructureToMemoryInfoKHR()
    print(jvulkanLib.vkCmdCopyAccelerationStructureToMemoryKHR)
    retval = jvulkanLib.vkCmdCopyAccelerationStructureToMemoryKHR(commandBuffer, pInfo)
    return {"commandBuffer" : commandBuffer,"pInfo" : pInfo,"retval" : retval}
def vkCmdCopyMemoryToAccelerationStructureKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkCopyMemoryToAccelerationStructureInfoKHR()
    print(jvulkanLib.vkCmdCopyMemoryToAccelerationStructureKHR)
    retval = jvulkanLib.vkCmdCopyMemoryToAccelerationStructureKHR(commandBuffer, pInfo)
    return {"commandBuffer" : commandBuffer,"pInfo" : pInfo,"retval" : retval}
def vkGetAccelerationStructureDeviceAddressKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pInfo" in indict.keys():
         pInfo = indict["pInfo"]
    else: 
         pInfo = VkAccelerationStructureDeviceAddressInfoKHR()
    print(jvulkanLib.vkGetAccelerationStructureDeviceAddressKHR)
    retval = jvulkanLib.vkGetAccelerationStructureDeviceAddressKHR(device, pInfo)
    return {"device" : device,"pInfo" : pInfo,"retval" : retval}
def vkCmdWriteAccelerationStructuresPropertiesKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "accelerationStructureCount" in indict.keys():
         accelerationStructureCount = indict["accelerationStructureCount"]
    else: 
         accelerationStructureCount = c_uint()
    if "pAccelerationStructures" in indict.keys():
         pAccelerationStructures = indict["pAccelerationStructures"]
    else: 
         pAccelerationStructures = pointer(VkAccelerationStructureKHR_T())
    if "queryType" in indict.keys():
         queryType = indict["queryType"]
    else: 
         queryType = c_int()
    if "queryPool" in indict.keys():
         queryPool = indict["queryPool"]
    else: 
         queryPool = VkQueryPool_T()
    if "firstQuery" in indict.keys():
         firstQuery = indict["firstQuery"]
    else: 
         firstQuery = c_uint()
    print(jvulkanLib.vkCmdWriteAccelerationStructuresPropertiesKHR)
    retval = jvulkanLib.vkCmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, accelerationStructureCount, pAccelerationStructures, queryType, queryPool, firstQuery)
    return {"commandBuffer" : commandBuffer,"accelerationStructureCount" : accelerationStructureCount,"pAccelerationStructures" : pAccelerationStructures,"queryType" : queryType,"queryPool" : queryPool,"firstQuery" : firstQuery,"retval" : retval}
def vkGetDeviceAccelerationStructureCompatibilityKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pVersionInfo" in indict.keys():
         pVersionInfo = indict["pVersionInfo"]
    else: 
         pVersionInfo = VkAccelerationStructureVersionInfoKHR()
    if "pCompatibility" in indict.keys():
         pCompatibility = indict["pCompatibility"]
    else: 
         pCompatibility = pointer(c_int())
    print(jvulkanLib.vkGetDeviceAccelerationStructureCompatibilityKHR)
    retval = jvulkanLib.vkGetDeviceAccelerationStructureCompatibilityKHR(device, pVersionInfo, pCompatibility)
    return {"device" : device,"pVersionInfo" : pVersionInfo,"pCompatibility" : pCompatibility,"retval" : retval}
def vkGetAccelerationStructureBuildSizesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "buildType" in indict.keys():
         buildType = indict["buildType"]
    else: 
         buildType = c_int()
    if "pBuildInfo" in indict.keys():
         pBuildInfo = indict["pBuildInfo"]
    else: 
         pBuildInfo = VkAccelerationStructureBuildGeometryInfoKHR()
    if "pMaxPrimitiveCounts" in indict.keys():
         pMaxPrimitiveCounts = indict["pMaxPrimitiveCounts"]
    else: 
         pMaxPrimitiveCounts = pointer(c_uint())
    if "pSizeInfo" in indict.keys():
         pSizeInfo = indict["pSizeInfo"]
    else: 
         pSizeInfo = VkAccelerationStructureBuildSizesInfoKHR()
    print(jvulkanLib.vkGetAccelerationStructureBuildSizesKHR)
    retval = jvulkanLib.vkGetAccelerationStructureBuildSizesKHR(device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo)
    return {"device" : device,"buildType" : buildType,"pBuildInfo" : pBuildInfo,"pMaxPrimitiveCounts" : pMaxPrimitiveCounts,"pSizeInfo" : pSizeInfo,"retval" : retval}
def vkCmdTraceRaysKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pRaygenShaderBindingTable" in indict.keys():
         pRaygenShaderBindingTable = indict["pRaygenShaderBindingTable"]
    else: 
         pRaygenShaderBindingTable = VkStridedDeviceAddressRegionKHR()
    if "pMissShaderBindingTable" in indict.keys():
         pMissShaderBindingTable = indict["pMissShaderBindingTable"]
    else: 
         pMissShaderBindingTable = VkStridedDeviceAddressRegionKHR()
    if "pHitShaderBindingTable" in indict.keys():
         pHitShaderBindingTable = indict["pHitShaderBindingTable"]
    else: 
         pHitShaderBindingTable = VkStridedDeviceAddressRegionKHR()
    if "pCallableShaderBindingTable" in indict.keys():
         pCallableShaderBindingTable = indict["pCallableShaderBindingTable"]
    else: 
         pCallableShaderBindingTable = VkStridedDeviceAddressRegionKHR()
    if "width" in indict.keys():
         width = indict["width"]
    else: 
         width = c_uint()
    if "height" in indict.keys():
         height = indict["height"]
    else: 
         height = c_uint()
    if "depth" in indict.keys():
         depth = indict["depth"]
    else: 
         depth = c_uint()
    print(jvulkanLib.vkCmdTraceRaysKHR)
    retval = jvulkanLib.vkCmdTraceRaysKHR(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable, width, height, depth)
    return {"commandBuffer" : commandBuffer,"pRaygenShaderBindingTable" : pRaygenShaderBindingTable,"pMissShaderBindingTable" : pMissShaderBindingTable,"pHitShaderBindingTable" : pHitShaderBindingTable,"pCallableShaderBindingTable" : pCallableShaderBindingTable,"width" : width,"height" : height,"depth" : depth,"retval" : retval}
def vkCreateRayTracingPipelinesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "deferredOperation" in indict.keys():
         deferredOperation = indict["deferredOperation"]
    else: 
         deferredOperation = VkDeferredOperationKHR_T()
    if "pipelineCache" in indict.keys():
         pipelineCache = indict["pipelineCache"]
    else: 
         pipelineCache = VkPipelineCache_T()
    if "createInfoCount" in indict.keys():
         createInfoCount = indict["createInfoCount"]
    else: 
         createInfoCount = c_uint()
    if "pCreateInfos" in indict.keys():
         pCreateInfos = indict["pCreateInfos"]
    else: 
         pCreateInfos = VkRayTracingPipelineCreateInfoKHR()
    if "pAllocator" in indict.keys():
         pAllocator = indict["pAllocator"]
    else: 
         pAllocator = VkAllocationCallbacks()
    if "pPipelines" in indict.keys():
         pPipelines = indict["pPipelines"]
    else: 
         pPipelines = pointer(VkPipeline_T())
    print(jvulkanLib.vkCreateRayTracingPipelinesKHR)
    retval = jvulkanLib.vkCreateRayTracingPipelinesKHR(device, deferredOperation, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines)
    return {"device" : device,"deferredOperation" : deferredOperation,"pipelineCache" : pipelineCache,"createInfoCount" : createInfoCount,"pCreateInfos" : pCreateInfos,"pAllocator" : pAllocator,"pPipelines" : pPipelines,"retval" : retval}
def vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipeline" in indict.keys():
         pipeline = indict["pipeline"]
    else: 
         pipeline = VkPipeline_T()
    if "firstGroup" in indict.keys():
         firstGroup = indict["firstGroup"]
    else: 
         firstGroup = c_uint()
    if "groupCount" in indict.keys():
         groupCount = indict["groupCount"]
    else: 
         groupCount = c_uint()
    if "dataSize" in indict.keys():
         dataSize = indict["dataSize"]
    else: 
         dataSize = c_ulong()
    if "pData" in indict.keys():
         pData = indict["pData"]
    else: 
         pData = c_void_p()
    print(jvulkanLib.vkGetRayTracingCaptureReplayShaderGroupHandlesKHR)
    retval = jvulkanLib.vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData)
    return {"device" : device,"pipeline" : pipeline,"firstGroup" : firstGroup,"groupCount" : groupCount,"dataSize" : dataSize,"pData" : pData,"retval" : retval}
def vkCmdTraceRaysIndirectKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pRaygenShaderBindingTable" in indict.keys():
         pRaygenShaderBindingTable = indict["pRaygenShaderBindingTable"]
    else: 
         pRaygenShaderBindingTable = VkStridedDeviceAddressRegionKHR()
    if "pMissShaderBindingTable" in indict.keys():
         pMissShaderBindingTable = indict["pMissShaderBindingTable"]
    else: 
         pMissShaderBindingTable = VkStridedDeviceAddressRegionKHR()
    if "pHitShaderBindingTable" in indict.keys():
         pHitShaderBindingTable = indict["pHitShaderBindingTable"]
    else: 
         pHitShaderBindingTable = VkStridedDeviceAddressRegionKHR()
    if "pCallableShaderBindingTable" in indict.keys():
         pCallableShaderBindingTable = indict["pCallableShaderBindingTable"]
    else: 
         pCallableShaderBindingTable = VkStridedDeviceAddressRegionKHR()
    if "indirectDeviceAddress" in indict.keys():
         indirectDeviceAddress = indict["indirectDeviceAddress"]
    else: 
         indirectDeviceAddress = c_ulong()
    print(jvulkanLib.vkCmdTraceRaysIndirectKHR)
    retval = jvulkanLib.vkCmdTraceRaysIndirectKHR(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable, indirectDeviceAddress)
    return {"commandBuffer" : commandBuffer,"pRaygenShaderBindingTable" : pRaygenShaderBindingTable,"pMissShaderBindingTable" : pMissShaderBindingTable,"pHitShaderBindingTable" : pHitShaderBindingTable,"pCallableShaderBindingTable" : pCallableShaderBindingTable,"indirectDeviceAddress" : indirectDeviceAddress,"retval" : retval}
def vkGetRayTracingShaderGroupStackSizeKHR(indict):
    if "device" in indict.keys():
         device = indict["device"]
    else: 
         device = VkDevice_T()
    if "pipeline" in indict.keys():
         pipeline = indict["pipeline"]
    else: 
         pipeline = VkPipeline_T()
    if "group" in indict.keys():
         group = indict["group"]
    else: 
         group = c_uint()
    if "groupShader" in indict.keys():
         groupShader = indict["groupShader"]
    else: 
         groupShader = c_int()
    print(jvulkanLib.vkGetRayTracingShaderGroupStackSizeKHR)
    retval = jvulkanLib.vkGetRayTracingShaderGroupStackSizeKHR(device, pipeline, group, groupShader)
    return {"device" : device,"pipeline" : pipeline,"group" : group,"groupShader" : groupShader,"retval" : retval}
def vkCmdSetRayTracingPipelineStackSizeKHR(indict):
    if "commandBuffer" in indict.keys():
         commandBuffer = indict["commandBuffer"]
    else: 
         commandBuffer = VkCommandBuffer_T()
    if "pipelineStackSize" in indict.keys():
         pipelineStackSize = indict["pipelineStackSize"]
    else: 
         pipelineStackSize = c_uint()
    print(jvulkanLib.vkCmdSetRayTracingPipelineStackSizeKHR)
    retval = jvulkanLib.vkCmdSetRayTracingPipelineStackSizeKHR(commandBuffer, pipelineStackSize)
    return {"commandBuffer" : commandBuffer,"pipelineStackSize" : pipelineStackSize,"retval" : retval}
