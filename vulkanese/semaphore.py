from vulkan import *

class Semaphore:
    def __init__(self, device, flags=0):
        self.extant = True
        self.device = device
        self.semaphore_create = VkSemaphoreCreateInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, flags=flags
        )
        self.vkSemaphore = vkCreateSemaphore(
            self.device.vkDevice, self.semaphore_create, None
        )
        
    def release(self):
        if self.extant:
            vkDestroySemaphore(self.device.vkDevice, self.vkSemaphore, None)
            self.extant = False
            
class Fence:
    def __init__(self, device, flags=0):
        self.device = device
        self.extant = True
        self.fenceCreateInfo = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=flags
        )
        self.vkFence = vkCreateFence(self.device.vkDevice, self.fenceCreateInfo, None)
        
    def release(self):
        if self.extant:
            vkDestroyFence(self.device.vkDevice, self.vkFence, None)
            self.extant = False
            
    def wait(self):
        
        # The command will not have finished executing until the fence is signalled.
        # So we wait here.
        # We will directly after this read our buffer from the GPU,
        # and we will not be sure that the command has finished executing unless we wait for the fence.
        # Hence, we use a fence here.
        vkWaitForFences(
            device=self.device.vkDevice,
            fenceCount=1,
            pFences=[self.vkFence],
            waitAll=VK_TRUE,
            timeout=1000000000,
        )
        vkResetFences(device=self.device.vkDevice, fenceCount=1, pFences=[self.vkFence])
