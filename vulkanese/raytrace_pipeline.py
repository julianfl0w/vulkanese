import ctypes
import os
import time
import json
from vulkan import *

from . import vulkanese
from . import pipeline

from . import sinode

# import faulthandler

# faulthandler.enable()

here = os.path.dirname(os.path.abspath(__file__))


def getVulkanesePath():
    return here


from enum import Enum


class StageIndices(Enum):
    eRaygen = 0
    eMiss = 1
    eMiss2 = 2
    eClosestHit = 3
    eShaderGroupCount = 4


class RaytracePipeline(pipeline.Pipeline):
    def __init__(self, device, setupDict):
        Pipeline.__init__(self, device, setupDict)

        self.stages = [s.shader_stage_create for s in self.stageDict.values()]

        # in raytracing, there may be alternative shaders
        # for example, an occlusion miss shader and a diffraction one
        # there are always 4 stages:
        #  Raygen
        #  Miss
        #  Hit
        #  Callable
        # Each of these has its own SBT, represented as a strided region
        missCount = 1
        hitCount = 1
        handleCount = 1 + missCount + hitCount

        self.SBTDict = {}
        baseSBT = {"stride": 0, "size": 0}
        self.SBTDict["gen"] = baseSBT.copy()
        self.SBTDict["callable"] = baseSBT.copy()
        self.SBTDict["miss"] = baseSBT.copy()
        self.SBTDict["hit"] = baseSBT.copy()
        self.SBTDict["gen"]["stage"] = VK_SHADER_STAGE_RAYGEN_BIT_KHR
        self.SBTDict["callable"]["stage"] = VK_SHADER_STAGE_CALLABLE_BIT_KHR
        self.SBTDict["miss"]["stage"] = VK_SHADER_STAGE_MISS_BIT_KHR
        self.SBTDict["hit"]["stage"] = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR

        for stage, stageDict in self.SBTDict.items():
            print("processing RayTrace stage " + stage)
            for shader in self.stageDict.values():
                if eval(shader.stage) & stageDict["stage"]:
                    # stageDict["size"  ] += shader.setupDict["SIZEBYTES"]
                    stageDict["size"] += 65536
                    stageDict["stride"] = max(stageDict["stride"], 65536)

            # Allocate a buffer for storing the SBT.
            bufferDict = {
                "usage": "VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR",
                "descriptorSet": "global",
                "rate": "VK_VERTEX_INPUT_RATE_VERTEX",
                "memProperties": "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT",
                # "sharingMode"    : "VK_SHARING_MODE_EXCLUSIVE",
                "sharingMode": "0",
                "SIZEBYTES": 6553600,
                "qualifier": "in",
                "type": "vec3",
                "format": "VK_FORMAT_R32G32B32_SFLOAT",
                # "stage"          : "VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR",
                "stage": "VK_SHADER_STAGE_RAYGEN_BIT_KHR",
                "stride": 1,
            }

            stageDict["buffer"] = Buffer(self.device, bufferDict)

            print("getting device address")

            vkGetBufferDeviceAddress = vkGetInstanceProcAddr(
                self.instance.vkInstance, "vkGetBufferDeviceAddressKHR"
            )
            print(self.vkDevice)
            print(stageDict["buffer"].bufferDeviceAddressInfo)
            deviceAddress = vkGetBufferDeviceAddress(
                self.vkDevice, stageDict["buffer"].bufferDeviceAddressInfo
            )

            print("creating strided region")
            stageDict["vkStridedDeviceAddressRegion"] = VkStridedDeviceAddressRegionKHR(
                deviceAddress=deviceAddress,
                stride=self.setupDict["stride"],
                size=self.setupDict["SIZEBYTES"],
            )

        # Get the shader group handles
        vkGetRayTracingShaderGroupHandlesKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkGetRayTracingShaderGroupHandlesKHR"
        )
        result = vkGetRayTracingShaderGroupHandlesKHR(
            self.vkDevice,
            self.pipeline.vkPipeline,
            0,
            handleCount,
            dataSize,
            handles.data(),
        )
        assert result == VK_SUCCESS

        # Shader groups
        # Intersection shaders allow arbitrary intersection geometry
        # For now they are unused. therefore VK_SHADER_UNUSED_KHR is appropriate
        self.shaderGroupCreateInfo = VkRayTracingShaderGroupCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            pNext=None,
            type=VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            generalShader=VK_SHADER_UNUSED_KHR,
            closestHitShader=VK_SHADER_UNUSED_KHR,
            anyHitShader=VK_SHADER_UNUSED_KHR,
            intersectionShader=VK_SHADER_UNUSED_KHR,
            pShaderGroupCaptureReplayHandle=None,
        )
        rtShaderGroups = [self.shaderGroupCreateInfo]

        # Push constant: we want to be able to update constants used by the shaders
        # pushConstant = vkPushConstantRange(
        # 	VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
        # 	0, sizeof(PushConstantRay)};
        # VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        # pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        # .pPushConstantRanges    = &pushConstant;

        # Assemble the shader stages and recursion depth info into the ray tracing pipeline
        # In this case, self.rtShaderGroups.size() == 4: we have one raygen group,
        # two miss shader groups, and one hit group.
        # The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
        # hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
        # as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
        # in the ray generation to avoid deep recursion.
        self.rayPipelineInfo = VkRayTracingPipelineCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            pNext=None,
            flags=0,
            stageCount=len(self.stages),  # Stages are shaders
            pStages=self.stages,
            groupCount=len(rtShaderGroups),
            pGroups=rtShaderGroups,
            maxPipelineRayRecursionDepth=2,  # Ray depth
            layout=self.pipelineLayout,
        )

        vkCreateRayTracingPipelinesKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkCreateRayTracingPipelinesKHR"
        )

        self.vkPipeline = vkCreateRayTracingPipelinesKHR(
            device=self.vkDevice,
            deferredOperation=None,
            pipelineCache=None,
            createInfoCount=1,
            pCreateInfos=[self.rayPipelineInfo],
            pAllocator=None,
        )

        # create the sbt after creating the pipeline
        self.sbt = ShaderBindingTable(self)

        # wrap it all up into a command buffer
        print("Creating commandBuffer")

        vkCmdBindPipeline(
            cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, self.rtPipeline
        )
        vkCmdBindDescriptorSets(
            cmdBuf,
            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            self.rtPipelineLayout,
            0,
            size(descSets),
            descSets,
            0,
            nullptr,
        )

        vkCmdTraceRaysKHR(
            cmdBuf,
            self.shaderDict["rgen"].vkStridedDeviceAddressRegion,
            self.shaderDict["miss"].vkStridedDeviceAddressRegion,
            self.shaderDict["hit"].vkStridedDeviceAddressRegion,
            self.shaderDict["call"],
            self.outputWidthPixels,
            self.outputHeightPixels,
            1,
        )
