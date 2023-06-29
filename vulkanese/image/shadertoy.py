#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import cv2
import math
import json 

here = os.path.dirname(os.path.abspath(__file__))
print(sys.path)

import requests

from dotenv import load_dotenv

sys.path.append(os.path.join(here, "..", ".."))
sys.path.append(os.path.join(here, "..", "..", "..", "sinode"))
import sinode.sinode as sinode
import vulkanese as ve
import vulkan as vk

class ShaderToy(ve.shader.Shader):
    def __init__(
        self,
        **kwargs
    ):
        self.proc_kwargs(**kwargs)
        self.WIDTH = int(self.WIDTH)
        self.HEIGHT = int(self.HEIGHT)
        print(self.shaderToyAPIKey)
        url = f"https://www.shadertoy.com/api/v1/shaders/{self.shaderID}?key={self.shaderToyAPIKey}"
        print(url)
        #response = requests.post(url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.post(url, headers=headers)

        print(response)
        json_data = json.loads(response.text)
        print(json.dumps(json_data, indent = 2))
        code = json_data["Shader"]["renderpass"][0]["code"]
        code = code.encode().decode('unicode_escape')
        print(code)

        
        self.setDefaults(
            shaderID = "ldlXRS",
            shaderToyAPIKey = "none",
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            shaderToyStride=0.001,
            originX = 0,
            originY = 0,
            name="ShaderToy",
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            sourceText=code,
            constantsDict=dict(
                HEIGHT=self.HEIGHT,
                WIDTH=self.WIDTH,
                WORKGROUP_SIZE=32,  # Workgroup size in compute shader.
            ),
            workgroupCount=[
                math.ceil(float(self.WIDTH) / 32),
                math.ceil(float(self.HEIGHT) / 32),
                1,
            ],
        )



        self.imageData = ve.buffer.StorageBuffer(
            device=self.device,
            name="imageData",
            qualifier="writeonly",
            memtype="uint",
            format="VK_FORMAT_R8G8B8A8_UINT",
            shape=[self.WIDTH * self.HEIGHT],
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )

        self.paramsBuffer = ve.buffer.StorageBuffer(
            device=self.device,
            name="paramsBuffer",
            qualifier="readonly",
            memtype="double",
            shape=[16],
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )

        self.buffers=[self.imageData, self.paramsBuffer]

        ve.shader.Shader.__init__(self, device=self.device)
        self.finalize()

        # print the object hierarchy
        print("Object tree:")
        print(json.dumps(self.device.asDict(), indent=4))

    def getImage(self):
        pa = np.frombuffer(self.imageData.pmap, np.uint8)
        pa = pa.reshape((self.HEIGHT, self.WIDTH, 4))
        # pa = self.imageData.get()
        return pa

    def dumpPos(self):
        print([self.originX, self.originY, self.shaderToyStride])

    def setCoords(self, coords):
        self.originX, self.originY, self.shaderToyStride = coords

    def zoom(self, amount):
        shaderToyWidth = self.shaderToyStride*self.WIDTH
        shaderToyHeight= self.shaderToyStride*self.HEIGHT
        self.shaderToyStride/=amount
        newshaderToyWidth = self.shaderToyStride*self.WIDTH
        newshaderToyHeight= self.shaderToyStride*self.HEIGHT

        self.originX += (shaderToyWidth-newshaderToyWidth)/2
        self.originY += (shaderToyHeight-newshaderToyHeight)/2

        self.dumpPos()
        
    def pan(self, x, y):
        self.originX += x*self.shaderToyStride
        self.originY += y*self.shaderToyStride
        self.dumpPos()

    def run(self):
        self.paramsBuffer.set(np.array([self.originX, self.originY, self.shaderToyStride, 0] + [0]*12))
        ve.shader.Shader.run(self)
        return self.getImage()

if __name__ == "__main__":  
    # device selection and instantiation
    instance_inst = ve.instance.Instance(verbose=True)
    screen = ve.screen.FullScreen()
    
    load_dotenv(os.path.join(here, ".env"))

    # choose a device
    print("naively choosing device 0")
    device = instance_inst.getDevice(0)
    shaderToy = ShaderToy(
        device=device,
        shaderID = "Wt33Wf",
        shaderToyAPIKey = os.getenv("shadertoy_appkey"),
        instance=instance_inst,
        parent=device,
        WIDTH=screen.display.width/2,
        HEIGHT=screen.display.height/2,
    )
    
    while 1:
        img = shaderToy.run()
        key = screen.write(img)
        speed = 20
        if key == ord("q"):
            break
        elif key == -1:
            pass
        elif key == ord("a"):
            shaderToy.zoom(1/1.1)
        elif key == ord("s"):
            shaderToy.zoom(1.1)
        elif key == 82: #up
            shaderToy.pan(0, -speed)
        elif key == 84: #down
            shaderToy.pan(0, speed)
        elif key == 83: #right
            shaderToy.pan(speed, 0)
        elif key == 81: #left
            shaderToy.pan(-speed, 0)
        else:
            print(key)

    # elegantly free all memory
    instance_inst.release()
