#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import cv2
import math

here = os.path.dirname(os.path.abspath(__file__))
print(sys.path)

sys.path.append(os.path.join(here, "..", ".."))
sys.path.append(os.path.join(here, "..", "..", "..", "sinode"))
import sinode.sinode as sinode
import vulkanese as ve
import vulkan as vk

# from vulkanese.vulkanese import *


#######################################################


class Mandlebrot(ve.shader.Shader):
    def __init__(
        self,
        WIDTH=3200,  # Size of rendered mandelbrot set.
        HEIGHT=2400,  # Size of renderered mandelbrot set.
        **kwargs
    ):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.imageData = ve.buffer.StorageBuffer(
            device=device,
            name="imageData",
            qualifier="writeonly",
            memtype="uint",
            format="VK_FORMAT_R8G8B8A8_UINT",
            shape=[WIDTH * HEIGHT],
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )

        self.paramsBuffer = ve.buffer.StorageBuffer(
            device=device,
            name="paramsBuffer",
            qualifier="readonly",
            memtype="float",
            shape=[16],
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )

        self.proc_kwargs(**kwargs)

        self.setDefaults(
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            device=device,
            mandleStride=0.001,
            originX = 0,
            originY = 0,
            name="mandlebrot",
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            sourceFilename=os.path.join(here, "mandlebrot.template.comp"),
            buffers=[self.imageData, self.paramsBuffer],
            constantsDict=dict(
                HEIGHT=HEIGHT,
                WIDTH=WIDTH,
                WORKGROUP_SIZE=32,  # Workgroup size in compute shader.
            ),
            workgroupCount=[
                math.ceil(float(WIDTH) / 32),
                math.ceil(float(HEIGHT) / 32),
                1,
            ],
        )

        ve.shader.Shader.__init__(self, device=device)
        self.finalize()

        # print the object hierarchy
        print("Object tree:")
        print(json.dumps(device.asDict(), indent=4))

    def getImage(self):
        pa = np.frombuffer(self.imageData.pmap, np.uint8)
        pa = pa.reshape((self.HEIGHT, self.WIDTH, 4))
        # pa = self.imageData.get()
        return pa

    def zoom(self, amount):
        xmin = self.originX
        ymin = self.originY
        xmax = self.originX + self.mandleStride*self.WIDTH
        ymax = self.originY + self.mandleStride*self.HEIGHT

        mandleWidth = self.mandleStride*self.WIDTH
        mandleHeight= self.mandleStride*self.HEIGHT
        self.mandleStride/=amount
        newMandleWidth = self.mandleStride*self.WIDTH
        newMandleHeight= self.mandleStride*self.HEIGHT

        self.originX += (mandleWidth-newMandleWidth)/2
        self.originY += (mandleHeight-newMandleHeight)/2
        
    def pan(self, x, y):
        self.originX += x*self.mandleStride
        self.originY += y*self.mandleStride

    def run(self):
        self.paramsBuffer.set(np.array([self.originX, self.originY, self.mandleStride, 0] + [0]*12))
        ve.shader.Shader.run(self)
        return self.getImage()


if __name__ == "__main__":
    # device selection and instantiation
    instance_inst = ve.instance.Instance(verbose=True)
    screen = ve.screen.FullScreen()
    print("available Devices:")
    # for i, d in enumerate(instance_inst.getDeviceList()):
    # 	print("    " + str(i) + ": " + d.deviceName)
    print("")

    # choose a device
    print("naively choosing device 0")
    device = instance_inst.getDevice(0)
    mandle = Mandlebrot(
        device=device,
        instance=instance_inst,
        parent=device,
        WIDTH=screen.display.width,
        HEIGHT=screen.display.height,
    )
    
    while 1:
        img = mandle.run()
        key = screen.write(img)
        if key == ord("q"):
            break
        elif key == -1:
            pass
        elif key == ord("a"):
            mandle.zoom(1.1)
        elif key == ord("s"):
            mandle.zoom(1/1.1)
        elif key == 82: #up
            mandle.pan(0, 5)
        elif key == 84: #down
            mandle.pan(0, -5)
        elif key == 83: #right
            mandle.pan(5, 0)
        elif key == 81: #left
            mandle.pan(-5, 0)
        else:
            print(key)

    # elegantly free all memory
    instance_inst.release()
