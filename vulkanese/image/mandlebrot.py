#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import cv2 as cv
import copy
from PIL import Image
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
        imageData = ve.buffer.StorageBuffer(
            device=device,
            name="imageData",
            type="Pixel",
            qualifier="writeonly",
            memtype="vec4",
            shape=[WIDTH * HEIGHT],
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            location=0,
        )

        self.proc_kwargs(**kwargs)

        self.setDefaults(
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            device=device,
            name="mandlebrot",
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            existingBuffers=[imageData],
            sourceFilename=os.path.join(here, "mandlebrot.template.comp"),
            buffers=[imageData],
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

        self.run()

        pa = np.frombuffer(imageData.pmap, np.float32)
        pa = pa.reshape((HEIGHT, WIDTH, 4))
        pa *= 255

        # Now we save the acquired color data to a .png.
        image = Image.fromarray(pa.astype(np.uint8))
        image.save("mandelbrot.png")


if __name__ == "__main__":
    # device selection and instantiation
    instance_inst = ve.instance.Instance(verbose=True)
    print("available Devices:")
    # for i, d in enumerate(instance_inst.getDeviceList()):
    # 	print("    " + str(i) + ": " + d.deviceName)
    print("")

    # choose a device
    print("naively choosing device 0")
    device = instance_inst.getDevice(0)
    mandle = Mandlebrot(device=device, instance=instance_inst, parent=device)
    mandle.run()

    # elegantly free all memory
    instance_inst.release()
