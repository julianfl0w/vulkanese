#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
vkpath = "C:\\Users\\jcloi\\Documents\\vulkanese\\vulkanese"
sys.path.append(vkpath)
from vulkanese import *
here = os.path.dirname(os.path.abspath(__file__))
print(sys.path)
#from vulkanese.vulkanese import *

# device selection and instantiation
instance_inst = Instance()
print("available Devices:")
for i, d in enumerate(instance_inst.getDeviceList()):
	print("    " + str(i) + ": " + d.deviceName)
print("")

# choose a device
print("naively choosing device 0")
device = instance_inst.getDevice(0)

# read the setup dictionary
setupDictPath = os.path.join("layouts", "hello_triangle.json")
with open(setupDictPath, 'r') as f:
	setupDict = json.loads(f.read())

# apply setup to device
print("Applying the following layout:")
print(json.dumps(setupDict, indent = 4))
pipelines = device.applyLayout(setupDict)
print("")

# print the object hierarchy
print("Object tree:")
print(json.dumps(device.asDict(), indent=4))
rasterPipeline = pipelines[0]

verticesPos = \
np.array([[0.0, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]], dtype=np.dtype('f4'))
#np.array([[-0.5, -0.0], [0.0, 0.5], [0.5, 0.0]], dtype=np.dtype('f4'))	

verticesColor = \
np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.dtype('f4'))

clock = time.perf_counter 

# Main loop
last_time = clock() * 1000
fps = 0
running = True
while running:
	# timing
	fps += 1
	if clock() * 1000 - last_time >= 1000:
		last_time = clock() * 1000
		print("FPS: %s" % fps)
		fps = 0

	# get quit, mouse, keypress etc
	for event in rasterPipeline.surface.getEvents():
		if event.type == sdl2.SDL_QUIT:
			running = False
			vkDeviceWaitIdle(device.vkDevice)
			break
	
	vp = verticesPos.flatten()
	vp += 0.0001
	rasterPipeline.setBuffer("vertex", "POSITION", vp)
	vp = verticesColor.flatten()
	rasterPipeline.setBuffer("vertex", "COLOR", vp)
	# draw the frame!
	rasterPipeline.draw_frame()

# elegantly free all memory
instance_inst.release()
