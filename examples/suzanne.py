#!/bin/env python
import ctypes
import os
import time
import sys
import json
import open3d as o3d
import numpy as np
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
setupDictPath = os.path.join("layouts", "standard_raster.json")
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

clock = time.perf_counter

# read the resources 
print("Testing IO for textured meshes ...")
textured_mesh = o3d.io.read_triangle_mesh("resources/suzanne.obj")
print(textured_mesh)
print(np.asarray(textured_mesh.vertices))
print(np.asarray(textured_mesh.triangles))


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
	
	# draw the frame!
	rasterPipeline.draw_frame()

# elegantly free all memory
instance_inst.release()
