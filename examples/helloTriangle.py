#!/bin/env python

# coding: utf-8

# flake8: noqa
import ctypes
import os
import time
import sys
import json
vkpath = "C:\\Users\\jcloi\\Documents\\vulkanese\\vulkanese"
sys.path.append(vkpath)
from vulkanese import *

print(sys.path)
#from vulkanese.vulkanese import *

here = os.path.dirname(os.path.abspath(__file__))
# device selection and instantiation
instance_inst = Instance()
print("available Devices:")
for i, d in enumerate(instance_inst.getDeviceList()):
	print("    " + str(i) + ": " + d.deviceName)
print("")

# load the CommandBuffer setup dictionary
setupDictPath = os.path.join(getVulkanesePath(), "layouts", "standard_raster.json")
with open(setupDictPath, 'r') as f:
	setupDict = json.loads(f.read())
	
# for now, only one commandPool per device
print("naively choosing device 0")
device      = instance_inst.getDevice(0)

# As of now, SimpleVulkan has no settable options for
# Device or CommandPool class. setupDict describes 
# CommandBuffer and its children 
print("Applying the following layout:")
print(json.dumps(setupDict, indent = 4))
pipelines = device.applyLayout(setupDict)

print("")
print("Object tree:")
print(device)
rasterPipeline = pipelines[0]

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
	
	# draw the frame!
	rasterPipeline.draw_frame()
	
	if fps > 5:
		break

# elegantly free all memory
instance_inst.release()
