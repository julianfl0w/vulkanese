#!/bin/env python

# coding: utf-8

# flake8: noqa
import ctypes
import os
import time
import sys
import json
from vulkan_lib import *

here = os.path.dirname(os.path.abspath(__file__))

# device selection and instantiation
instance_inst = Instance()
print("available Devices:")
for i, d in enumerate(instance_inst.getDeviceList()):
	print("    " + str(i) + ": " + d.deviceName)
print("")

# load the CommandBuffer setup dictionary
setupDictPath = os.path.join(here, "layouts", "standard_raster.json")
with open(setupDictPath, 'r') as f:
	setupDict = json.loads(f.read())
	
# for now, only one commandPool per device
print("naively choosing device 0")
device      = instance_inst.getDevice(0)
commandPool = device.createCommandPool()

# As of now, SimpleVulkan has no settable options for
# Device or CommandPool class. setupDict describes 
# CommandBuffer and its children 
print("Applying the following CommandBuffer layout:")
print(json.dumps(setupDict, indent = 4))
commandBuffer = commandPool.createCommandBuffer(setupDict)

# Python peculiarities
if sys.version_info >= (3, 3):
	clock = time.perf_counter
else:
	clock = time.clock

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
	for event in commandBuffer.getEvents():
		if event.type == sdl2.SDL_QUIT:
			running = False
			vkDeviceWaitIdle(device.vkDevice)
			break
	
	# draw the frame!
	commandBuffer.draw_frame()

# elegantly free all memory
instance_inst.release()
