#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import trimesh
import cv2 as cv
import open3d as o3d
import copy
from exutils import *

here = os.path.dirname(os.path.abspath(__file__))
print(sys.path)

localtest = True
if localtest == True:
	vkpath = os.path.join(here, "..", "vulkanese")
	sys.path.append(vkpath)
	from vulkanese import *
else:
	from vulkanese.vulkanese import *


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
setupDictPath = os.path.join("resources", "standard_raytrace.json")
pipelines = device.applyLayoutFile(setupDictPath)

# print the object hierarchy
print("Object tree:")
print(json.dumps(device.asDict(), indent=4))
rasterPipeline = pipelines[0]

pyramidMesh = getPyramid()
TRANSLATION = (0.0, 0.5, 0.5)
pyramidMesh.translate(TRANSLATION)
pyramidVerticesColor = np.array([[[1.0, 0.0, 0.0]]*3 + [[1.0, 1.0, 0.0]]*3 + [[0.0, 0.0, 1.0]]*3 + [[0.0, 1.0, 1.0]]*3], dtype=np.dtype('f4'))
pyramidVerticesColorHSV = cv.cvtColor(pyramidVerticesColor, cv.COLOR_BGR2HSV)
print(np.asarray(pyramidMesh.vertices))

# Main loop
clock = time.perf_counter 
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
	
	R = pyramidMesh.get_rotation_matrix_from_xyz((0, -np.pi/20000, 0))
	pyramidMesh.rotate(R, center=(0,0,TRANSLATION[2]))
	meshVert = np.asarray(pyramidMesh.vertices, dtype = 'f4')
	#print(np.asarray(pyramidMesh.vertices).flatten())
	rasterPipeline.setBuffer("vertex", "INDEX", np.asarray(pyramidMesh.triangles, dtype='u2').flatten())

	rasterPipeline.setBuffer("vertex", "POSITION", meshVert)
	pyramidVerticesColorHSV[:,:,0] = np.fmod(pyramidVerticesColorHSV[:,:,0] + 0.01, 360)
	#pyramidVerticesColor =  cv.cvtColor(pyramidVerticesColorHSV, cv.COLOR_HSV2RGB)
	vp = pyramidVerticesColor.flatten()
	rasterPipeline.setBuffer("vertex", "COLOR", vp)
	# draw the frame!
	rasterPipeline.draw_frame()

# elegantly free all memory
instance_inst.release()
