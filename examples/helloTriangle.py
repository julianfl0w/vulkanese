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

LF = 0
TOP = 1
RF = 2
LB = 3
RB = 4
#                left front  top              right front      left back           right back
pyramidVerticesPos = \
np.array([[-0.5, 0.0,  0.5], [0.0, -1.0, 0.5], [ 0.5, 0.0, 0.5]], dtype=np.dtype('f4'))

pyramidVerticesColor = \
np.array([[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]], dtype=np.dtype('f4'))

pyramidVerticesIndex = [[LF,TOP,RF],[LF,TOP,LB],[RF,TOP,RB],[RB,TOP,LB]]


BREDTH = 0.7
HEIGHT = 0.7
TRANSLATION = (0.0, 0.5, 0.5)
unitTri  = [[0.0, -HEIGHT, 0.0], [-BREDTH/2, 0.0,  BREDTH/2], [ BREDTH/2, 0.0, BREDTH/2]]
#unitTri  = [[p+TRANSLATION[i] for i, p in enumerate(v)] for v in unitTri]
flatTri2 = [[-0.5, 0.0,  0.0], [0.0, -1.0, 0.0], [ 0.5, 0.0, 0.0]]
pyramidVerticesPos = \
np.array(unitTri)
print("PVP " + str(pyramidVerticesPos))

pyramidVerticesColor = \
np.array([[[1.0, 0.0, 0.0]]*3 + [[1.0, 1.0, 0.0]]*3 + [[0.0, 0.0, 1.0]]*3 + [[0.0, 1.0, 1.0]]*3], dtype=np.dtype('f4'))

pyramidVerticesIndex = [[0,1,2]]



#pyramidMesh objects can be created from existing faces and vertex data
#pyramidMesh = trimesh.Trimesh(vertices=pyramidVerticesPos, faces=[[0, 1, 2]])
pyramidMesh = o3d.geometry.TriangleMesh()

pyramidVerticesColorHSV = cv.cvtColor(pyramidVerticesColor, cv.COLOR_BGR2HSV)
print(pyramidVerticesColorHSV)
pyramidMesh.vertices = o3d.utility.Vector3dVector(pyramidVerticesPos)
pyramidMesh.triangles = o3d.utility.Vector3iVector(pyramidVerticesIndex)

newT = copy.deepcopy(pyramidMesh)
R = newT.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
for i in range(3):
	newT.rotate(R, center=(0,0,0))
	pyramidMesh += newT
	
pyramidMesh.translate(TRANSLATION)
print(np.asarray(pyramidMesh.vertices))

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
	
	#print(pyramidMesh.vertices)
	#print(pyramidMesh.vertices.dtype)
	
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
