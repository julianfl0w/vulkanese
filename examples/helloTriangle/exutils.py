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


def getPyramid():

    # create a pyramid
    # first, create a single triangle
    BREDTH = 0.7
    HEIGHT = 0.7
    unitTri = [
        [0.0, -HEIGHT, 0.0],
        [-BREDTH / 2, 0.0, BREDTH / 2],
        [BREDTH / 2, 0.0, BREDTH / 2],
    ]
    pyramidVerticesPos = np.array(unitTri)
    pyramidVerticesIndex = [[0, 1, 2]]

    pyramidMesh = o3d.geometry.TriangleMesh()
    pyramidMesh.vertices = o3d.utility.Vector3dVector(pyramidVerticesPos)
    pyramidMesh.triangles = o3d.utility.Vector3iVector(pyramidVerticesIndex)

    newT = copy.deepcopy(pyramidMesh)
    R = newT.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
    # create 3 more faces
    for i in range(3):
        newT.rotate(R, center=(0, 0, 0))
        pyramidMesh += newT

    return pyramidMesh
