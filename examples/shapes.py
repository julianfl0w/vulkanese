import open3d as o3d
import cv2 as cv
import numpy as np
import copy


class Pyramid:
    def __init__(self):

        # create a pyramid
        # first, create a single triangle
        BREDTH = 0.7
        HEIGHT = 0.7
        unitTri = [
            [0.0, -HEIGHT, 0.0],
            [-BREDTH / 2, 0.0, BREDTH / 2],
            [BREDTH / 2, 0.0, BREDTH / 2],
        ]
        verticesPos = np.array(unitTri)
        verticesIndex = [[0, 1, 2]]

        self.mesh = o3d.geometry.TriangleMesh()

        triangle = o3d.geometry.TriangleMesh()
        triangle.vertices = o3d.utility.Vector3dVector(verticesPos)
        triangle.triangles = o3d.utility.Vector3iVector(verticesIndex)

        newT = copy.deepcopy(triangle)
        R = newT.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
        # create 3 more faces
        for i in range(4):
            self.mesh += newT
            newT.rotate(R, center=(0, 0, 0))

        self.verticesColorBGR = np.array(
            [
                [[1.0, 0.0, 0.0]] * 3
                + [[1.0, 1.0, 0.0]] * 3
                + [[0.0, 0.0, 1.0]] * 3
                + [[0.0, 1.0, 1.0]] * 3
            ],
            dtype=np.float32,
        )
        self.verticesColorHSV = cv.cvtColor(self.verticesColorBGR, cv.COLOR_BGR2HSV)

        # self.verticesColorHSV[:, :, 0] = np.fmod(
        #    verticesColorHSV[:, :, 0] + 0.01, 360
        # )

        self.TRANSLATION = (0.0, 0.5, 0.5)
        self.mesh.translate(self.TRANSLATION)
        print(np.asarray(self.mesh.vertices))

    def rotate(self, fps_last):
        # rotate the pyrimid
        R = self.mesh.get_rotation_matrix_from_xyz(
            (0, -np.pi / max(6 * fps_last, 1), 0)
        )
        self.mesh.rotate(R, center=(0, 0, self.TRANSLATION[2]))
        # print(np.asarray(self.mesh.triangles, dtype="u4"))
