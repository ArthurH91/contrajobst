import sys
import numpy as np
import pinocchio as pin

import hppfcl

from pinocchio.visualize import MeshcatVisualizer

import meshcat.geometry as g
import meshcat.transformations as tf


RED = np.array([249, 136, 126, 125]) / 255
RED_FULL = np.array([249, 136, 126, 255]) / 255

GREEN = np.array([170, 236, 149, 125]) / 255
GREEN_FULL = np.array([170, 236, 149, 255]) / 255

BLUE = np.array([144, 169, 183, 125]) / 255
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

YELLOW = np.array([1, 1, 0, 0.5])
YELLOW_FULL = np.array([1, 1, 0, 1.0])

BLACK = np.array([0, 0, 0, 0.5])
BLACK_FULL = np.array([0, 0, 0, 1.0])


model = pin.Model()

geom_model = pin.GeometryModel()
geometries = [
    hppfcl.Capsule(0.1, 0.8),
    hppfcl.Box(1, 1, 1),
]

T1 = pin.SE3.Identity()
T2 = pin.SE3.Identity()


geom_obj_1 = pin.GeometryObject("obj{}".format(1), 0, 0, hppfcl.Capsule(0.1, 0.8), T1)
color = RED
geom_obj_1.meshColor = color
geom_model.addGeometryObject(geom_obj_1)

geom_obj_2 = pin.GeometryObject("obj{}".format(2), 0, 0, hppfcl.Box(1, 1, 1), T2)
color = YELLOW
geom_obj_2.meshColor = color
geom_model.addGeometryObject(geom_obj_2)

req = hppfcl.DistanceRequest()
req.enable_nearest_points = True
res = hppfcl.DistanceResult()
dist = hppfcl.distance(geometries[0], T1, geometries[1], T2, req, res)

r_w = 0.1

geom_np = hppfcl.Sphere(r_w)


cp1 = res.getNearestPoint1()
cp2 = res.getNearestPoint2()

print(cp1 - cp2)

cp1_T = pin.SE3(np.eye(3), cp1)
cp2_T = pin.SE3(np.eye(3), cp2)


geom_obj_3 = pin.GeometryObject("obj{}".format(3), 0, 0, geom_np, cp1_T)
color = GREEN_FULL
geom_obj_3.meshColor = color
geom_model.addGeometryObject(geom_obj_3)

geom_obj_4 = pin.GeometryObject("obj{}".format(4), 0, 0, geom_np, cp2_T)
color = BLUE_FULL
geom_obj_4.meshColor = color
geom_model.addGeometryObject(geom_obj_4)


# print(dist)

viz = MeshcatVisualizer(model, geom_model, geom_model)

# Initialize the viewer.
viz.initViewer(open=True)
viz.loadViewerModel("shapes")
viz.display(np.zeros(0))

input("press enter to continue")
