# This examples shows how to load and move a robot in meshcat.
# Note: this feature requires Meshcat to be installed, this can be done using
# pip install --user meshcat

import pinocchio as pin
import numpy as np
import sys
import os
from os.path import dirname, join, abspath

from pinocchio.visualize import MeshcatVisualizer

# Load the URDF model.
# Conversion with str seems to be necessary when executing this file with ipython
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
# urdf_filename = "talos_reduced.urdf"
# urdf_model_path = join(join(model_path,"talos_data/robots"),urdf_filename)
urdf_filename = "franka2.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

q0 = pin.neutral(model)


mesh = visual_model.geometryObjects[0].geometry
print(mesh)
try:
    mesh.buildConvexRepresentation(True)
    convex = mesh.convex
except:
    convex = None

if convex is not None:
    placement = pin.SE3.Identity()
    placement.translation[0] = 2.0
    geometry = pin.GeometryObject("convex", 0, convex, placement)
    geometry.meshColor = np.ones((4))
    visual_model.addGeometryObject(geometry)

joints = []

for jn in model.joints:
    print(jn)
    joints.append(jn)
print("-" * 30)

jointsToLock = [
    "root_joint:",
    "panda2_finger_joint1:",
    "panda2_finger_joint2:",
    "universe",
]

# jointsToLockIDs = []
# for jn in jointsToLock:
#     if model.existJointName(jn):
#         jointsToLockIDs.append(model.getJointId(jn))
#     else:
#         print("Warning: joint " + str(jn) + " does not belong to the model!")

jointsToLockIDs = [1, 9, 10]


geom_models = [visual_model, collision_model]
model_reduced, geometric_models_reduced = pin.buildReducedModel(
    model,
    list_of_geom_models=geom_models,
    list_of_joints_to_lock=jointsToLockIDs,
    reference_configuration=q0,
)

visual_model_reduced, collision_model_reduced = (
    geometric_models_reduced[0],
    geometric_models_reduced[1],
)


viz = MeshcatVisualizer(model_reduced, collision_model_reduced, visual_model_reduced)

# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz.initViewer(open=False)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

# Load the robot in the viewer.
viz.loadViewerModel()

# Display a robot configuration.
q0 = pin.neutral(model_reduced)
viz.display(q0)
viz.displayCollisions(True)
viz.displayVisuals(True)

viz["obstacle"].set_transform(pin.SE3.Identity())
