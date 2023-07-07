import pinocchio as pin
import numpy as np
import sys
import os
from os.path import dirname, join, abspath

from pinocchio.visualize import MeshcatVisualizer
from wrapper_meshcat import MeshcatWrapper

# Load the URDF model.
# Conversion with str seems to be necessary when executing this file with ipython
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "franka2.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

q0 = pin.neutral(model)


jointsToLock = [
    "root_joint:",
    "panda2_finger_joint1:",
    "panda2_finger_joint2:",
    "universe",
]


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


MeshcatVis = MeshcatWrapper()
vis = MeshcatVis.visualize(
    robot_model=model_reduced,
    robot_collision_model=collision_model_reduced,
    robot_visual_model=visual_model_reduced,
)


q0 = pin.neutral(model_reduced)
vis.display(q0)
vis.displayCollisions(True)
vis.displayVisuals(True)
