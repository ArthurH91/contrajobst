from os.path import dirname, join, abspath

import numpy as np
import pinocchio as pin

import matplotlib.pyplot as plt
import hppfcl

from wrapper_robot import RobotWrapper


###* LOADING THE ROBOT

pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "franka2.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)


# Creation of the robot

robot_wrapper = RobotWrapper(
    name_robot="franka",
    belong_to_example_robot_data=False,
    urdf_model_path=urdf_model_path,
    mesh_dir=mesh_dir,
)
rmodel, cmodel, vmodel = robot_wrapper()
rdata = rmodel.createData()
cdata = cmodel.createData()

# Initial configuration of the robot

INITIAL_CONFIG = pin.randomConfiguration(rmodel)

# Updating the pinocchio models
pin.framesForwardKinematics(rmodel, rdata, INITIAL_CONFIG)
pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

for oMg, geometry_objects in zip(cdata.oMg, cmodel.geometryObjects):
    if not isinstance(geometry_objects.geometry, hppfcl.Box):
        pin.computeJointJacobians(rmodel, rdata, INITIAL_CONFIG)

        # Getting the frame jacobian from the geometry object in the LOCAL reference frame
        jacobian = pin.computeFrameJacobian(
            rmodel,
            rdata,
            INITIAL_CONFIG,
            geometry_objects.parentFrame,
            pin.LOCAL,
        )
        print(
            f"norm = {np.linalg.norm(jacobian)}, type = {geometry_objects.geometry}, name = {geometry_objects.name}, geom_id : {cmodel.getGeometryId(geometry_objects.name)}, parent_frame = {geometry_objects.parentFrame} "
        )
