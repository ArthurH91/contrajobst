import os
from os.path import dirname, join, abspath
import argparse
import json, codecs

import numpy as np
import pinocchio as pin
import hppfcl
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper

from utils import get_transform, linear_gradient, check_limits, check_auto_collisions, get_q_iter_from_Q

### BOOLS
DISPLAY = True
CHECK_LIMITS = False    

###* HYPERPARAMS
nq = 7

name = "results_250_q_box"


def display_traj(vis, Q_min, nq=7):
    for k in range(int(len(Q_min) / nq)):
        vis.display(np.array(Q_min[k * nq : (k + 1) * nq]))
        input()


# Openning the files
path = os.getcwd()

results_json = codecs.open(
    path + "/results/" + name + ".json", "r", encoding="utf-8"
).read()
# Loading the json file
results = json.loads(results_json)


q_dot = []
dist_min_obstacle = []
initial_cost = []
principal_cost = []
terminal_cost = []

obstacle_cost = []
grad = []
Q_trs_list = []

q_dot.append(results["q_dot"])
dist_min_obstacle.append(results["dist_min_obs"])
initial_cost.append(results["initial_cost"])
principal_cost.append(results["principal_cost"])
terminal_cost.append(results["terminal_cost"])
obstacle_cost.append(results["obstacle_cost"])
grad.append(np.linalg.norm(results["grad"]))
Q_trs_list.append(results["Q_trs"])
    
if __name__ == "__main__":

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


    if DISPLAY:
        # Generate a reachable target
        TARGET = pin.SE3.Identity()
        TARGET.translation = np.array([-0.1, -0.15, 0.86])

        # OBSTACLES POSES

        width = 1e-2
        height = 2e-1
        length = 5e-1

        dist_between_front_behind = 0.5

        # Generate a reachable obstacle
        OBSTACLE1_FRONT = TARGET.copy()
        OBSTACLE1_FRONT.translation = np.array([0.2, -0.1, 0.86])


        OBSTACLE2_BEHIND = OBSTACLE1_FRONT.copy()
        OBSTACLE2_BEHIND.translation += np.array([- dist_between_front_behind,0,0])

        OBSTACLE3_BOTTOM = OBSTACLE1_FRONT.copy()
        OBSTACLE4_RIGHT = OBSTACLE1_FRONT.copy()
        OBSTACLE5_LEFT = OBSTACLE1_FRONT.copy()


        OBSTACLE3_BOTTOM.translation += np.array([-dist_between_front_behind/2,0,-0.1])
        OBSTACLE4_RIGHT.translation += np.array([-dist_between_front_behind/2,length/2,0])
        OBSTACLE5_LEFT.translation += np.array([-dist_between_front_behind/2,-length/2,0])


        BIG_BOX = (OBSTACLE1_FRONT, OBSTACLE2_BEHIND, OBSTACLE3_BOTTOM, OBSTACLE4_RIGHT, OBSTACLE5_LEFT)


        # OBSTACLES DIMENSIONS


        OBSTACLE_DIM12_FRONT_BEHIND = np.array([width,length,height])
        OBSTACLE_DIM3_BOTTOM = np.array([dist_between_front_behind, length ,width])
        OBSTACLE45_LEFT_RIGHT = np.array([dist_between_front_behind, width ,height])


        BIG_BOX_DIM = (OBSTACLE_DIM12_FRONT_BEHIND,OBSTACLE_DIM12_FRONT_BEHIND, OBSTACLE_DIM3_BOTTOM, OBSTACLE45_LEFT_RIGHT, OBSTACLE45_LEFT_RIGHT )


        # Initial configuration of the robot
        INITIAL_CONFIG = pin.neutral(rmodel)
        # Creating the HPPFCL Shapes for the obstacles and the target
        TARGET_SHAPE = hppfcl.Sphere(5e-2)

        OBSTACLE_SHAPE12_FRONT_BEHIND = hppfcl.Box(OBSTACLE_DIM12_FRONT_BEHIND)
        OBSTACLE_SHAPE3_BOTTOM = hppfcl.Box(OBSTACLE_DIM3_BOTTOM)
        OBSTACLE_SHAPE45_LEFT_RIGHT = hppfcl.Box(OBSTACLE45_LEFT_RIGHT)

        BIG_BOX_SHAPE = (OBSTACLE_SHAPE12_FRONT_BEHIND, OBSTACLE_SHAPE12_FRONT_BEHIND, OBSTACLE_SHAPE3_BOTTOM, OBSTACLE_SHAPE45_LEFT_RIGHT, OBSTACLE_SHAPE45_LEFT_RIGHT)

        # Generating the meshcat visualizer
        MeshcatVis = MeshcatWrapper()
        vis = MeshcatVis.visualize(
            TARGET,
            OBSTACLE=BIG_BOX,
            robot_model=rmodel,
            robot_collision_model=cmodel,
            robot_visual_model=vmodel,
            obstacle_type="box",
            OBSTACLE_DIM=BIG_BOX_DIM,
        )
        vis = vis[0]
        vis.display(INITIAL_CONFIG)
        input()
        display_traj(vis, Q_trs_list[0])
            
    if CHECK_LIMITS:
        print(check_limits(rmodel,rdata, Q_trs_list[0]))
        
        collisions = []
        t_collision = []
        for t in range(int(len(Q_trs_list[0])/rmodel.nq)):
            q_t = get_q_iter_from_Q(Q_trs_list[0], t, rmodel.nq)
            pin.framesForwardKinematics(rmodel, rdata, q_t)
            pin.updateFramePlacements(rmodel, rdata)
            pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
            result = check_auto_collisions(rmodel, rdata, cmodel, cdata)
            if len(result) != 0:
                collisions.append(result)
                t_collision.append(t)
        print(f"collisions : {collisions} \n time : {t_collision} ")
    