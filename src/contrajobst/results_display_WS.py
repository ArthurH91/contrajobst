import os
from os.path import dirname, join, abspath

import json, codecs

import numpy as np
import pinocchio as pin
import hppfcl
import matplotlib.pyplot as plt

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper
from problem_traj_obs_with_obstacle_v2 import NLP_with_obs
from solver_newton_mt import SolverNewtonMt
from utils import display_last_traj, get_transform, get_difference_between_q_iter

###* HYPERPARAMS
T = 10
PLOT = False
DISPLAY = True


def display_traj(vis, Q_min, nq=7):
    for k in range(int(len(Q_min) / nq)):
        vis.display(np.array(Q_min[k * nq : (k + 1) * nq]))
        input()


if __name__ == "__main__":
    # Openning the files
    path = os.getcwd()

    results_json = codecs.open(
        path + "/results/results_theta_-18_06_WS_600_.json", "r", encoding="utf-8"
    ).read()

    results_bw_json = codecs.open(
        path + "/results/results_theta_-18_06_WS_600_bw.json", "r", encoding="utf-8"
    ).read()

    # Loading the json files
    results = json.loads(results_json)

    results_bw = json.loads(results_bw_json)

    theta_list = results["theta"]
    q_dot = []
    dist_min_obstacle = []
    initial_cost = []
    principal_cost = []
    terminal_cost = []

    obstacle_cost = []
    grad = []
    Q_min_list = []

    q_dot_bw = []
    dist_min_obstacle_bw = []
    initial_cost_bw = []
    principal_cost_bw = []
    terminal_cost_bw = []

    obstacle_cost_bw = []
    grad_bw = []
    Q_min_list_bw = []

    for theta in theta_list:
        q_dot.append(results["q_dot_" + str(round(theta, 3))])
        dist_min_obstacle.append(results["dist_min_obs_" + str(round(theta, 3))])
        initial_cost.append(results["initial_cost_" + str(round(theta, 3))])
        principal_cost.append(results["principal_cost_" + str(round(theta, 3))])
        terminal_cost.append(results["terminal_cost_" + str(round(theta, 3))])
        obstacle_cost.append(results["obstacle_cost_" + str(round(theta, 3))])
        grad.append(np.linalg.norm(results["grad_" + str(round(theta, 3))]))
        Q_min_list.append(results["Q_min_" + str(round(theta, 3))])

        q_dot_bw.append(results_bw["q_dot_" + str(round(theta, 3))])
        dist_min_obstacle_bw.append(results_bw["dist_min_obs_" + str(round(theta, 3))])
        initial_cost_bw.append(results_bw["initial_cost_" + str(round(theta, 3))])
        principal_cost_bw.append(results_bw["principal_cost_" + str(round(theta, 3))])
        terminal_cost_bw.append(results_bw["terminal_cost_" + str(round(theta, 3))])
        obstacle_cost_bw.append(results_bw["obstacle_cost_" + str(round(theta, 3))])
        grad_bw.append(np.linalg.norm(results_bw["grad_" + str(round(theta, 3))]))
        Q_min_list_bw.append(results_bw["Q_min_" + str(round(theta, 3))])


if PLOT:
    ###* SPEED

    plt.figure()
    for k in range(len(theta_list)):
        plt.plot(q_dot[k], "-o", label="theta = " + str(round(theta_list[k], 3)))

    plt.xlabel("Theta")
    plt.ylabel("Speed")
    plt.legend()
    plt.title("Speed through iterations")

    ###* DISTANCE MIN TO OBSTACLE
    plt.figure()

    for k in range(len(theta_list)):
        plt.plot(
            dist_min_obstacle[k], "-o", label="theta = " + str(round(theta_list[k], 3))
        )

    plt.plot(np.zeros(len(dist_min_obstacle[k])), label="Collision")
    plt.xlabel("Theta")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.title("Distance min to obstacle through iterations")

    ###* EVALUATION OF COSTS

    plt.figure()

    plt.subplot(221)
    plt.plot(theta_list, initial_cost, "o-")
    plt.ylabel("Initial cost")
    plt.xlabel("theta")
    plt.yscale("log")
    plt.title("Initial cost through theta (pose of obstacle)")

    plt.subplot(222)
    plt.plot(theta_list, principal_cost, "o-")
    plt.ylabel("Running cost")
    plt.xlabel("theta")
    plt.yscale("log")
    plt.title("Running cost through theta (pose of obstacle)")

    plt.subplot(223)
    plt.plot(theta_list, obstacle_cost, "o-")
    plt.ylabel("Obstacle cost")
    plt.xlabel("theta")
    plt.title("Obstacle cost through theta (pose of obstacle)")

    plt.subplot(224)
    plt.plot(theta_list, terminal_cost, "o-")
    plt.ylabel("Terminal cost")
    plt.xlabel("theta")
    plt.yscale("log")
    plt.title("Terminal cost through theta (pose of obstacle)")
    plt.suptitle("Costs through theta")

    ###* STUDY OF THE GRADIENT

    plt.figure()
    plt.plot(theta_list, grad, "-o")
    plt.ylabel("Gradient")
    plt.xlabel("Theta")
    plt.yscale("log")
    plt.title("Gradient norm through theta (pose of obstacle)")

    ###! BACKWARD GRAPHS

    ###* SPEED

    plt.figure()
    for k in range(len(theta_list)):
        plt.plot(q_dot_bw[k], "-o", label="theta = " + str(round(theta_list[k], 3)))

    plt.xlabel("Theta")
    plt.ylabel("Speed")
    plt.legend()
    plt.title("Speed through iterations (BACKWARD)")

    ###* DISTANCE MIN TO OBSTACLE
    plt.figure()

    for k in range(len(theta_list)):
        plt.plot(
            dist_min_obstacle_bw[k],
            "-o",
            label="theta = " + str(round(theta_list[k], 3)),
        )

    plt.plot(np.zeros(len(dist_min_obstacle_bw[k])), label="Collision")
    plt.xlabel("Theta")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.title("Distance min to obstacle through iterations (BACKWARD)")

    ###* EVALUATION OF COSTS

    plt.figure()

    plt.subplot(221)
    plt.plot(theta_list, initial_cost_bw, "o-")
    plt.ylabel("Initial cost")
    plt.xlabel("theta")
    plt.yscale("log")
    plt.title("Initial cost through theta (pose of obstacle)")

    plt.subplot(222)
    plt.plot(theta_list, principal_cost_bw, "o-")
    plt.ylabel("Running cost")
    plt.xlabel("theta")
    plt.yscale("log")
    plt.title("Running cost through theta (pose of obstacle)")

    plt.subplot(223)
    plt.plot(theta_list, obstacle_cost_bw, "o-")
    plt.ylabel("Obstacle cost")
    plt.xlabel("theta")
    plt.title("Obstacle cost through theta (pose of obstacle)")

    plt.subplot(224)
    plt.plot(theta_list, terminal_cost_bw, "o-")
    plt.ylabel("Terminal cost")
    plt.xlabel("theta")
    plt.yscale("log")
    plt.title("Terminal cost through theta (pose of obstacle)")
    plt.suptitle("Costs through theta (BACKWARD)")

    ###* STUDY OF THE GRADIENT

    plt.figure()
    plt.plot(theta_list, grad_bw, "-o")
    plt.ylabel("Gradient")
    plt.xlabel("Theta")
    plt.yscale("log")
    plt.title("Gradient norm through theta (pose of obstacle) (BACKWARD)")

    plt.show()

if DISPLAY:
    # * Generate a reachable target
    TARGET = pin.SE3.Identity()
    TARGET.translation = np.array([0, 0, 1])

    # * Generate a reachable obstacle
    OBSTACLE_translation = TARGET.translation / 2 + [
        0.2,
        0,
        1.0,
    ]
    rotation = np.identity(3)
    rotation[1, 1] = 0
    rotation[2, 2] = 0
    rotation[1, 2] = -1
    rotation[2, 1] = 1
    OBSTACLE_rotation = rotation
    OBSTACLE = TARGET.copy()
    OBSTACLE.translation = OBSTACLE_translation
    OBSTACLE.rotation = OBSTACLE_rotation

    ###* LOADING THE ROBOT
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)

    # Generating the meshcat visualizer

    # pin.seed(SEED)

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
    INITIAL_CONFIG = pin.neutral(rmodel)

    # Initial trajectory
    Q0 = np.concatenate([INITIAL_CONFIG] * (T))

    MeshcatVis = MeshcatWrapper()
    vis, meshcatvis = MeshcatVis.visualize(
        TARGET,
        OBSTACLE,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
        obstacle_type="sphere",
        OBSTACLE_DIM=1e-1,
    )

    vis.display(INITIAL_CONFIG)

    for k, theta in enumerate(theta_list):
        OBSTACLE.translation = TARGET.translation / 2 + [
            0.2 + theta,
            0 + theta,
            1.0 + theta,
        ]

        meshcatvis["obstacle"].set_transform(get_transform(OBSTACLE))
        print(
            f"Press enter for the {k} -th trajectory where theta = {theta} with the forward WS"
        )
        display_traj(vis, Q_min_list[k])
        print(
            f"Now press enter for the {k} -th trajectory where theta = {theta} but with the backward WS"
        )
        display_traj(vis, Q_min_list_bw[k])
