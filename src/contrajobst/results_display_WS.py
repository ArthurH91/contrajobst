import os
from os.path import dirname, join, abspath
import argparse
import json, codecs

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper

from utils import get_transform

###* PARSERS

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--plot", help="plot the results", action="store_true", default=False
)
parser.add_argument(
    "-d", "--display", help="display the results", action="store_true", default=False
)

args = parser.parse_args()

###* HYPERPARAMS
T = 10
PLOT = args.plot
DISPLAY = args.display
name = "results_theta_-18_06_WS_600_dtheta1e-3"


def display_traj(vis, Q_min, nq=7):
    for k in range(int(len(Q_min) / nq)):
        vis.display(np.array(Q_min[k * nq : (k + 1) * nq]))
        input()


if __name__ == "__main__":
    # Openning the files
    path = os.getcwd()

    results_json = codecs.open(
        path + "/results/" + name + ".json", "r", encoding="utf-8"
    ).read()

    results_bw_json = codecs.open(
        path + "/results/" + name + "bw" + ".json", "r", encoding="utf-8"
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
    ###* EVALUATION OF COSTS

    plt.figure()

    plt.plot(theta_list, initial_cost, "o-", label="Initial cost (FORWARD)")
    plt.plot(theta_list, principal_cost, "o-", label="Regularization (FORWARD)")
    plt.plot(theta_list, obstacle_cost, "o-", label="Obstacle cost (FORWARD)")
    plt.plot(theta_list, terminal_cost, "o-", label="Terminal cost (FORWARD)")

    plt.plot(theta_list, initial_cost_bw, "o-", label="Initial cost (BACKWARD)")
    plt.plot(theta_list, principal_cost_bw, "o-", label="Regularization (BACKWARD)")
    plt.plot(theta_list, obstacle_cost_bw, "o-", label="Obstacle cost (BACKWARD)")
    plt.plot(theta_list, terminal_cost_bw, "o-", label="Terminal cost (BACKWARD)")

    plt.ylabel("Cost")
    plt.xlabel("theta")
    plt.legend()

    plt.title("Cost through theta (pose of obstacle)")

    ###* STUDY OF THE GRADIENT

    plt.figure()
    plt.plot(theta_list, grad, "-o", label="Forward")
    plt.plot(theta_list, grad_bw, "-o", label="Backward")
    plt.ylabel("Gradient")
    plt.xlabel("Theta")
    plt.yscale("log")
    plt.legend()
    plt.title("Gradient norm through theta (pose of obstacle)")

    ###* STUDY OF THE SPEED IN 3D

    theta_array = np.array(theta_list)

    it = np.arange(0, 9, 1)

    theta_mesh, it_mesh = np.meshgrid(it, theta_array)

    # FORWARD
    q_dot_array = np.array(q_dot)
    ls = LightSource(270, 45)
    rgb = ls.shade(q_dot_array, cmap=cm.nipy_spectral, vert_exag=0.1, blend_mode="soft")
    fig = plt.figure()

    ax1 = fig.add_subplot(221, projection="3d")
    surf = ax1.plot_surface(
        it_mesh,
        theta_mesh,
        q_dot_array,
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        facecolors=rgb,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax1.contour(it_mesh, theta_mesh, q_dot_array, zdir="z", offset=-10, cmap="coolwarm")
    ax1.contour(
        it_mesh, theta_mesh, q_dot_array, zdir="x", offset=-0.5, cmap="coolwarm"
    )
    ax1.contour(it_mesh, theta_mesh, q_dot_array, zdir="y", offset=10, cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax1.set_xlabel("Theta")
    ax1.set_ylabel("Iterations")
    ax1.set_zlabel("Speed")
    ax1.set_title("Speed through iterations and theta (FORWARD)")

    # BACKWARD

    q_dot_bw_array = np.array(q_dot_bw)
    ax2 = fig.add_subplot(222, projection="3d")
    surf = ax2.plot_surface(
        it_mesh,
        theta_mesh,
        q_dot_array,
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        facecolors=rgb,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax2.contour(
        it_mesh, theta_mesh, q_dot_bw_array, zdir="z", offset=-10, cmap="coolwarm"
    )
    ax2.contour(
        it_mesh, theta_mesh, q_dot_bw_array, zdir="x", offset=-0.5, cmap="coolwarm"
    )
    ax2.contour(
        it_mesh, theta_mesh, q_dot_bw_array, zdir="y", offset=10, cmap="coolwarm"
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax2.set_xlabel("Theta")
    ax2.set_ylabel("Iterations")
    ax2.set_zlabel("Speed")
    ax2.set_title("Speed through iterations and theta (BACKWARD)")

    ###* STUDY OF THE OBSTACLE COLLISIONS IN 3D

    dist_min_obstacle_array = np.array(dist_min_obstacle)

    ls = LightSource(270, 45)
    rgb = ls.shade(
        dist_min_obstacle_array, cmap=cm.nipy_spectral, vert_exag=0.1, blend_mode="soft"
    )
    ax = fig.add_subplot(223, projection="3d")
    surf = ax.plot_surface(
        it_mesh,
        theta_mesh,
        dist_min_obstacle_array,
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        facecolors=rgb,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.contour(
        it_mesh,
        theta_mesh,
        dist_min_obstacle_array,
        zdir="z",
        offset=-10,
        cmap="coolwarm",
    )
    ax.contour(
        it_mesh,
        theta_mesh,
        dist_min_obstacle_array,
        zdir="x",
        offset=-0.5,
        cmap="coolwarm",
    )
    ax.contour(
        it_mesh,
        theta_mesh,
        dist_min_obstacle_array,
        zdir="y",
        offset=10,
        cmap="coolwarm",
    )

    ax.set_xlabel("Theta")
    ax.set_ylabel("Iterations")
    ax.set_zlabel("Distance min to obstacle (m)")
    ax.set_title("Distance min to obstacle through iterations and theta (FORWARD)")

    # BACKWARD

    dist_min_obstacle_bw_array = np.array(dist_min_obstacle_bw)

    ls = LightSource(270, 45)
    rgb = ls.shade(
        dist_min_obstacle_bw_array,
        cmap=cm.nipy_spectral,
        vert_exag=0.1,
        blend_mode="soft",
    )
    ax = fig.add_subplot(224, projection="3d")
    surf = ax.plot_surface(
        it_mesh,
        theta_mesh,
        dist_min_obstacle_bw_array,
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        facecolors=rgb,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.contour(
        it_mesh,
        theta_mesh,
        dist_min_obstacle_bw_array,
        zdir="z",
        offset=-10,
        cmap="coolwarm",
    )
    ax.contour(
        it_mesh,
        theta_mesh,
        dist_min_obstacle_bw_array,
        zdir="x",
        offset=-0.5,
        cmap="coolwarm",
    )
    ax.contour(
        it_mesh,
        theta_mesh,
        dist_min_obstacle_bw_array,
        zdir="y",
        offset=10,
        cmap="coolwarm",
    )

    ax.set_xlabel("Theta")
    ax.set_ylabel("Iterations")
    ax.set_zlabel("Distance min to obstacle (m)")
    ax.set_title("Distance min to obstacle through iterations and theta (BACKWARD)")
    plt.suptitle("Distance min to the obstacle")
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
