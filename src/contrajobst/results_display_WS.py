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

from utils import get_transform, linear_gradient, check_limits

###* PARSERS

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--plot", help="plot the results", action="store_true", default=False
)
parser.add_argument(
    "-d", "--display", help="display the results", action="store_true", default=False
)
parser.add_argument(
    "-b",
    "--backward",
    help="display the backward trajectory",
    action="store_true",
    default=False,
)

parser.add_argument(
    "-f",
    "--forward",
    help="display the forward trajectory",
    action="store_true",
    default=False,
)

# parser.add_argument(
#     "-q",
#     "--q",
#     help = ""
    
# )


args = parser.parse_args()

###* HYPERPARAMS
T = 10
nq = 7
PLOT = args.plot
PLOT2 = False
DISPLAY = args.display
ONLY_BACKWARD = args.backward
ONLY_FORWARD = args.forward
CHECK_LIMITS = True 

name = "results_250_q_box"


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
    # Loading the json file
    results = json.loads(results_json)
    theta_list = results["theta"]


    if not ONLY_FORWARD:
        results_bw_json = codecs.open(
            path + "/results/" + name + "bw" + ".json", "r", encoding="utf-8"
        ).read()
    # Loading the json file
    results_bw = json.loads(results_bw_json)
    theta_list = results_bw["theta_bw"]



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
        if not ONLY_BACKWARD:
            q_dot.append(results["q_dot_" + str(round(theta, 3))])
            dist_min_obstacle.append(results["dist_min_obs_" + str(round(theta, 3))])
            initial_cost.append(results["initial_cost_" + str(round(theta, 3))])
            principal_cost.append(results["principal_cost_" + str(round(theta, 3))])
            terminal_cost.append(results["terminal_cost_" + str(round(theta, 3))])
            obstacle_cost.append(results["obstacle_cost_" + str(round(theta, 3))])
            grad.append(np.linalg.norm(results["grad_" + str(round(theta, 3))]))
            Q_min_list.append(results["Q_min_" + str(round(theta, 3))])

        if not ONLY_FORWARD:
            q_dot_bw.append(results_bw["q_dot_" + str(round(theta, 3))])
            dist_min_obstacle_bw.append(results_bw["dist_min_obs_" + str(round(theta, 3))])
            initial_cost_bw.append(results_bw["initial_cost_" + str(round(theta, 3))])
            principal_cost_bw.append(results_bw["principal_cost_" + str(round(theta, 3))])
            terminal_cost_bw.append(results_bw["terminal_cost_" + str(round(theta, 3))])
            obstacle_cost_bw.append(results_bw["obstacle_cost_" + str(round(theta, 3))])
            grad_bw.append(np.linalg.norm(results_bw["grad_" + str(round(theta, 3))]))
            Q_min_list_bw.append(results_bw["Q_min_" + str(round(theta, 3))])

    # Creating the robot

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

if PLOT:
    ###* EVALUATION OF COSTS

    plt.figure()

    if not ONLY_BACKWARD:
        plt.plot(theta_list, initial_cost, "o-", label="Initial cost (FORWARD)")
        plt.plot(theta_list, principal_cost, "o-", label="Regularization (FORWARD)")
        plt.plot(theta_list, obstacle_cost, "o-", label="Obstacle cost (FORWARD)")
        plt.plot(theta_list, terminal_cost, "o-", label="Terminal cost (FORWARD)")
    if not ONLY_FORWARD:
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
    if not ONLY_BACKWARD:
        plt.plot(theta_list, grad, "-o", label="Forward")
    if not ONLY_FORWARD:
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
    ax1.contour(it_mesh, theta_mesh, q_dot_array, zdir="z", offset=-1, cmap="coolwarm")
    ax1.contour(
        it_mesh, theta_mesh, q_dot_array, zdir="x", offset=-0.5, cmap="coolwarm"
    )
    ax1.contour(it_mesh, theta_mesh, q_dot_array, zdir="y", offset=1, cmap="coolwarm")
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
        q_dot_bw_array,
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
        it_mesh, theta_mesh, q_dot_bw_array, zdir="z", offset=-1, cmap="coolwarm"
    )
    ax2.contour(
        it_mesh, theta_mesh, q_dot_bw_array, zdir="x", offset=-0.5, cmap="coolwarm"
    )
    ax2.contour(
        it_mesh, theta_mesh, q_dot_bw_array, zdir="y", offset=1, cmap="coolwarm"
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
        dist_min_obstacle_array,
        cmap=cm.nipy_spectral,
        vert_exag=0.1,
        blend_mode="soft",
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
        offset=-1,
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
        offset=1,
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

    ###* STATS

    # SPEED

    q_dot_mean = []
    q_dot_std = []

    q_dot_bw_mean = []
    q_dot_bw_std = []

    for i in range(len(q_dot)):
        q_dot_mean.append(np.mean(q_dot[i]))
        q_dot_bw_mean.append(np.mean(q_dot_bw[i]))

        q_dot_std.append(np.std(q_dot[i]))
        q_dot_bw_std.append(np.std(q_dot_bw[i]))

    # DIST MIN TO OBSTACLE

    dist_min_to_obs_mean = []
    dist_min_to_obs_std = []

    dist_min_to_obs_bw_mean = []
    dist_min_to_obs_bw_std = []

    for i in range(len(q_dot)):
        dist_min_to_obs_mean.append(np.mean(dist_min_obstacle[i]))
        dist_min_to_obs_bw_mean.append(np.mean(dist_min_obstacle_bw[i]))

        dist_min_to_obs_std.append(np.std(dist_min_obstacle[i]))
        dist_min_to_obs_bw_std.append(np.std(dist_min_obstacle_bw[i]))

    plt.figure()
    plt.subplot(212)
    if not ONLY_BACKWARD:
        plt.plot(theta_list, q_dot_mean, "o", label="Mean (Forward)")
        plt.fill_between(
            theta_list,
            np.array(q_dot_mean) + np.array(q_dot_std),
            np.array(q_dot_mean) - np.array(q_dot_std),
            alpha=0.3,
        )
    if not ONLY_FORWARD:
        plt.plot(theta_list, q_dot_bw_mean, "o", label="Mean (Backward)")
        plt.fill_between(
            theta_list,
            np.array(q_dot_bw_mean) + np.array(q_dot_bw_std),
            np.array(q_dot_bw_mean) - np.array(q_dot_bw_std),
            alpha=0.3,
        )
    plt.legend()
    plt.title("Mean of speed through thetas")
    plt.ylabel("Speed")
    plt.xlabel("Theta")

    plt.subplot(211)
    if not ONLY_BACKWARD:
        plt.plot(theta_list, dist_min_to_obs_mean, "o", label="Mean (Forward)")
        plt.fill_between(
            theta_list,
            np.array(dist_min_to_obs_mean) + np.array(dist_min_to_obs_std),
            np.array(dist_min_to_obs_mean) - np.array(dist_min_to_obs_std),
            alpha=0.3,
        )
    if not ONLY_FORWARD:
        plt.plot(theta_list, dist_min_to_obs_bw_mean, "o", label="Mean (Backward)")
        plt.fill_between(
            theta_list,
            np.array(dist_min_to_obs_bw_mean) + np.array(dist_min_to_obs_bw_std),
            np.array(dist_min_to_obs_bw_mean) - np.array(dist_min_to_obs_bw_std),
            alpha=0.3,
        )
    plt.plot(theta_list, np.zeros(len(theta_list)), "--", label="Collision")

    plt.legend()
    plt.ylabel("Distance min to obstacle")
    plt.xlabel("Theta")
    plt.title("Mean of distance to obstacle through thetas")
    plt.suptitle("Comparison of means and standard deviation")

    ###* COMPARISON SPEED / OBSTACLE COST

    plt.figure()
    if not ONLY_BACKWARD:
        plt.plot(
            theta_list,
            q_dot_mean,
            "o",
            label="Mean (Forward)",
        )

        plt.fill_between(
            theta_list,
            np.array(q_dot_mean) + np.array(q_dot_std),
            np.array(q_dot_mean) - np.array(q_dot_std),
            alpha=0.3,
        )
        plt.plot(
            theta_list,
            2.5e4 * np.array(obstacle_cost),
            "o--",
            label="Obstacle cost (FORWARD)",
            dashes=(5, 10),
        )
    if not ONLY_FORWARD:
        plt.plot(
            theta_list,
            q_dot_bw_mean,
            "o",
            label="Mean (Backward)",
        )
        plt.fill_between(
            theta_list,
            np.array(q_dot_bw_mean) + np.array(q_dot_bw_std),
            np.array(q_dot_bw_mean) - np.array(q_dot_bw_std),
            alpha=0.3,
        )
        plt.plot(
            theta_list,
            2.5e4 * np.array(obstacle_cost_bw),
            "o--",
            label="Obstacle cost (BACKWARD)",
            dashes=(5, 10),
        )

    plt.legend()
    plt.title("Mean of speed through thetas")
    plt.ylabel("Speed")
    plt.xlabel("Theta")

    ### STUDY OF THE q

    # STUDY OF q_i_0

if PLOT2:
    color1 = "#FB575D"
    color2 = "#3575D5"

    n_theta = 240
    divide = 10
    q_i_0_list = []
    q_i_1_list = []
    q_i_2_list = []
    q_i_3_list = []
    q_i_4_list = []
    q_i_5_list = []
    q_i_6_list = []


    for Q in Q_min_list_bw[:n_theta]:
        q_i_0 = []
        q_i_1 = []
        q_i_2 = []
        q_i_3 = []
        q_i_4 = []
        q_i_5 = []
        q_i_6 = []
        for k in range(T):
            q = Q[k * nq : (k + 1) * nq]
            q_i_0.append(q[0])
            q_i_1.append(q[1])
            q_i_2.append(q[2])
            q_i_3.append(q[3])
            q_i_4.append(q[4])
            q_i_5.append(q[5])
            q_i_6.append(q[6])

        q_i_0_list.append(q_i_0)
        q_i_1_list.append(q_i_1)
        q_i_2_list.append(q_i_2)
        q_i_3_list.append(q_i_3)
        q_i_4_list.append(q_i_4)
        q_i_5_list.append(q_i_5)
        q_i_6_list.append(q_i_6)

    col = linear_gradient(color1, color2, n_theta)

    plt.subplot(331)
    for theta_iter in range(n_theta):
        if theta_iter % divide == 0:
            plt.plot(q_i_0_list[theta_iter], "-o", color=col["hex"][theta_iter])

    plt.ylabel("Angle")
    plt.title("Evolution of q_i_0")

    plt.subplot(332)
    for theta_iter in range(n_theta):
        if theta_iter % divide == 0:
            plt.plot(q_i_1_list[theta_iter], "-o", color=col["hex"][theta_iter])

    plt.ylabel("Angle")
    plt.title("Evolution of q_i_1")

    plt.subplot(333)
    for theta_iter in range(n_theta):
        if theta_iter % divide == 0:
            plt.plot(q_i_2_list[theta_iter], "-o", color=col["hex"][theta_iter])

    plt.ylabel("Angle")
    plt.title("Evolution of q_i_2")

    plt.subplot(334)
    for theta_iter in range(n_theta):
        if theta_iter % divide == 0:
            plt.plot(q_i_6_list[theta_iter], "-o", color=col["hex"][theta_iter])

    plt.ylabel("Angle")
    plt.title("Evolution of q_i_3")

    plt.subplot(335)
    for theta_iter in range(n_theta):
        if theta_iter % divide == 0:
            plt.plot(q_i_4_list[theta_iter], "-o", color=col["hex"][theta_iter])

    plt.xlabel("Step of the trajectory")
    plt.ylabel("Angle")
    plt.title("Evolution of q_i_4")

    plt.subplot(336)
    for theta_iter in range(n_theta):
        if theta_iter % divide == 0:
            plt.plot(q_i_5_list[theta_iter], "-o", color=col["hex"][theta_iter])

    plt.xlabel("Step of the trajectory")
    plt.ylabel("Angle")
    plt.title("Evolution of q_i_5")

    plt.subplot(337)
    for theta_iter in range(n_theta):
        if theta_iter % divide == 0:
            plt.plot(q_i_6_list[theta_iter], "-o", color=col["hex"][theta_iter])

    plt.xlabel("Step of the trajectory")
    plt.ylabel("Angle")
    plt.title("Evolution of q_i_6")

    plt.suptitle("Evolution of q_i_j through the trajectory w.r.t theta ")


    ### 3D

    theta_array = np.array(theta_list)

    step = np.arange(0, 10, 1)

    theta_mesh, step_mesh = np.meshgrid(step, theta_array)

    ls = LightSource(270, 45)
    rgb = ls.shade(
        np.array(q_i_0_list), cmap=cm.nipy_spectral, vert_exag=0.1, blend_mode="soft"
    )
    fig = plt.figure()
    ax1 = fig.add_subplot(331, projection="3d")
    surf = ax1.plot_surface(
        step_mesh,
        theta_mesh,
        np.array(q_i_0_list),
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax1.set_xlabel("Theta")
    ax1.set_ylabel("Steps")
    ax1.set_zlabel("Angle")
    ax1.set_title("q_0 through steps and theta (BACKWARD)")

    ax2 = fig.add_subplot(332, projection="3d")
    surf = ax2.plot_surface(
        step_mesh,
        theta_mesh,
        np.array(q_i_1_list),
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax2.set_xlabel("Theta")
    ax2.set_ylabel("Steps")
    ax2.set_zlabel("Angle")
    ax2.set_title("q_1 through steps and theta (BACKWARD)")

    ax3 = fig.add_subplot(333, projection="3d")
    surf = ax3.plot_surface(
        step_mesh,
        theta_mesh,
        np.array(q_i_2_list),
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax3.set_xlabel("Theta")
    ax3.set_ylabel("Steps")
    ax3.set_zlabel("Angle")
    ax3.set_title("q_2 through steps and theta (BACKWARD)")

    ax4 = fig.add_subplot(334, projection="3d")
    surf = ax4.plot_surface(
        step_mesh,
        theta_mesh,
        np.array(q_i_3_list),
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax4.set_xlabel("Theta")
    ax4.set_ylabel("Steps")
    ax4.set_zlabel("Angle")
    ax4.set_title("q_3 through steps and theta (BACKWARD)")

    ax5 = fig.add_subplot(335, projection="3d")
    surf = ax5.plot_surface(
        step_mesh,
        theta_mesh,
        np.array(q_i_4_list),
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax5.set_xlabel("Theta")
    ax5.set_ylabel("Steps")
    ax5.set_zlabel("Angle")
    ax5.set_title("q_4 through steps and theta (BACKWARD)")


    ax6 = fig.add_subplot(336, projection="3d")
    surf = ax6.plot_surface(
        step_mesh,
        theta_mesh,
        np.array(q_i_5_list),
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax6.set_xlabel("Theta")
    ax6.set_ylabel("Steps")
    ax6.set_zlabel("Angle")
    ax6.set_title("q_5 through steps and theta (BACKWARD)")

    ax7 = fig.add_subplot(337, projection="3d")
    surf = ax7.plot_surface(
        step_mesh,
        theta_mesh,
        np.array(q_i_6_list),
        rstride=1,
        cstride=1,
        alpha=0.3,
        lw=0.5,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax7.set_xlabel("Theta")
    ax7.set_ylabel("Steps")
    ax7.set_zlabel("Angle")
    ax7.set_title("q_6 through steps and theta (BACKWARD)")

    plt.suptitle(
        "Evolution of each angle of the robot through the trajectory and through theta"
    )

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
        if not ONLY_BACKWARD:
            print(
                f"Press enter for the {k} -th trajectory where theta = {round(theta,4)} with the forward WS"
            )
            display_traj(vis, Q_min_list[k])
        if not ONLY_FORWARD:
            print(
                f"Now press enter for the {k} -th trajectory where theta = {round(theta,4)} but with the backward WS"
            )
            display_traj(vis, Q_min_list_bw[k])

if CHECK_LIMITS: 
    print(check_limits(rmodel, Q))