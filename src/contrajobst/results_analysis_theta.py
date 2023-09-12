import os
import json, codecs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib.colors import LightSource
from matplotlib import cm

# OPTIONS

PLOT = True


# Openning the files
path = os.getcwd()

theta_json = codecs.open(
    path + "/results/results_theta_-18_06_WS_600_dtheta1e-3.json",
    "r",
    encoding="utf-8",
).read()

# Loading the json files
results = json.loads(theta_json)

theta_list = results["theta"]
q_dot = []
dist_min_obstacle = []
initial_cost = []
principal_cost = []
terminal_cost = []

obstacle_cost = []
grad = []


for theta in theta_list:
    q_dot.append(results["q_dot_" + str(round(theta, 3))])
    dist_min_obstacle.append(results["dist_min_obs_" + str(round(theta, 3))])
    initial_cost.append(results["initial_cost_" + str(round(theta, 3))])
    principal_cost.append(results["principal_cost_" + str(round(theta, 3))])
    terminal_cost.append(results["terminal_cost_" + str(round(theta, 3))])
    obstacle_cost.append(results["obstacle_cost_" + str(round(theta, 3))])
    grad.append(np.linalg.norm(results["grad_" + str(round(theta, 3))]))

if PLOT:
    ###* EVALUATION OF COSTS

    plt.figure()

    plt.plot(theta_list, initial_cost, "o-", label="Initial cost")
    plt.plot(theta_list, principal_cost, "o-", label="Regularization")
    plt.plot(theta_list, obstacle_cost, "o-", label="Obstacle cost")
    plt.plot(theta_list, terminal_cost, "o-", label="Terminal cost")
    plt.ylabel("Cost")
    plt.xlabel("theta")
    plt.legend()
    plt.title("Cost through theta (pose of obstacle)")

    ###* STUDY OF THE GRADIENT

    plt.figure()
    plt.plot(theta_list, grad, "-o")
    plt.ylabel("Gradient")
    plt.xlabel("Theta")
    plt.yscale("log")
    plt.title("Gradient norm through theta (pose of obstacle)")

    ###* STUDY OF THE SPEED IN 3D

    theta_array = np.array(theta_list)

    it = np.arange(0, 9, 1)

    theta_mesh, it_mesh = np.meshgrid(it, theta_array)

    q_dot_array = np.array(q_dot)
    ls = LightSource(270, 45)
    rgb = ls.shade(q_dot_array, cmap=cm.nipy_spectral, vert_exag=0.1, blend_mode="soft")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
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
    ax.contour(it_mesh, theta_mesh, q_dot_array, zdir="z", offset=-10, cmap="coolwarm")
    ax.contour(it_mesh, theta_mesh, q_dot_array, zdir="x", offset=-0.5, cmap="coolwarm")
    ax.contour(it_mesh, theta_mesh, q_dot_array, zdir="y", offset=10, cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel("Theta")
    ax.set_ylabel("Iterations")
    ax.set_zlabel("Speed")
    ax.set_title("Speed through iterations and theta")

    ###* STUDY OF THE OBSTACLE COLLISIONS IN 3D

    dist_min_obstacle_array = np.array(dist_min_obstacle)

    ls = LightSource(270, 45)
    rgb = ls.shade(q_dot_array, cmap=cm.nipy_spectral, vert_exag=0.1, blend_mode="soft")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
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
    ax.set_title("Distance min to obstacle through iterations and theta")

    plt.show()
