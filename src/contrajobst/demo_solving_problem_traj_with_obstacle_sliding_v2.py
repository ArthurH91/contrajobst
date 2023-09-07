# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from os.path import dirname, join, abspath
import json

import numpy as np
import pinocchio as pin
from scipy.optimize import fmin
import hppfcl

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper
from problem_traj_obs_with_obstacle_v2 import NLP_with_obs
from solver_newton_mt import SolverNewtonMt
from utils import display_last_traj, get_transform, get_difference_between_q_iter

###* SETTING UP THE VARIABLES
# * HYPERPARMS
T = 10
WEIGHT_Q0 = 0.001
WEIGHT_DQ = 1e-3
WEIGHT_OBS = 20
WEIGHT_TERM_POS = 3
MAX_ITER = 700
EPS_SOLVER = 1e-6


# * Generate a reachable target
TARGET = pin.SE3.Identity()
TARGET.translation = np.array([0, 0, 1])

# * BOOLEANS FOR OPTIONS
WITH_DISPLAY = True
WITH_PLOT = False
WITH_WARMSTART = False
GOING_BACKWARD = False
WITH_DATA_SAVE = True

###* LOADING THE ROBOT
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "franka2.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)


if __name__ == "__main__":
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

    # Creating the HPPFCL Shapes for the obstacles and the target
    TARGET_SHAPE = hppfcl.Sphere(5e-2)
    OBSTACLE_SHAPE = hppfcl.Sphere(1e-1)
    theta_list = np.arange(-0.18, -0.16, 1e-2)

    Q_list = np.zeros(len(theta_list) * (T) * rmodel.nq)
    i = 0  # for filling Q_list

    if WITH_DATA_SAVE:
        data = {}

    for theta in theta_list:
        print(f"theta = {theta}")

        # * Generate a reachable obstacle
        OBSTACLE_translation = TARGET.translation / 2 + [
            0.2 + theta,
            0 + theta,
            1.0 + theta,
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

        # Creating the QP
        ca = NLP_with_obs(
            rmodel,
            cmodel,
            q0=INITIAL_CONFIG,
            TARGET=TARGET,
            TARGET_SHAPE=TARGET_SHAPE,
            OBSTACLE=OBSTACLE,
            OBSTACLE_SHAPE=OBSTACLE_SHAPE,
            eps_collision_avoidance=1e-5,
            T=T,
            WEIGHT_Q0=WEIGHT_Q0,
            WEIGHT_DQ=WEIGHT_DQ,
            WEIGHT_OBS=WEIGHT_OBS,
            WEIGHT_TERM=WEIGHT_TERM_POS,
        )
        LM_solver = SolverNewtonMt(
            ca.cost,
            ca.grad,
            ca.hess,
            max_iter=MAX_ITER,
            callback=None,
            verbose=True,
            eps=EPS_SOLVER,
        )
        LM_solver(Q0)
        Q_min = LM_solver._xval_k
        Q_list[(T) * i * rmodel.nq : (T) * (i + 1) * rmodel.nq] = Q_min

        q_dot = []
        for k in range(1, T):
            q_dot.append(
                np.linalg.norm(get_difference_between_q_iter(Q_min, k, rmodel.nq))
            )

        del LM_solver

        if WITH_WARMSTART:
            Q0 = Q_min

        if WITH_DATA_SAVE:
            data["Q_min_" + str(round(theta, 3))] = Q_min.tolist()
            data["q_dot_" + str(round(theta, 3))] = q_dot
            data["dist_min_obs_" + str(round(theta, 3))] = ca._dist_min_obs_list
            data["initial_cost_" + str(round(theta, 3))] = ca._initial_cost
            data["principal_cost_" + str(round(theta, 3))] = ca._principal_cost
            data["obstacle_cost_" + str(round(theta, 3))] = ca._obstacle_cost
            data["terminal_cost_" + str(round(theta, 3))] = ca._terminal_cost
            data["grad_" + str(round(theta, 3))] = ca.gradval.tolist()
        i += 1

    Q0 = np.concatenate([INITIAL_CONFIG] * (T))
    Q_list_reversed = np.zeros(len(theta_list) * (T) * rmodel.nq)
    i = 0  # for filling Q_list

    if GOING_BACKWARD:
        for theta in np.flip(theta_list):
            print(f"theta = {theta}")

            # * Generate a reachable obstacle
            OBSTACLE_translation = TARGET.translation / 2 + [
                0.2 + theta,
                0 + theta,
                1.0 + theta,
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

            # Creating the QP
            ca = NLP_with_obs(
                rmodel,
                cmodel,
                TARGET=TARGET,
                TARGET_SHAPE=TARGET_SHAPE,
                OBSTACLE=OBSTACLE,
                OBSTACLE_SHAPE=OBSTACLE_SHAPE,
                eps_collision_avoidance=1e-5,
                T=T,
                q0=INITIAL_CONFIG,
                WEIGHT_Q0=WEIGHT_Q0,
                WEIGHT_DQ=WEIGHT_DQ,
                WEIGHT_OBS=WEIGHT_OBS,
                WEIGHT_TERM=WEIGHT_TERM_POS,
            )
            LM_solver = SolverNewtonMt(
                ca.cost,
                ca.grad,
                ca.hess,
                max_iter=MAX_ITER,
                callback=None,
                verbose=True,
                eps=EPS_SOLVER,
            )
            LM_solver(Q0)
            Q_min = LM_solver._xval_k
            Q_list_reversed[(T) * i * rmodel.nq : (T) * (i + 1) * rmodel.nq] = Q_min

            if WITH_WARMSTART:
                Q0 = Q_min

            i += 1

    if WITH_DATA_SAVE:
        data["theta"] = theta_list.tolist()
        with open("results_theta_018017noWS.json", "w") as outfile:
            json.dump(data, outfile)

    # Generating the meshcat visualizer
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
    # Displaying the initial configuration of the robot
    vis.display(INITIAL_CONFIG)
    for k in range(len(theta_list)):
        theta = theta_list[k]
        print(f"press enter for displaying the {k}-th trajectory where theta = {theta}")
        input()
        Q = Q_list[(T) * rmodel.nq * k : (T) * rmodel.nq * (k + 1)]
        if GOING_BACKWARD:
            if k > 0:
                Q_reversed = Q_list_reversed[
                    -(T) * rmodel.nq * (k + 1) : -(T) * rmodel.nq * k
                ]
            else:
                Q_reversed = Q_list_reversed[-(T) * rmodel.nq * (k + 1) :]
        OBSTACLE_translation = TARGET.translation / 2 + [
            0.2 + theta,
            0 + theta,
            1.0 + theta,
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

        meshcatvis["obstacle"].set_transform(get_transform(OBSTACLE))
        display_last_traj(vis, Q, INITIAL_CONFIG, T)

        if GOING_BACKWARD:
            print(
                f"Now press enter for the {k} -th trajectory where theta = {theta} but with a different warm start"
            )
            display_last_traj(vis, Q_reversed, INITIAL_CONFIG, T)
