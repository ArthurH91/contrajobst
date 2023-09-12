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
import matplotlib.pyplot as plt
import hppfcl

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper
from problem_traj_obs_with_obstacle_v2 import NLP_with_obs
from solver_newton_mt import SolverNewtonMt
from utils import (
    display_last_traj,
    get_difference_between_q_iter,
    plot_end_effector_positions,
)


# ### HYPERPARMS
T = 10
WEIGHT_Q0 = 0.001
WEIGHT_DQ = 1e-3
WEIGHT_OBS = 10
WEIGHT_TERM_POS = 3
MAX_ITER = 500
EPS_SOLVER = 2e-6

###* OPTIONS
WITH_PLOTTING = False
WITH_DISPLAY = True
SAVE_RESULTS = False
PROFILER = False

# Generate a reachable target
TARGET = pin.SE3.Identity()
TARGET.translation = np.array([0, 0, 1])

# Generate a reachable obstacle
OBSTACLE_translation = np.array([0.2, 0, 1])
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
    # Creating the HPPFCL Shapes for the obstacles and the target
    TARGET_SHAPE = hppfcl.Sphere(5e-2)
    OBSTACLE_SHAPE = hppfcl.Sphere(1e-1)
    # Creating the QP
    NLP = NLP_with_obs(
        rmodel,
        cmodel,
        q0=INITIAL_CONFIG,
        TARGET=TARGET,
        TARGET_SHAPE=TARGET_SHAPE,
        OBSTACLE=OBSTACLE,
        OBSTACLE_SHAPE=OBSTACLE_SHAPE,
        eps_collision_avoidance=0,
        T=T,
        WEIGHT_Q0=WEIGHT_Q0,
        WEIGHT_DQ=WEIGHT_DQ,
        WEIGHT_OBS=WEIGHT_OBS,
        WEIGHT_TERM=WEIGHT_TERM_POS,
    )

    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        TARGET,
        OBSTACLE,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
        obstacle_type="sphere",
        OBSTACLE_DIM=1e-1,
    )
    vis = vis[0]

    # Displaying the initial configuration of the robot
    vis.display(INITIAL_CONFIG)

    # Initial trajectory
    Q0 = np.concatenate([INITIAL_CONFIG] * (T))

    # Trust region solver
    trust_region_solver = SolverNewtonMt(
        NLP.cost,
        NLP.grad,
        NLP.hess,
        max_iter=MAX_ITER,
        callback=None,
        verbose=True,
        eps=EPS_SOLVER,
        bool_plot_results=WITH_PLOTTING,
    )
    if PROFILER:
        import cProfile, pstats, io
        from pstats import SortKey

        pr = cProfile.Profile()
        pr.enable()
    trust_region_solver(Q0)

    if PROFILER:
        pr.disable()
    list_fval_mt, list_gradfkval_mt, list_alphak_mt, list_reguk = (
        trust_region_solver._fval_history,
        trust_region_solver._gradfval_history,
        trust_region_solver._alphak_history,
        trust_region_solver._reguk_history,
    )
    Q_trs = trust_region_solver._xval_k

    q_dot = []

    for k in range(1, T):
        q_dot.append(np.linalg.norm(get_difference_between_q_iter(Q_trs, k, rmodel.nq)))

    if SAVE_RESULTS:
        results = {
            "name": "Iteration 80",
            "Q_trs": Q_trs.tolist(),
            "q_dot": q_dot,
            "dist_min_obs": NLP._dist_min_obs_list,
            "initial_cost": NLP._initial_cost,
            "principal_cost": NLP._principal_cost,
            "obstacle_cost": NLP._obstacle_cost,
            "terminal_cost": NLP._terminal_cost,
            "grad": NLP.gradval.tolist(),
        }
        with open("results_it_80.json", "w") as outfile:
            json.dump(results, outfile)

    if WITH_PLOTTING:
        plt.figure()
        plt.plot(q_dot, "-o")
        plt.xlabel("Iterations")
        plt.ylabel("Speed")
        plt.title("Evolution of speed through iterations")

        plot_end_effector_positions(
            rmodel, cmodel, rdata, Q_trs, T, rmodel.nq, TARGET, TARGET_SHAPE
        )

        plt.figure()
        plt.plot(NLP._dist_min_obs_list, "-o", label=" Distance min")
        plt.plot(np.zeros(len(NLP._dist_min_obs_list)), label="Collision")
        plt.ylabel("Distance (m)")
        plt.xlabel("Iterations")
        plt.legend()
        plt.suptitle("Distance min of robot to obstacle")
        plt.show()

    if WITH_DISPLAY:
        print(
            "Press enter for displaying the trajectory of the newton's method from Marc Toussaint"
        )
        display_last_traj(vis, Q_trs, INITIAL_CONFIG, T)

        while True:
            print("replay?")
            print(
                "Press enter for displaying the trajectory of the newton's method from Marc Toussaint"
            )
            display_last_traj(vis, Q_trs, INITIAL_CONFIG, T)

    if PROFILER:
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
