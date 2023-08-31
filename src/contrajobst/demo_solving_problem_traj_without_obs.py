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

import numpy as np
import pinocchio as pin
import time
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import hppfcl

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper
from problem_traj_without_obs import NLP_without_obs
from solver_newton_mt import SolverNewtonMt
from utils import (
    display_last_traj,
    get_difference_between_q_iter_sup,
    get_q_iter_from_Q,
)


###* HYPERPARMS

MAX_ITER = 3000

T = 5
WEIGHT_Q0 = 0.001
# WEIGHT_DQ = (
#     0.00001  #! CONVERGENCE IN 5 ITERATIONS BUT SKIPS STEPS (WITH T = 5, T = 10, T = 30)
# )
# WEIGHT_DQ = 0.0001  #! CONVERGENCE IN 2xx ITERATIONS BUT SKIPS STEPS (WITH T = 30) NO CONVERGENCE WITH T = 5 BUT SKIPS STEPS AS WELL
WEIGHT_DQ = (
    0.001  #! CONVERGENCE IN 6XX ITERATIONS WITH T = 5, 3XX WITH T = 10, 3XX WITH T = 30
)

# WEIGHT_Q0 = 0.01
# WEIGHT_DQ = 0.01  #! NO CONVERGENCE FOR T = 5, T = 10, 11X ITERATIONS WITH WEIGHT_Q0 = 0.01 WITH T = 30. NO CONV FOR T = 10 & T = 5
WEIGHT_TERM_POS = 4


# Generate a reachable target
TARGET = pin.SE3.Identity()
TARGET.translation = np.array([0, 0, 1])

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

    # Initial config of the robot
    INITIAL_CONFIG = pin.neutral(rmodel)

    # Shape of the target
    TARGET_SHAPE = hppfcl.Sphere(5e-2)

    # NLP
    NLP = NLP_without_obs(
        rmodel,
        cmodel,
        INITIAL_CONFIG,
        TARGET,
        TARGET_SHAPE,
        T,
        WEIGHT_Q0,
        WEIGHT_DQ,
        WEIGHT_TERM_POS,
    )

    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        TARGET,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
    )
    vis = vis[0]

    # Displaying the initial configuration of the robot
    vis.display(INITIAL_CONFIG)
    # Initial trajectory
    # Q0 = np.concatenate([INITIAL_CONFIG] * (T - 1))
    Q0 = np.concatenate([INITIAL_CONFIG] * (T))
    # Q0 = np.concatenate((Q0, pin.randomConfiguration(rmodel)))

    # Trust region solver
    trust_region_solver = SolverNewtonMt(
        NLP.cost,
        NLP.grad,
        NLP.hess,
        max_iter=MAX_ITER,
        callback=None,
        verbose=True,
        bool_plot_results=False,
        eps=1e-5,
    )

    trust_region_solver(Q0)
    list_fval_mt, list_gradfkval_mt, list_alphak_mt, list_reguk = (
        trust_region_solver._fval_history,
        trust_region_solver._gradfval_history,
        trust_region_solver._alphak_history,
        trust_region_solver._reguk_history,
    )
    Q_trs = trust_region_solver._xval_k

    q_dot = []

    for k in range(0, T - 1):
        q_dot.append(
            np.linalg.norm(get_difference_between_q_iter_sup(Q_trs, k, rmodel.nq))
        )

    plt.plot(q_dot)
    plt.xlabel("Iterations")
    plt.ylabel("Speed")
    plt.title("Evolution of speed through iterations")
    plt.show()

    print(
        "Press enter for displaying the trajectory of the newton's method from Marc Toussaint"
    )
    display_last_traj(vis, Q_trs, INITIAL_CONFIG, T)
