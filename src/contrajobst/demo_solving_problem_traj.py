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
from numpy.linalg import norm
import pinocchio as pin
import time
from scipy.optimize import fmin, fmin_bfgs
import matplotlib.pyplot as plt
import hppfcl
import pydiffcol

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper
from problem_traj_obs import CollisionAvoidance
from solver_newton_mt import SolverNewtonMt
from utils import display_last_traj, numdiff, get_q_iter_from_Q

# ### HYPERPARMS
T = 5
WEIGHT_Q0 = 0.001
WEIGHT_DQ = 0.001
WEIGHT_OBS = 1e-1
WEIGHT_TERM_POS = 4

# Generate a reachable target
TARGET = pin.SE3.Identity()
TARGET.translation = np.array([-0.25, 0, 1.6])

# Generate a reachable obstacle
OBSTACLE_translation = TARGET.translation / 2 + [0.2, 0, 0.8]
rotation = np.identity(3)
rotation[1, 1] = 0
rotation[2, 2] = 0
rotation[1, 2] = -1
rotation[2, 1] = 1
OBSTACLE_rotation = rotation
OBSTACLE = TARGET.copy()
OBSTACLE.translation = OBSTACLE_translation
OBSTACLE.rotation = OBSTACLE_rotation


# Max iterations of the solver
MAX_ITER = 50

WITH_DISPLAY = True
WITH_PLOT = True
WITH_NUMDIFF_SOLVE = False
WARMSTART_IPOPT_WITH_TRS = False
WITH_CASADI = False


###* LOADING THE ROBOT

pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "franka2.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)


### HELPERS (Finite difference comutation of the gradient and the hessian)
def grad_numdiff(Q: np.ndarray):
    return numdiff(ca.cost, Q)


def hess_numdiff(Q: np.ndarray):
    return numdiff(grad_numdiff, Q)


def obstacle_cost_function(Q: np.ndarray, eps=1e-4):
    # Going through all the configurations of the robot
    cost = 0
    for t in range(T):
        # Getting each configuration specifically
        q_t = get_q_iter_from_Q(Q, t, rmodel.nq)

        # Results requests from pydiffcol
        req = pydiffcol.DistanceRequest()
        res = pydiffcol.DistanceResult()

        # Updating the pinocchio models
        pin.framesForwardKinematics(rmodel, rdata, q_t)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

        # Getting the shape and the position of the end effector
        EndeffID = rmodel.getFrameId("panda2_leftfinger")
        EndeffID_geom = cmodel.getGeometryId("panda2_leftfinger_0")
        endeff_pos = rdata.oMf[EndeffID]
        endeff_shape = cmodel.geometryObjects[EndeffID_geom].geometry

        # Computing the nearest neighbors of the end effector and the obstacle
        dist_endeff_obs = pydiffcol.distance(
            endeff_shape, endeff_pos, OBSTACLE_SHAPE, OBSTACLE, req, res
        )

        # Computing the positions of the joints at each configuration
        # for oMg, geometry_objects in zip(gdata.oMg, gmodel.geometryObjects):
        #     print(geometry_objects)
        dist_endeff_target = pydiffcol.distance(
            endeff_shape, endeff_pos, TARGET_SHAPE, TARGET, req, res
        )
        cost += 4 * (dist_endeff_target) ** 2
        if dist_endeff_obs < eps:
            print("contact")
            cost += (dist_endeff_obs - eps) ** 2

    return cost


if __name__ == "__main__":
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
    # OBSTACLE_SHAPE = hppfcl.Box(5e-1, 5e-1, 5e-2)
    OBSTACLE_SHAPE = hppfcl.Sphere(1e-1)
    # Creating the QP
    ca = CollisionAvoidance(
        rmodel,
        rdata,
        cmodel,
        cdata,
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
    vis.display(pin.neutral(rmodel))
    # Initial trajectory
    Q0 = np.concatenate([INITIAL_CONFIG] * (T + 1))

    print(ca.cost(Q0))
    # Trust region solver
    trust_region_solver = SolverNewtonMt(
        ca.cost,
        grad_numdiff,
        hess_numdiff,
        max_iter=MAX_ITER,
        callback=None,
        verbose=True,
    )

    trust_region_solver(Q0)
    list_fval_mt, list_gradfkval_mt, list_alphak_mt, list_reguk = (
        trust_region_solver._fval_history,
        trust_region_solver._gradfval_history,
        trust_region_solver._alphak_history,
        trust_region_solver._reguk_history,
    )
    Q_trs = trust_region_solver._xval_k

    if WITH_NUMDIFF_SOLVE:
        # Trust region solver with finite difference
        trust_region_solver_nd = SolverNewtonMt(
            ca.cost,
            grad_numdiff,
            hess_numdiff,
            max_iter=MAX_ITER,
            callback=None,
            verbose=True,
        )
        res = trust_region_solver_nd(Q0)
        list_fval_mt_nd, list_gradfkval_mt_nd, list_alphak_mt_nd, list_reguk_nd = (
            trust_region_solver_nd._fval_history,
            trust_region_solver_nd._gradfval_history,
            trust_region_solver_nd._alphak_history,
            trust_region_solver_nd._reguk_history,
        )
        Q_nd = trust_region_solver_nd._xval_k

    if WITH_DISPLAY:
        print(
            "Press enter for displaying the trajectory of the newton's method from Marc Toussaint"
        )
        display_last_traj(vis, Q_trs, INITIAL_CONFIG, T)
        if WITH_NUMDIFF_SOLVE:
            print("Now the trajectory of the same method but with the num diff")
            display_last_traj(vis, Q_nd, INITIAL_CONFIG, T)

    if WITH_PLOT:
        plt.subplot(411)
        plt.plot(list_fval_mt, "-ob", label="Marc Toussaint's method")
        if WITH_NUMDIFF_SOLVE:
            plt.plot(list_fval_mt_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Cost")
        plt.legend()

        plt.subplot(412)
        plt.plot(list_gradfkval_mt, "-ob", label="Marc Toussaint's method")
        if WITH_NUMDIFF_SOLVE:
            plt.plot(list_gradfkval_mt_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Gradient")
        plt.legend()

        plt.subplot(413)
        plt.plot(list_alphak_mt, "-ob", label="Marc Toussaint's method")
        if WITH_NUMDIFF_SOLVE:
            plt.plot(list_alphak_mt_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Alpha")
        plt.legend()

        plt.subplot(414)
        plt.plot(list_reguk, "-ob", label="Marc Toussaint's method")
        if WITH_NUMDIFF_SOLVE:
            plt.plot(list_reguk_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Regularization")
        plt.xlabel("Iterations")
        plt.legend()

        plt.suptitle(
            " Comparison between Marc Toussaint's Newton method and finite difference method"
        )
        plt.show()
