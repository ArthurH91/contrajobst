from os.path import dirname, join, abspath

import pinocchio as pin
import matplotlib.pyplot as plt
import time
import numpy as np
import hppfcl
import pydiffcol

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from solver_newton_mt import SolverNewtonMt
from problem_traj_without_obs import NLP_without_obs
from utils import numdiff, display_last_traj, get_q_iter_from_Q


WITH_NUMDIFF = False
WITH_COMPUTE_TRAJ = False

# ### HYPERPARMS
T = 5
WEIGHT_Q0 = 0.001
WEIGHT_DQ = 0.001
WEIGHT_TERM_POS = 4

###* TARGET
# Generate a reachable target
TARGET = pin.SE3.Identity()
TARGET.translation = np.array([-0.25, 0, 1.6])
TARGET_SHAPE = hppfcl.Sphere(5e-2)

###* OBSTACLE
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
OBSTACLE_SHAPE = hppfcl.Sphere(1e-1)


###* TRAJECTORY TO EVALUATE

Q_eval = np.array(
    [
        -0.03199447,
        0.11442232,
        -0.00668186,
        -0.08579295,
        -0.01782012,
        0.42290925,
        -0.07617945,
        -0.06399285,
        0.22882778,
        -0.01336614,
        -0.17161702,
        -0.0356322,
        0.8458371,
        -0.15230576,
        -0.09599908,
        0.3431998,
        -0.02005528,
        -0.25750298,
        -0.05342813,
        1.26880179,
        -0.22832543,
        -0.12801704,
        0.4575251,
        -0.02675185,
        -0.3434765,
        -0.07119901,
        1.69181727,
        -0.3041812,
        -0.16004896,
        0.57183157,
        -0.03346017,
        -0.42948027,
        -0.08892488,
        2.11484191,
        -0.37976211,
        -0.19194626,
        0.68685329,
        -0.04020041,
        -0.5122528,
        -0.10623125,
        2.53667727,
        -0.45326312,
    ]
)


###* HELPERS
def grad_numdiff(q: np.ndarray):
    return numdiff(NLP.cost, q)


def hess_numdiff(q: np.ndarray):
    return numdiff(grad_numdiff, q)


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
    # pin.seed(1)
    ###* LOADING THE ROBOT

    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)

    # Creating the robot
    robot_wrapper = RobotWrapper(
        name_robot="franka",
        belong_to_example_robot_data=False,
        urdf_model_path=urdf_model_path,
        mesh_dir=mesh_dir,
    )
    rmodel, cmodel, vmodel = robot_wrapper()
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Initial position of the robot
    q0 = pin.neutral(rmodel)
    Q0 = np.concatenate([q0] * (T + 1))

    # Generating the shape of the target
    # The target shape is a ball of 5e-2 radii at the TARGET position

    TARGET_SHAPE = hppfcl.Sphere(5e-2)

    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        TARGET,
        OBSTACLE=OBSTACLE,
        obstacle_type="sphere",
        OBSTACLE_DIM=1e-1,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
    )
    if WITH_COMPUTE_TRAJ:
        # Creating the QP
        NLP = NLP_without_obs(
            rmodel,
            cmodel,
            q0,
            TARGET,
            TARGET_SHAPE,
            T,
            WEIGHT_Q0,
            WEIGHT_DQ,
            WEIGHT_TERM_POS,
        )

        # Displaying the initial configuration
        vis.display(q0)

        # Solving the problem
        trust_region_solver = SolverNewtonMt(
            NLP.cost,
            NLP.grad,
            NLP.hess,
            verbose=True,
            bool_plot_results=True,
            max_iter=1000,
        )
        res = trust_region_solver(Q0)

        Q_eval = trust_region_solver._xval_k
        display_last_traj(vis, Q_eval, q0, T)
        print(Q_eval)

    print(obstacle_cost_function(Q_eval))
