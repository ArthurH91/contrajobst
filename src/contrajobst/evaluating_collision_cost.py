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
WITH_COMPUTE_TRAJ = True

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
k = -0.08
OBSTACLE_translation = TARGET.translation / 2 + [0.2 + k, 0 + k, 0.8 + k]
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


def eval_cost_function_all_configurations(Q, link_name, bool_cost=True):
    config0 = []
    config1 = []
    config2 = []
    config3 = []
    config4 = []

    for t in range(T):
        theta = np.arange(-0.2, 0.2, 1e-4)
        q_4 = get_q_iter_from_Q(Q, t, rmodel.nq)
        eps = 1e-5
        # Results requests from pydiffcol
        req = pydiffcol.DistanceRequest()
        res = pydiffcol.DistanceResult()

        # Updating the pinocchio models
        pin.framesForwardKinematics(rmodel, rdata, q_4)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

        cost_panda2_link = []

        for oMg, geometry_objects in zip(cdata.oMg, cmodel.geometryObjects):
            if geometry_objects.name == link_name:
                for theta_val in theta:
                    # Creating the obstacle
                    OBSTACLE_translation = TARGET.translation / 2 + [
                        0.2 + theta_val,
                        0 + theta_val,
                        0.8 + theta_val,
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
                    OBSTACLE_SHAPE = hppfcl.Sphere(1e-1)

                    dist = pydiffcol.distance(
                        geometry_objects.geometry,
                        oMg,
                        OBSTACLE_SHAPE,
                        OBSTACLE,
                        req,
                        res,
                    )
                    if dist < eps and bool_cost:
                        cost_panda2_link.append((dist - eps) ** 2)
                    elif bool_cost:
                        cost_panda2_link.append(0)
                    else:
                        cost_panda2_link.append(dist)
                else:
                    pass
        if t == 0:
            config0 = cost_panda2_link.copy()
        if t == 1:
            config1 = cost_panda2_link.copy()
        if t == 2:
            config2 = cost_panda2_link.copy()
        if t == 3:
            config3 = cost_panda2_link.copy()
        if t == 4:
            config4 = cost_panda2_link.copy()
    return config0, config1, config2, config3, config4


if __name__ == "__main__":
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
            verbose=False,
            bool_plot_results=False,
            max_iter=1000,
        )
        res = trust_region_solver(Q0)

        Q_eval = trust_region_solver._xval_k
    display_last_traj(vis, Q_eval, q0, T)

    links = [
        "panda2_link5_sc_3",
        "panda2_link5_sc_4",
        "panda2_link6_sc_0",
        "panda2_link6_sc_1",
        "panda2_link6_sc_2",
        "panda2_link7_sc_0",
        "panda2_link7_sc_1",
        "panda2_link7_sc_2",
    ]
    subplots = [421, 422, 423, 424, 425, 426, 427, 428]
    theta = np.arange(-0.2, 0.2, 1e-4)
    plt.subplots(layout="constrained")

    for name, plotnumber in zip(links, subplots):
        (
            config0,
            config1,
            config2,
            config3,
            config4,
        ) = eval_cost_function_all_configurations(Q_eval, name, bool_cost=False)

        plt.subplot(plotnumber)
        plt.plot(theta, config0, label="q_0")
        plt.plot(theta, config1, label="q_1")
        plt.plot(theta, config2, label="q_2")
        plt.plot(theta, config3, label="q_3")
        plt.plot(theta, config4, label="q_4")
        plt.title(name, pad=20)
        plt.legend()
        plt.xlabel("Theta")
        plt.ylabel("Distance function")
    plt.suptitle(
        "Distance obstacle - links in fonction of the position of the obstacle"
    )
    plt.show()
