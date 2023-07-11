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


def obstacle_cost_function(Q: np.ndarray, eps=1e-4):
    # List of coefficients to move the obstacle
    factor_obs = [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04]

    # Creating the lists of the costs
    cost_panda2_link5_sc_3 = [0]
    cost_panda2_link5_sc_4 = [0]
    cost_panda2_link6_sc_0 = [0]
    cost_panda2_link6_sc_1 = [0]
    cost_panda2_link6_sc_2 = [0]
    cost_panda2_link7_sc_0 = [0]
    cost_panda2_link7_sc_1 = [0]
    cost_panda2_link7_sc_2 = [0]

    for factor in factor_obs:
        # Creating the obstacle
        OBSTACLE_translation = TARGET.translation / 2 + [
            0.2 + factor,
            0 + factor,
            0.8 + factor,
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

        print(f"factor : {factor}")

        # Going through all the configurations of the robot
        for t in range(T):
            # Getting each configuration specifically
            q_t = get_q_iter_from_Q(Q, t, rmodel.nq)

            # Results requests from pydiffcol
            req = pydiffcol.DistanceRequest()
            res = pydiffcol.DistanceResult()

            # Updating the pinocchio models
            pin.framesForwardKinematics(rmodel, rdata, q_t)
            pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

            # Computing the positions of the joints at each configuration
            for oMg, geometry_objects in zip(cdata.oMg, cmodel.geometryObjects):
                if isinstance(geometry_objects.geometry, hppfcl.Sphere) or isinstance(
                    geometry_objects.geometry, hppfcl.Cylinder
                ):
                    dist = pydiffcol.distance(
                        geometry_objects.geometry,
                        oMg,
                        OBSTACLE_SHAPE,
                        OBSTACLE,
                        req,
                        res,
                    )

                    if dist < eps:
                        print(
                            f"contact here : {geometry_objects.name}, for the configuration number : {t}"
                        )
                        if geometry_objects.name == "panda2_link5_sc_3":
                            if t < T - 1:
                                cost_panda2_link5_sc_3.append(
                                    (dist - eps) ** 2 + cost_panda2_link5_sc_3[-1]
                                )
                            else:
                                cost_panda2_link5_sc_3.append((dist - eps) ** 2)
                        if geometry_objects.name == "panda2_link5_sc_4":
                            if t < T - 1:
                                cost_panda2_link5_sc_4.append(
                                    (dist - eps) ** 2 + cost_panda2_link5_sc_4[-1]
                                )
                            else:
                                cost_panda2_link5_sc_4.append((dist - eps) ** 2)
                        if geometry_objects.name == "panda2_link6_sc_0":
                            if t < T - 1:
                                cost_panda2_link6_sc_0.append(
                                    (dist - eps) ** 2 + cost_panda2_link6_sc_0[-1]
                                )
                            else:
                                cost_panda2_link6_sc_0.append((dist - eps) ** 2)
                        if geometry_objects.name == "panda2_link6_sc_1":
                            if t < T - 1:
                                cost_panda2_link6_sc_1.append(
                                    (dist - eps) ** 2 + cost_panda2_link6_sc_1[-1]
                                )
                            else:
                                cost_panda2_link6_sc_1.append((dist - eps) ** 2)
                        if geometry_objects.name == "panda2_link6_sc_2":
                            if t < T - 1:
                                cost_panda2_link6_sc_2.append(
                                    (dist - eps) ** 2 + cost_panda2_link6_sc_2[-1]
                                )
                            else:
                                cost_panda2_link6_sc_2.append((dist - eps) ** 2)
                        if geometry_objects.name == "panda2_link7_sc_0":
                            if t < T - 1:
                                cost_panda2_link7_sc_0.append(
                                    (dist - eps) ** 2 + cost_panda2_link7_sc_0[-1]
                                )
                            else:
                                cost_panda2_link7_sc_0.append((dist - eps) ** 2)
                        if geometry_objects.name == "panda2_link7_sc_1":
                            if t < T - 1:
                                cost_panda2_link7_sc_1.append(
                                    (dist - eps) ** 2 + cost_panda2_link7_sc_1[-1]
                                )
                            else:
                                cost_panda2_link7_sc_1.append((dist - eps) ** 2)
                        if geometry_objects.name == "panda2_link7_sc_2":
                            if t < T - 1:
                                cost_panda2_link7_sc_2.append(
                                    (dist - eps) ** 2 + cost_panda2_link7_sc_2[-1]
                                )
                            else:
                                cost_panda2_link7_sc_2.append((dist - eps) ** 2)

    return (
        cost_panda2_link5_sc_3,
        cost_panda2_link5_sc_4,
        cost_panda2_link6_sc_0,
        cost_panda2_link6_sc_1,
        cost_panda2_link6_sc_2,
        cost_panda2_link7_sc_0,
        cost_panda2_link7_sc_1,
        cost_panda2_link7_sc_2,
    )


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
    # display_last_traj(vis, Q_eval, q0, T)

    (
        cost_panda2_link5_sc_3,
        cost_panda2_link5_sc_4,
        cost_panda2_link6_sc_0,
        cost_panda2_link6_sc_1,
        cost_panda2_link6_sc_2,
        cost_panda2_link7_sc_0,
        cost_panda2_link7_sc_1,
        cost_panda2_link7_sc_2,
    ) = obstacle_cost_function(Q_eval)

    plt.figure()
    plt.subplot(221)
    for t in range(T):
        plt.plot(cost_panda2_link5_sc_3[t * T : (t + 1) * T])
        plt.legend(f"{-0.1 + t * 0.02}")
    plt.xlabel("Configurations")
    plt.ylabel("Cost")
    plt.title(" Link 5 SC 3")
    plt.subplot(222)
    for t in range(T):
        plt.plot(cost_panda2_link5_sc_4[t * T : (t + 1) * T])
        plt.legend(f"{-0.1 + t * 0.02}")
    plt.xlabel("Configurations")
    plt.ylabel("Cost")
    plt.title(" Link 5 SC 4")
    plt.subplot(223)
    for t in range(T):
        plt.plot(cost_panda2_link6_sc_0[t * T : (t + 1) * T])
        plt.legend(f"{-0.1 + t * 0.02}")
    plt.xlabel("Configurations")
    plt.ylabel("Cost")
    plt.title(" Link 6 SC 0")
    plt.subplot(224)
    for t in range(T):
        plt.plot(cost_panda2_link6_sc_1[t * T : (t + 1) * T])
        plt.legend(f"{-0.1 + t * 0.02}")
    plt.xlabel("Configurations")
    plt.ylabel("Cost")
    plt.title(" Link 6 SC 1")

    plt.figure()

    plt.subplot(221)
    for t in range(T):
        plt.plot(cost_panda2_link6_sc_2[t * T : (t + 1) * T])
        plt.legend(f"{-0.1 + t * 0.02}")
    plt.xlabel("Configurations")
    plt.ylabel("Cost")
    plt.title(" Link 6 SC 2")
    plt.subplot(222)
    for t in range(T):
        plt.plot(cost_panda2_link7_sc_0[t * T : (t + 1) * T])
        plt.legend(f"{-0.1 + t * 0.02}")
    plt.xlabel("Configurations")
    plt.ylabel("Cost")
    plt.title(" Link 7 SC 0")
    plt.subplot(223)
    for t in range(T):
        plt.plot(cost_panda2_link7_sc_1[t * T : (t + 1) * T])
        plt.legend(f"{10* (-0.1 + t * 0.02)}")
    plt.xlabel("Configurations")
    plt.ylabel("Cost")
    plt.title(" Link 7 SC 1")
    plt.subplot(224)
    for t in range(T):
        plt.plot(cost_panda2_link7_sc_2[t * T : (t + 1) * T])
        plt.legend(f"{-0.1 + t * 0.02}")
    plt.xlabel("Configurations")
    plt.ylabel("Cost")
    plt.title(" Link 7 SC 2")
    plt.figlegend("")
    plt.show()
