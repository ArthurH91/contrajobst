from os.path import dirname, join, abspath

import pinocchio as pin
import matplotlib.pyplot as plt
import time
import numpy as np
import hppfcl
from scipy import optimize

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from solver_newton_mt import SolverNewtonMt
from problem_kin import ProblemInverseKinematics
from utils import generate_reachable_target, numdiff


WITH_NUMDIFF = False
WITH_FMIN = True

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


###* HELPERS
def grad_numdiff(q: np.ndarray):
    return numdiff(QP.cost, q)


def hess_numdiff(q: np.ndarray):
    return numdiff(grad_numdiff, q)


def callback(q):
    vis.display(q)
    time.sleep(1e-3)


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

    # Generating the shape of the target
    # The target shape is a ball of 5e-2 radii at the TARGET position

    TARGET_SHAPE = hppfcl.Sphere(5e-2)

    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis, vis_pin = MeshcatVis.visualize(
        TARGET,
        OBSTACLE=OBSTACLE,
        obstacle_type="sphere",
        OBSTACLE_DIM=1e-1,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
    )

    # Creating the QP
    QP = ProblemInverseKinematics(
        rmodel,
        cmodel,
        TARGET,
        TARGET_SHAPE,
        obstacle=OBSTACLE,
        obstacle_shape=OBSTACLE_SHAPE,
    )

    # Initial configuration
    q0 = pin.randomConfiguration(rmodel)
    rmodel.q0 = q0

    # Displaying the initial configuration
    vis.display(q0)

    # Solving the problem
    trust_region_solver = SolverNewtonMt(
        QP.cost,
        QP.grad,
        QP.hessian,
        callback=callback,
        verbose=True,
        bool_plot_results=True,
        max_iter=1000,
    )
    res = trust_region_solver(q0)

    if WITH_NUMDIFF:
        trust_region_solver = SolverNewtonMt(
            QP.cost,
            grad_numdiff,
            grad_numdiff,
            callback=callback,
            verbose=True,
            bool_plot_results=True,
            max_iter=100,
        )
        res = trust_region_solver(q0)

    if WITH_FMIN:
        q_fmin = optimize.fmin_ncg(
            QP.cost, q0, QP.grad, fhess=QP.hessian, callback=callback
        )
