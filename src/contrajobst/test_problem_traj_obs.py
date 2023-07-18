import unittest
from os.path import dirname, join, abspath

import numpy as np
import pinocchio as pin
import hppfcl
import pydiffcol

from problem_traj_obs import CollisionAvoidance
from wrapper_robot import RobotWrapper
from utils import numdiff, generate_reachable_target

np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

# This module is for testing problem_traj_obs.py


class TestQuadraticProblemNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ###* HYPERPARMS
        cls._T = 5
        cls._WEIGHT_Q0 = 0.001
        cls._WEIGHT_DQ = 0.001
        cls._WEIGHT_OBS = 1
        cls._WEIGHT_TERM = 4
        cls._eps = 1e-5

        ###* TARGET
        # Generate a reachable target
        cls._TARGET = pin.SE3.Identity()
        cls._TARGET.translation = np.array([-0.25, 0, 1.6])
        cls._TARGET_SHAPE = hppfcl.Sphere(5e-2)

        ###* OBSTACLE
        k = -0.08
        OBSTACLE_translation = cls._TARGET.translation / 2 + [0.2 + k, 0 + k, 0.8 + k]
        rotation = np.identity(3)
        rotation[1, 1] = 0
        rotation[2, 2] = 0
        rotation[1, 2] = -1
        rotation[2, 1] = 1
        OBSTACLE_rotation = rotation
        cls._OBSTACLE = cls._TARGET.copy()
        cls._OBSTACLE.translation = OBSTACLE_translation
        cls._OBSTACLE.rotation = OBSTACLE_rotation
        cls._OBSTACLE_SHAPE = hppfcl.Sphere(1e-1)

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
        cls._rmodel, cls._cmodel, cls._vmodel = robot_wrapper()
        cls._rdata = cls._rmodel.createData()
        cls._cdata = cls._cmodel.createData()

        # Storing the IDs of the frame of the end effector

        cls._EndeffID = cls._rmodel.getFrameId("panda2_leftfinger")
        cls._EndeffID_geom = cls._cmodel.getGeometryId("panda2_leftfinger_0")

        # Generating the shape of the target
        # The target shape is a ball of 5e-2 radii at the TARGET position

        cls._TARGET_SHAPE = hppfcl.Sphere(5e-2)

    ###! TESTS

    def test_cost(self):
        """Test whether the cost is correctly computed."""

        # Initial position of the robot
        q0 = pin.neutral(self._rmodel)

        # Updating the models
        pin.framesForwardKinematics(self._rmodel, self._rdata, q0)
        pin.updateGeometryPlacements(
            self._rmodel, self._rdata, self._cmodel, self._cdata
        )

        # Results requests from pydiffcol
        self._req = pydiffcol.DistanceRequest()
        self._res = pydiffcol.DistanceResult()
        self._req.derivative_type = pydiffcol.DerivativeType.FirstOrderRS

        # Obtaining the cartesian position of the end effector.
        self.endeff_Transform = self._rdata.oMf[self._EndeffID]
        self.endeff_Shape = self._cmodel.geometryObjects[self._EndeffID_geom].geometry

        self._ca = CollisionAvoidance(
            self._rmodel,
            self._rdata,
            self._cmodel,
            self._cdata,
            self._TARGET,
            self._TARGET_SHAPE,
            self._OBSTACLE,
            self._OBSTACLE_SHAPE,
            self._eps,
            self._T,
            q0,
            self._WEIGHT_Q0,
            self._WEIGHT_DQ,
            self._WEIGHT_OBS,
            self._WEIGHT_TERM,
        )
        Q_neutral = np.repeat(q0, self._T + 1)

        dist_endeff_target = pydiffcol.distance(
            self.endeff_Shape,
            self.endeff_Transform,
            self._TARGET_SHAPE,
            self._TARGET,
            self._req,
            self._res,
        )

        # Testing the initial cost.
        neutral_cost = self._ca.cost(Q_neutral)
        print(neutral_cost)

        self.assertEqual(
            self._ca._initial_cost,
            0,
            "The initial cost should be equal to zero because the neutral trajectory isn't moving and starts at q0.",
        )

        # Testing the running obstacle residual.

        self.assertEqual(
            self._ca._principal_cost,
            0,
            "The principal residuals should be equal to 0 because the trajectory isn't moving and is out of an obstacle.",
        )

        # Testing the terminal cost.
        self.assertAlmostEqual(
            0.5 * (self._WEIGHT_TERM * dist_endeff_target) ** 2,
            self._ca._terminal_cost,
            8,
            "The terminal cost should be equal to the squared distance between the end effector and the target with a weight factor.",
        )


if __name__ == "__main__":
    unittest.main()
