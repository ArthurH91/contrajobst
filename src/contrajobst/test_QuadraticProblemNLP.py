import unittest

import numpy as np
import pinocchio as pin
import hppfcl

from problem_traj import QuadratricProblemNLP
from wrapper_robot import RobotWrapper
from utils import numdiff, generate_reachable_target

np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)


class TestQuadraticProblemNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup of the environement
        robot_wrapper = RobotWrapper()
        cls._robot, cls._rmodel, cls._gmodel = robot_wrapper()
        cls._rdata = cls._rmodel.createData()
        cls._gdata = cls._gmodel.createData()

        # Configuration array reaching the target

        cls._p_target, cls._q_target = generate_reachable_target(
            cls._rmodel, cls._rdata, returnConfiguration=True
        )
        cls._q_init = pin.randomConfiguration(cls._rmodel)
        cls._q_inter = pin.randomConfiguration(cls._rmodel)
        cls._Q_target = np.concatenate((cls._q_init, cls._q_inter, cls._q_target))


        # The target shape is a ball of 5e-2 radii at the TARGET position
        TARGET_SHAPE = hppfcl.Sphere(5e-2)

        # Configuring the computation of the QP
        WEIGHT_Q0 = 0.001
        WEIGHT_DQ = 0.001
        WEIGHT_TERM_POS = 4
        cls._T = 2
        cls._QP = QuadratricProblemNLP(
            cls._robot,
            cls._rmodel,
            cls._gmodel,
            cls._q_init,
            cls._p_target,
            TARGET_SHAPE,
            cls._T,
            WEIGHT_Q0,
            WEIGHT_DQ,
            WEIGHT_TERM_POS,
        )
        cls._QP._Q = cls._Q_target

    # Tests
    def test_gradient_finite_difference(self):
        """Testing the gradient of the cost with the finite difference method"""

        grad_numdiff = numdiff(self._QP.cost, self._Q_target)
        self._QP.grad(self._Q_target)
        self.assertAlmostEqual(
            np.linalg.norm(self._QP.gradval - grad_numdiff),
            0,
            msg="The gradient is not the same as the finite difference one",
        )


if __name__ == "__main__":
    # Start of the unit tests
    unittest.main()
