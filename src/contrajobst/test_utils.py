from os.path import dirname, join, abspath

import unittest
import numpy as np
from utils import (
    get_q_iter_from_Q,
    get_transform,
    get_difference_between_q_iter,
    generate_reachable_target,
    check_limits,
    check_auto_collisions
)
from wrapper_robot import RobotWrapper
import pinocchio as pin


class TestUtils(unittest.TestCase):
    def test_get_iter_from_Q(self):
        """Testing the function _get_iter_from_Q by comparing the first array of Q and the q_init, which should be the first array of Q"""
        q1 = np.ones((6))
        q2 = 2 * np.ones((6))
        Q = np.concatenate((q1, q2))
        self.assertTrue(
            np.array_equal(get_q_iter_from_Q(Q, 0, 6), q1),
            msg="q_iter obtained thanks to the method QP._get_q_iter_from_Q differs from the real q_iter",
        )
        self.assertTrue(
            np.array_equal(get_q_iter_from_Q(Q, 1, 6), q2),
            msg="q_iter obtained thanks to the method QP._get_q_iter_from_Q differs from the real q_iter",
        )

    def test_get_difference_between_q_iter(self):
        """Testing the function get_difference_between_q"""

        q1 = np.ones((6))
        q2 = 2 * np.ones((6))
        Q = np.concatenate((q1, q2))
        self.assertTrue(
            np.array_equal(get_difference_between_q_iter(Q, 1, 6), np.ones(6)),
            msg="Error while doing the difference between the arrays",
        )

    def test_get_transform(self):
        """Testing the function get_transform"""
        T = pin.SE3.Random()
        T_test = get_transform(T)
        self.assertTrue(
            np.all(np.isfinite(T_test)),
            msg="get_transform does not return a finite array",
        )
        self.assertTrue(
            np.array_equal(T_test[:3, 3], T.translation),
            msg=" Problem while comparing translation parts",
        )
        self.assertTrue(
            np.array_equal(T_test[:3, :3], T.rotation),
            msg="Problem while comparing the rotation parts",
        )

    def test_generate_reachable_target(self):
        """Testing the function generate_reachable_target by making sure it returns a finite array."""
        from wrapper_robot import RobotWrapper

        robot_wrapper = RobotWrapper()
        robot, rmodel, gmodel = robot_wrapper()
        rdata = rmodel.createData()

        p, q = generate_reachable_target(
            rmodel, rdata, "tool0", returnConfiguration=True
        )
        pin.framesForwardKinematics(rmodel, rdata, q)

        p_endeff = rdata.oMf[-1].translation
        dist_endeff_target = p_endeff - p.translation

        self.assertTrue(np.all(np.isfinite(p.translation)))
        self.assertTrue(np.isclose(np.linalg.norm(dist_endeff_target), 0, atol=1e-5))

    def test_check_limits(self):
        """Testing the function check_limits by making sure it tests the right limits
        """
        
        import example_robot_data
        robot=example_robot_data.load('ur10')
        upper_pos_limit = robot.model.upperPositionLimit
        lower_pos_limit = robot.model.lowerPositionLimit
        vel_limit = robot.model.velocityLimit
        data = robot.model.createData()
        
        # Checking whether the position limit fails correctly 
        q_upper_pos_limit_0 = robot.model.upperPositionLimit + np.array([1,0,0,0,0,0])
        self.assertFalse(check_limits(robot.model,data, q_upper_pos_limit_0, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[1], msg= "Problem of boolean while checking the limit of position")
        self.assertEqual(check_limits(robot.model,data, q_upper_pos_limit_0, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[3], q_upper_pos_limit_0[0], msg = "Problem of value when checking the limits of position")
        self.assertEqual(check_limits(robot.model,data, q_upper_pos_limit_0, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[5][0], 0.0, msg = "Problem of index when checking the limits of position")

        q_lower_pos_limit_1 = lower_pos_limit + np.array([0,-1,0,0,0,0])

        self.assertFalse(check_limits(robot.model,data, q_lower_pos_limit_1, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[1], msg= "Problem of boolean while checking the limit of position")
        self.assertEqual(check_limits(robot.model,data, q_lower_pos_limit_1, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[3], q_lower_pos_limit_1[1], msg = "Problem of value when checking the limits of position")
        self.assertEqual(check_limits(robot.model,data, q_lower_pos_limit_1, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[5][0], 1, msg = "Problem of index when checking the limits of position")

        # Checking whether the position limit succeed correctly 
        q_upper_pos_limit_2 = upper_pos_limit + np.array([0,0,-0.1,0,0,0])

        self.assertTrue(check_limits(robot.model, q_upper_pos_limit_2, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[1], msg= "Problem of boolean while checking the limit of position")
        
        # Checking the loop going through all the q
        Q = np.concatenate((q_upper_pos_limit_0, q_lower_pos_limit_1, q_upper_pos_limit_2))
        
        self.assertFalse(check_limits(robot.model,data, Q, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[1], msg= "Problem of boolean while checking the limit of position while testing with a traj")
        self.assertEqual(check_limits(robot.model,data, Q, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[3], [q_upper_pos_limit_0[0], q_lower_pos_limit_1[1]], msg = "Problem of value when checking the limits of position while testing with a traj")
        self.assertEqual(check_limits(robot.model,data, Q, CHECK_POS=True, CHECK_SPEED=False, CHECK_ACCEL= False)[5], [0,7], msg = "Problem of index when checking the limits of position while testing with a traj")

        # Testing the speed limits
        
        Q = np.concatenate((Q, np.zeros(len(q_upper_pos_limit_0))))

        self.assertFalse(check_limits(robot.model,data, Q, CHECK_POS=False, CHECK_SPEED=True, CHECK_ACCEL=False)[7], msg= "Problem of boolean while checking the limit of position while testing with a traj")


    def test_auto_collision(self):
        """Test whether the function detects an auto-collision or not.
        """
        
        # Importing the robot 
        
        pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

        model_path = join(pinocchio_model_dir, "franka_description/robots")
        mesh_dir = pinocchio_model_dir
        urdf_filename = "franka2.urdf"
        urdf_model_path = join(join(model_path, "panda"), urdf_filename)
        
        robot_wrapper = RobotWrapper(
            name_robot="franka",
            belong_to_example_robot_data=False,
            urdf_model_path=urdf_model_path,
            mesh_dir=mesh_dir,
        )
        rmodel, cmodel, vmodel = robot_wrapper()
        rdata = rmodel.createData()
        cdata = cmodel.createData()

        # Checking that the check works correctly
        q = pin.neutral(rmodel) # Shouldn't be a collision in neutral position
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        
        self.assertEqual(len(check_auto_collisions(rmodel, rdata, cmodel, cdata)), 0)
        
        q =  np.array([0,0,0,-3.5,0,0,0])
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        
        self.assertNotEqual(len(check_auto_collisions(rmodel, rdata, cmodel, cdata)), 0)

if __name__ == "__main__":
    unittest.main()
