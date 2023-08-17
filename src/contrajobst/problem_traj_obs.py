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

import numpy as np
import pinocchio as pin
import copy

import hppfcl
import pydiffcol

from wrapper_robot import RobotWrapper
from utils import get_q_iter_from_Q, get_difference_between_q_iter, numdiff

# This class is for defining the optimization problem and computing the cost function, its gradient and hessian.


class CollisionAvoidance:
    """Computes the cost, the gradient and the hessian of the NLP for collision avoidance"""

    def __init__(
        self,
        rmodel: pin.Model,
        rdata: pin.Data,
        cmodel: pin.Model,
        cdata: pin.Data,
        TARGET: pin.SE3,
        TARGET_SHAPE: hppfcl.ShapeBase,
        OBSTACLE: pin.SE3,
        OBSTACLE_SHAPE: hppfcl.ShapeBase,
        eps_collision_avoidance: float,
        T: int,
        q0: np.ndarray,
        WEIGHT_Q0: float,
        WEIGHT_DQ: float,
        WEIGHT_OBS: float,
        WEIGHT_TERM: float,
        WITH_DIFFCOL_FOR_TARGET=True,
    ):
        """Class which computes the cost, the gradient and the hessian of the NLP for collision avoidance.

        Parameters
        ----------
        rmodel : pin.Model
            Pinocchio Model of the robot.
        rdata : pin.Data
            Data of the model.
        cmodel : pin.Model
            Collision model of the robot.
        cdata : pin.Data
            Collision data of the robot
        TARGET : pin.SE3
            Position of the target, in a pin.SE3.
        TARGET_SHAPE : hppfcl.ShapeBase
            Shape of the target, have to be a convex one from hppfcl.ShapeBase.
        OBSTACLE : pin.SE3
            Position of the obstacle, in a pin.SE3.
        OBSTACLE_SHAPE : hppfcl.ShapeBase
            Shape of the obstacle, have to be a convex one from hppfcl.ShapeBase.
        eps_collision_avoidance : float
            Criteria of collision.
        T : int
            Number of configurations in a trajectory.
        q0 : np.ndarray
            Initial ocnfiguration of the robot.
        WEIGHT_Q0 : float
            Weight penalizing the initial position.
        WEIGHT_DQ : float
            Weight penalizing the displacement of the robot.
        WEIGHT_OBS : float
            Weight penalizing the collision with the obstacle.
        WEIGHT_TERM : float
            Weight penalizing the distance between the end effector and the target.
        WITH_DIFFCOL_FOR_TARGET : bool, optional
            , by default True
        """

        # Models of the robot
        self._rmodel = rmodel  # Robot Model of pinocchio
        self._cmodel = cmodel  # Collision Model of the robot
        self._nq = (
            self._rmodel.nq
        )  # storing nq, the number of articulations of the robot

        # Datas of the robot
        self._rdata = rdata  # Data of rmodel
        self._cdata = cdata  # Data of cmodel

        # Parameters of the optimization
        self._T = T  # Number of steps in the trajectory

        # Weights and helpers for the costs
        self._q0 = q0  # Initial configuraiton of the robot
        self._WEIGHT_Q0 = WEIGHT_Q0  # Weight of the initial cost
        self._eps_collision_avoidance = eps_collision_avoidance
        self._WEIGHT_DQ = WEIGHT_DQ  # Weight of the running cost
        self._WEIGHT_OBS = WEIGHT_OBS  # Weight of the obstacle cost
        self._WEIGHT_TERM = WEIGHT_TERM  # Weight of the terminal cost

        # Obstacle and target positions and shapes
        self._OBSTACLE = OBSTACLE
        self._OBSTACLE_SHAPE = OBSTACLE_SHAPE
        self._TARGET = TARGET
        self._TARGET_SHAPE = TARGET_SHAPE

        # Booleans
        self._WITH_DIFFCOL_FOR_TARGET = WITH_DIFFCOL_FOR_TARGET

        # Storing the IDs of the frame of the end effector

        self._EndeffID = self._rmodel.getFrameId("panda2_joint7")
        self._EndeffID_geom = self._cmodel.getGeometryId("panda2_link7_sc_5")
        self._Endeff_parent_frame = 68
        assert self._EndeffID_geom <= len(self._cmodel.geometryObjects)
        assert self._EndeffID <= len(self._rmodel.frames)

    def cost(self, Q: np.ndarray):
        """Computes the cost of the collision avoidance.

        Parameters
        ----------
        Q : np.ndarray
            Initial array of the initial trajectory of the robot.
        """

        ###* INITIAL RESIDUAL

        # This residual is to force the first configuration of the trajectory to always be the same.

        self._initial_residual = self._WEIGHT_Q0 * (
            get_q_iter_from_Q(Q, 0, self._nq) - self._q0
        )

        ###* RUNNING RESIDUAL AND COLLISION AVIODANCE RESIDUALS

        # Both residuals are computed at the same time because for each configuration of the trajectory,
        # all the collisions will be computed as well as the running residual.

        # Results requests from pydiffcol
        self._req = pydiffcol.DistanceRequest()
        self._res = pydiffcol.DistanceResult()
        self._req.derivative_type = pydiffcol.DerivativeType.FirstOrderRS

        # Creating an array storing all the residuals for the collisions
        self._running_obstacle_residual = np.zeros(1)

        # Going through all the configurations
        for t in range(1, self._T):
            # Obtaining the current configuration
            q_t = get_q_iter_from_Q(Q, t, self._nq)

            # Updating the pinocchio models
            pin.framesForwardKinematics(self._rmodel, self._rdata, q_t)
            pin.updateGeometryPlacements(
                self._rmodel, self._rdata, self._cmodel, self._cdata
            )

            # Computing the running residual
            self._running_obstacle_residual = np.concatenate(
                (
                    self._running_obstacle_residual,
                    self._WEIGHT_DQ * (get_difference_between_q_iter(Q, t, self._nq)),
                )
            )

            # Creating the np array that will stores all the obstacle residuals
            obstacle_residual_t = np.zeros(1)

            # Going through all the convex objects composing the robot and
            # computing the distance between them and the obstacle

            # Storing the number of geometrical objects composing the robot
            self._number_of_geom_objects = 0

            for oMg, geometry_objects in zip(
                self._cdata.oMg, self._cmodel.geometryObjects
            ):
                # Only selecting the shapes of the robot and not the environement
                if not isinstance(geometry_objects.geometry, hppfcl.Box):
                    self._number_of_geom_objects += 1

                    # Creating an array to temporarely store the residual
                    obstacle_residual_t_for_each_shape = np.zeros(3)
                    # Computing the distance between the given part of the robot and the obstacle
                    dist = pydiffcol.distance(
                        geometry_objects.geometry,
                        oMg,
                        self._OBSTACLE_SHAPE,
                        self._OBSTACLE,
                        self._req,
                        self._res,
                    )
                    # If the given part of the robot is too close to the obstacle, a residual is added. 0 Otherwise
                    if dist < self._eps_collision_avoidance:
                        obstacle_residual_t_for_each_shape = self._WEIGHT_OBS * (
                            self._res.w
                        )

                    # Adding the residual to the array storing the obstacle residuals for a given q_t
                    obstacle_residual_t = np.concatenate(
                        (obstacle_residual_t, obstacle_residual_t_for_each_shape)
                    )
            # Adding the residuals for a given q_t to the whole array of the obstacle residuals
            self._running_obstacle_residual = np.concatenate(
                (self._running_obstacle_residual, obstacle_residual_t[1:])
            )
        #! Getting rid of the 0
        self._running_obstacle_residual = self._running_obstacle_residual[1:]

        ###* TERMINAL RESIDUAL
        # Distance between the end effector at q_T of each trajectory and the target

        # Obtaining the cartesian position of the end effector.
        self.endeff_Transform = self._rdata.oMf[self._EndeffID]
        self.endeff_Shape = self._cmodel.geometryObjects[self._EndeffID_geom].geometry

        if self._WITH_DIFFCOL_FOR_TARGET:
            # Computing the distance between the target and the end effector
            dist_endeff_target = pydiffcol.distance(
                self.endeff_Shape,
                self.endeff_Transform,
                self._TARGET_SHAPE,
                self._TARGET,
                self._req,
                self._res,
            )
            self._terminal_residual = (self._WEIGHT_TERM) * self._res.w
        else:
            vect_dist_endeff_target = (
                self.endeff_Transform.translation - self._TARGET.translation
            )
            self._terminal_residual = (self._WEIGHT_TERM) * vect_dist_endeff_target

        ###* TOTAL RESIDUAL
        self._residual = np.concatenate(
            (
                self._initial_residual,
                self._running_obstacle_residual,
                self._terminal_residual,
            ),
            axis=None,
        )

        ###* COMPUTING COSTS
        self._initial_cost = 0.5 * sum(self._initial_residual**2)
        self._principal_cost = 0.5 * sum(self._running_obstacle_residual**2)
        self._terminal_cost = 0.5 * sum(self._terminal_residual**2)

        self.costval = self._initial_cost + self._terminal_cost + self._principal_cost

        return self.costval

    def grad(self, Q: np.ndarray):
        """Computes the gradient of the NLP given a trajectory.

        Parameters
        ----------
        Q : np.ndarray
            Array describing the trajectory of the robot. Q[0] should be equal to q_0 specified in CollisionAvoidance.__init__().

        Returns
        -------
        gradient : np.ndarray
            Gradient of the NLP.
        """

        # Computes the cost to initialize the variables.
        self.cost(Q)

        # Creating the array containing all the derivatives of the residuals
        self._derivative_residual = np.zeros((len(self._residual), self._nq * self._T))

        ###* DERIVATIVE OF THE INITIAL RESIDUAL
        self._derivative_residual[: self._nq, : self._nq] = np.eye(
            self._nq
        )  # Derivative of q0 - q_init wrt q0
        self._derivative_residual[self._nq : 2 * self._nq, : self._nq] = -np.eye(
            self._nq
        )  # Derivative of q1 - q0 wrt q0.

        for t in range(1, self._T):
            it = self._number_of_geom_objects * 3 + self._nq
            # Obtaining the current configuration
            q_t = get_q_iter_from_Q(Q, t, self._nq)

            # Updating the pinocchio models
            pin.framesForwardKinematics(self._rmodel, self._rdata, q_t)
            pin.updateGeometryPlacements(
                self._rmodel, self._rdata, self._cmodel, self._cdata
            )
            pin.computeAllTerms(
                self._rmodel, self._rdata, q_t, np.zeros(self._rmodel.nv)
            )
            ###* RUNNING AND OBSTACLE RESIDUALS

            # Creating the array storing the running and obstacle residuals
            # The 2 * self._nq is for the derivative of the running residual (in between the collisions avoidance residuals
            # and the 3 * self._number_of_geom is for the collision avoidance residuals.
            derivative_running_residual = np.zeros(
                (2 * self._nq + 3 * self._number_of_geom_objects, self._nq * self._T)
            )

            # Computing the running residual
            self._derivative_residual[
                self._nq + (t - 1) * it : 2 * self._nq + (t - 1) * it,
                t * self._nq : (t + 1) * self._nq,
            ] = np.eye(
                self._nq
            )  # derivative of q i - q_i-1 w.r.t. q_i
            if t < 4:
                self._derivative_residual[
                    self._nq + t * it : 2 * self._nq + t * it,
                    t * self._nq : (t + 1) * self._nq,
                ] = -np.eye(
                    self._nq
                )  # derivative of q_i +1 - q_i w.r.t. q_i

            # Derivatives of the obstacles residuals

            # Going through all the convex objects composing the robot and
            # computing the distance between them and the obstacle

            i = 0  # Iterator to count for the geometrical objects of the robot.

            for oMg, geometry_objects in zip(
                self._cdata.oMg, self._cmodel.geometryObjects
            ):
                # Only selecting the shapes of the robot and not the environement
                if not isinstance(geometry_objects.geometry, hppfcl.Box):
                    # Computing the distance between the given part of the robot and the obstacle

                    dist = pydiffcol.distance(
                        geometry_objects.geometry,
                        oMg,
                        self._OBSTACLE_SHAPE,
                        self._OBSTACLE,
                        self._req,
                        self._res,
                    )
                    # If the given part of the robot is too close to the obstacle, a residual is added. 0 Otherwise
                    if dist < self._eps_collision_avoidance:
                        # Computing the derivative of the terminal residual

                        # q_0 = get_q_iter_from_Q(self._Q, self._T, self._rmodel.nq)

                        # Computing the jacobians in pinocchio
                        pin.computeJointJacobians(self._rmodel, self._rdata, q_t)

                        # Computing the derivatives of the distance
                        _ = pydiffcol.distance_derivatives(
                            geometry_objects.geometry,
                            oMg,
                            self._OBSTACLE_SHAPE,
                            self._OBSTACLE,
                            self._req,
                            self._res,
                        )

                        # Getting the frame jacobian from the geometry object in the LOCAL reference frame
                        jacobian = pin.computeFrameJacobian(
                            self._rmodel,
                            self._rdata,
                            q_t,
                            geometry_objects.parentFrame,
                            pin.LOCAL,
                        )

                        # The jacobian here is the multiplication of the jacobian of the end effector and the jacobian of the distance between the geometry object and the obstacle
                        J = jacobian.T @ self._res.dw_dq1.T
                        self._derivative_residual[
                            self._nq
                            + it * (t - 1)
                            + i * 3 : self._nq
                            + it * (t - 1)
                            + (i + 1) * 3,
                            t * self._nq : (t + 1) * self._nq,
                        ] = (
                            self._WEIGHT_TERM * J.T
                        )

                    else:
                        self._derivative_residual[
                            self._nq
                            + it * (t - 1)
                            + i * 3 : self._nq
                            + it * (t - 1)
                            + (i + 1) * 3,
                            t * self._nq : (t + 1) * self._nq,
                        ] = np.zeros((3, self._nq))

                    i += 1

        ###* TERMINAL RESIDUAL DERIVATIVE

        # Computing the jacobians in pinocchio
        pin.computeJointJacobians(self._rmodel, self._rdata, q_t)

        if self._WITH_DIFFCOL_FOR_TARGET:
            # Computing the distance from the end effector to the target
            dist_endeff_target = pydiffcol.distance(
                self.endeff_Shape,
                self.endeff_Transform,
                self._TARGET_SHAPE,
                self._TARGET,
                self._req,
                self._res,
            )

            # Computing the distance derivatives
            _ = pydiffcol.distance_derivatives(
                self.endeff_Shape,
                self.endeff_Transform,
                self._OBSTACLE_SHAPE,
                self._OBSTACLE,
                self._req,
                self._res,
            )

            # # Getting the frame jacobian from the geometry object in the LOCAL reference frame
            jacobian = pin.computeFrameJacobian(
                self._rmodel,
                self._rdata,
                q_t,
                self._Endeff_parent_frame,
                pin.LOCAL,
            )

            self._derivative_residual[-3:, -7:] = (
                self._WEIGHT_TERM * (jacobian.T @ self._res.dw_dq1.T).T
            )
        else:
            J = pin.getFrameJacobian(
                self._rmodel, self._rdata, self._EndeffID, pin.LOCAL_WORLD_ALIGNED
            )
            self._derivative_terminal_residual = self._WEIGHT_TERM * J[:3]
            self._derivative_residual[-3:, -7:] = self._derivative_terminal_residual

        return self._derivative_residual.T @ self._residual

    def hess(self, Q: np.ndarray):
        """Returns the hessian of the cost function with regards to the gauss newton approximation"""
        self.cost(Q)
        self.grad(Q)
        self.hessval = self._derivative_residual.T @ self._derivative_residual
        return self.hessval


if __name__ == "__main__":
    print("ok")
