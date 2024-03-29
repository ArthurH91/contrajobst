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
from utils import (
    get_q_iter_from_Q,
    get_difference_between_q_iter,
    numdiff,
    get_difference_between_q_iter_sup,
    select_strategy,
    get_transform
)

# This class is for defining the optimization problem and computing the cost function, its gradient and hessian.


class NLP_with_obs:
    def __init__(
        self,
        rmodel: pin.Model,
        cmodel: pin.GeometryModel,
        TARGET: np.ndarray,
        TARGET_SHAPE: hppfcl.ShapeBase,
        OBSTACLE: tuple,
        OBSTACLE_SHAPE: tuple,
        T: int,
        q0: np.ndarray,
        WEIGHT_Q0: float,
        WEIGHT_DQ: float,
        WEIGHT_TERM: float,
        WEIGHT_OBS: float,
        eps_collision_avoidance=0,
        CAPSULE = False,
    ):
        """Class which computes the cost, the gradient and the hessian of the NLP for collision avoidance.

        Parameters
        ----------
        rmodel : pin.Model
            Pinocchio Model of the robot.
        cmodel : pin.Model
            Collision model of the robot.
        TARGET : pin.SE3
            Position of the target, in a pin.SE3.
        TARGET_SHAPE : hppfcl.ShapeBase
            Shape of the target, have to be a convex one from hppfcl.ShapeBase.
        OBSTACLE : tuple of pin.SE3
            Positions of the obstacles, in a tuple of pin.SE3.
        OBSTACLE_SHAPE : tuple of hppfcl.ShapeBase
            Shape of the obstacles, have to be from hppfcl.ShapeBase.
        T : int
            Number of configurations in a trajectory.
        q0 : np.ndarray
            Initial ocnfiguration of the robot.
        WEIGHT_Q0 : float
            Weight penalizing the initial position.
        WEIGHT_DQ : float
            Weight penalizing the displacement of the robot.
        WEIGHT_TERM : float
            Weight penalizing the distance between the end effector and the target.
        WEIGHT_OBS : float
            Weight penalizing the collision with the obstacle.
        eps_collision_avoidance : float, by default 1e-5.
            Criteria of collision.
        CAPSULE : bool, by default False.
            Transforming the cylinders + spheres in capsules for distances & distances derivatives.
        """

        # Models of the robot
        self._rmodel = rmodel  # Robot Model of pinocchio
        self._cmodel = cmodel  # Collision Model of the robot
        self._nq = (
            self._rmodel.nq
        )  # storing nq, the number of articulations of the robot

        self._rdata = rmodel.createData()  # Data of rmodel
        self._cdata = cmodel.createData()  # Data of cmodel

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
        
        if type(OBSTACLE) == tuple:
            self._number_of_obstacles = len(self._OBSTACLE)
        else:
            self._number_of_obstacles = 1
        # FLAGS
        self._CAPSULE = CAPSULE # BOOLEAN DETERMINING WHEHTER TO TAKE THE CAPSULE INTO ACCOUNT OR THE 2 SPHERES & THE CYLINDER
        
        # Storing the IDs of the frame of the end effector

        self._EndeffID = self._rmodel.getFrameId("panda2_joint7")
        self._EndeffID_geom = self._cmodel.getGeometryId("panda2_link7_sc_5")
        # self._EndeffID_geom = self._cmodel.getGeometryId("end_effector_geom")

        self._Endeff_parent_frame = 68
        assert self._EndeffID_geom <= len(self._cmodel.geometryObjects)
        assert self._EndeffID <= len(self._rmodel.frames)

    def cost(self, Q: np.ndarray, eps_obstacle=1e-5):
        """Computes the cost of the QP.

        Parameters
        ----------
        Q : np.ndarray
            Array of shape (T*rmodel.nq) in which all the configurations of the robot are, in a single column.

        Returns
        -------
        self._cost : float
            Sum of the costs
        """

        self._Q = Q

        ###* INITIAL RESIDUAL
        ### Computing the distance between q0 and q_init to make sure the robot starts at the right place
        self._initial_residual = (
            get_q_iter_from_Q(self._Q, 0, self._rmodel.nq) - self._q0
        )

        # Penalizing the initial residual
        self._initial_residual *= self._WEIGHT_Q0

        ###* RUNNING RESIDUAL
        ### Running residuals are computed by diffenciating between q_th and q_th +1

        self._principal_residual = np.zeros(self._rmodel.nq * (self._T - 1))
        for iter in range(0, self._T - 1):
            self._principal_residual[
                (iter) * self._rmodel.nq : (iter + 1) * self._rmodel.nq
            ] = (
                get_difference_between_q_iter_sup(Q, iter, self._rmodel.nq)
                * self._WEIGHT_DQ
            )

        ###* TERMINAL RESIDUAL
        ### Computing the distance between the last configuration and the target

        # Distance request for pydiffcol
        self._req, self._req_diff = select_strategy("first_order_gaussian")
        self._res = pydiffcol.DistanceResult()
        self._res_diff = pydiffcol.DerivativeResult()

        # self._req.derivative_type = pydiffcol.DerivativeType.FirstOrderRS

        # Obtaining the last configuration of Q
        q_last = get_q_iter_from_Q(self._Q, self._T - 1, self._rmodel.nq)

        # Forward kinematics of the robot at the configuration q.
        pin.framesForwardKinematics(self._rmodel, self._rdata, q_last)
        pin.updateGeometryPlacements(
            self._rmodel, self._rdata, self._cmodel, self._cdata, q_last
        )

        # Obtaining the cartesian position of the end effector.
        self.endeff_Transform = self._rdata.oMf[self._EndeffID]
        self.endeff_Shape = self._cmodel.geometryObjects[self._EndeffID_geom].geometry

        # #
        # dist_endeff_target = pydiffcol.distance(
        #     self.endeff_Shape,
        #     self.endeff_Transform,
        #     self._TARGET_SHAPE,
        #     self._TARGET,
        #     self._req,
        #     self._res,
        # )
        # self._terminal_residual = (self._WEIGHT_TERM) * self._res.w

        p_endeff_loc = self._cmodel.geometryObjects[self._EndeffID_geom].placement
        frame_parent_frame = 68
        
        
        p_endeff_world = (get_transform(self._rdata.oMf[frame_parent_frame]) @ get_transform(p_endeff_loc))
        
        
        dist_endeff_target = pydiffcol.distance(
            self.endeff_Shape,
            pin.SE3(p_endeff_world),
            self._TARGET_SHAPE,
            self._TARGET,
            self._req,
            self._res,
        )

        self._terminal_residual = (self._WEIGHT_TERM) * self._res.w


        # self._terminal_residual = (self._WEIGHT_TERM) * (p_endeff_world - self._TARGET.translation)
        
        ###* OBSTACLE RESIDUAL
        obstacle_residuals_list = []

        # Dict for interpreting the results
        self._dist_min_obs_list = {}           
        for k in range(self._number_of_obstacles):
                self._dist_min_obs_list["obs" + str(k)] = []
        
        
        self._non_zero_residual_t = []

        for t in range(1, self._T):
        
            dist_obs_list_t = {}    
            for k in range(self._number_of_obstacles):
               dist_obs_list_t["obs" + str(k)] = []
            self._number_of_objects = 0

            # Obtaining the current configuration
            q_t = get_q_iter_from_Q(Q, t, self._nq)

            # Updating the pinocchio models
            pin.framesForwardKinematics(self._rmodel, self._rdata, q_t)
            pin.updateGeometryPlacements(
                self._rmodel, self._rdata, self._cmodel, self._cdata
            )

            # Going through all the convex objects composing the robot and
            # computing the distance between them and the obstacle
            for oMg, geometry_objects in zip(
                self._cdata.oMg, self._cmodel.geometryObjects
            ):
                if self._CAPSULE:
                    # Only selecting the shapes of the robot and not the environement
                    if isinstance(geometry_objects.geometry, hppfcl.Cylinder):
                        self._number_of_objects += 1
                        # Going through all the obstacles
                        for cntr_obs, (obs_pose, obs_shape) in enumerate(zip(self._OBSTACLE, self._OBSTACLE_SHAPE)):
                            # Computing the distance between the given part of the robot and the obstacle
                            dist = pydiffcol.distance(
                                hppfcl.Capsule(geometry_objects.geometry.radius,geometry_objects.geometry.halfLength),
                                oMg,
                                obs_shape,
                                obs_pose,
                                self._req,
                                self._res,
                            )
                            
                            # Creating an array to temporarely store the residual
                            obstacle_residual_t_for_each_shape = np.zeros(3)
                            # If the given part of the robot is too close to the obstacle, a residual is added. 0 Otherwise
                            if dist < self._eps_collision_avoidance:
                                obstacle_residual_t_for_each_shape = self._WEIGHT_OBS * (
                                    self._res.w
                                )
                                self._non_zero_residual_t.append(
                                    self._WEIGHT_OBS * (self._res.w)
                                )

                            dist_obs_list_t["obs" + str(cntr_obs)].append(dist)

                            # Adding the residual to the list of residuals
                            obstacle_residuals_list.append(obstacle_residual_t_for_each_shape)
                else:
                    if not isinstance(geometry_objects.geometry, hppfcl.Box):
                        self._number_of_objects += 1
                        # Going through all the obstacles
                        for cntr_obs, (obs_pose, obs_shape) in enumerate(zip(self._OBSTACLE, self._OBSTACLE_SHAPE)):
                            # Computing the distance between the given part of the robot and the obstacle
                            dist = pydiffcol.distance(
                                geometry_objects.geometry,
                                oMg,
                                obs_shape,
                                obs_pose,
                                self._req,
                                self._res,
                            )
                        
                            # Creating an array to temporarely store the residual
                            obstacle_residual_t_for_each_shape = np.zeros(3)
                            # If the given part of the robot is too close to the obstacle, a residual is added. 0 Otherwise
                            if dist < self._eps_collision_avoidance:
                                obstacle_residual_t_for_each_shape = self._WEIGHT_OBS * (
                                    self._res.w
                                )
                                self._non_zero_residual_t.append(
                                    self._WEIGHT_OBS * (self._res.w)
                                )
                            print(dist_obs_list_t)
                            dist_obs_list_t["obs" + str(cntr_obs)].append(dist)

                            # Adding the residual to the list of residuals
                            obstacle_residuals_list.append(obstacle_residual_t_for_each_shape)
                            
            # Adding to the dict the minimal distance between obstacles : 
            for k in range(len(self._OBSTACLE)):
                self._dist_min_obs_list["obs" + str(k)].append(min(dist_obs_list_t["obs" + str(k)]))
        

        self._obstacle_residual = np.zeros(
            (len(obstacle_residuals_list) * len(obstacle_residual_t_for_each_shape),)
        )

        # Converting the list of residuals in an array.
        for i, j in enumerate(obstacle_residuals_list):
            self._obstacle_residual[3 * i : 3 * (i + 1)] = j

        ###* TOTAL RESIDUAL

        # Creating the ndarray to store all the residuals
        self._residual = np.zeros(
            (
                len(self._initial_residual)
                + len(self._principal_residual)
                + len(self._terminal_residual)
                + len(self._obstacle_residual)
            )
        )

        # Putting the residuals in the residual array
        self._residual[: len(self._initial_residual)] = self._initial_residual
        self._residual[
            len(self._initial_residual) : len(self._initial_residual)
            + len(self._principal_residual)
        ] = self._principal_residual
        self._residual[
            len(self._initial_residual)
            + len(self._principal_residual) : len(self._initial_residual)
            + len(self._principal_residual)
            + len(self._terminal_residual)
        ] = self._terminal_residual
        self._residual[
            len(self._initial_residual)
            + len(self._principal_residual)
            + len(self._terminal_residual) :
        ] = self._obstacle_residual

        ###* COMPUTING COSTS
        self._initial_cost = 0.5 * sum(self._initial_residual**2)
        self._principal_cost = 0.5 * sum(self._principal_residual**2)
        self._terminal_cost = 0.5 * sum(self._terminal_residual**2)
        self._obstacle_cost = 0.5 * sum(self._obstacle_residual**2)
        self.costval = (
            self._initial_cost
            + self._terminal_cost
            + self._principal_cost
            + self._obstacle_cost
        )

        return self.costval

    def cost_obstacle(self, Q: np.ndarray):
        self.cost(Q)

        return self._obstacle_cost

    def grad_obstacle(self, Q: np.ndarray):
        self.grad(Q)
        gradval = self._derivative_residual_sec_part.T @ self._obstacle_residual

        return gradval

    def numdiff_grad_obstacle(self, Q: np.ndarray):
        return numdiff(self.cost_obstacle, Q, eps=1e-8)

    def grad(self, Q: np.ndarray):
        """Returns the grad of the cost function.

        Parameters
        ----------
        Q : np.ndarray
            Array of shape (T*rmodel.nq) in which all the configurations of the robot are, in a single column.

        Returns
        -------
        gradient : np.ndarray
            Array of shape (T*rmodel.nq + 3) in which the values of the gradient of the cost function are computed.
        """

        ### COST AND RESIDUALS
        self.cost(Q)

        ### DERIVATIVES OF THE RESIDUALS

        # Computing the derivative of the initial residuals
        self._derivative_initial_residual = np.diag([self._WEIGHT_Q0] * self._rmodel.nq)

        # Computing the derivative of the principal residual
        nq, T = self._rmodel.nq, self._T
        J = np.zeros(((T - 1) * nq, (T) * nq))

        np.fill_diagonal(J, -self._WEIGHT_DQ)
        np.fill_diagonal(J[:, nq:], self._WEIGHT_DQ)

        self._derivative_principal_residual = J

        # Computing the derivative of the terminal residual
        q_terminal = get_q_iter_from_Q(self._Q, self._T - 1, self._rmodel.nq)
        pin.computeJointJacobians(self._rmodel, self._rdata, q_terminal)
        jacobian = pin.getFrameJacobian(
            self._rmodel, self._rdata, self._EndeffID, pin.LOCAL
        )

        # dist = pydiffcol.distance(
        #     self.endeff_Shape,
        #     self.endeff_Transform,
        #     self._TARGET_SHAPE,
        #     self._TARGET,
        #     self._req,
        #     self._res,
        # )
        # _ = pydiffcol.distance_derivatives(
        #     self.endeff_Shape,
        #     self.endeff_Transform,
        #     self._TARGET_SHAPE,
        #     self._TARGET,
        #     self._req,
        #     self._res,
        #     self._req_diff,
        #     self._res_diff
        # )

        # J = jacobian.T @ self._res_diff.dn_loc_dM1.T

        J = pin.getFrameJacobian(
                    self._rmodel, self._rdata, self._EndeffID, pin.LOCAL_WORLD_ALIGNED
                )[:3].T

        self._derivative_terminal_residual = self._WEIGHT_TERM * J.T

        # Putting them all together

        self._derivative_residual_first_part = np.zeros(
            [(self._T) * self._rmodel.nq + 3, (self._T) * self._rmodel.nq]
        )

        # Computing the initial residuals
        self._derivative_residual_first_part[
            : self._rmodel.nq, : self._rmodel.nq
        ] = self._derivative_initial_residual

        # Computing the principal residuals
        self._derivative_residual_first_part[
            self._rmodel.nq : -3, :
        ] = self._derivative_principal_residual

        # Computing the terminal residuals
        self._derivative_residual_first_part[
            -3:, -self._rmodel.nq :
        ] = self._derivative_terminal_residual

        ###* COMPUTING THE DERIVATIVES OF THE OBSTACLE RESIDUALS

        # Creating an array to store all the derivatives of the obstacles
        self._derivative_residual_sec_part = np.zeros(
            [len(self._obstacle_residual), (self._T) * self._rmodel.nq]
        )

        # Going through all the configurations
        for t in range(1, self._T):
            # Obtaining the current configuration
            q_t = get_q_iter_from_Q(Q, t, self._nq)

            # Updating the pinocchio models
            pin.framesForwardKinematics(self._rmodel, self._rdata, q_t)
            pin.updateGeometryPlacements(
                self._rmodel, self._rdata, self._cmodel, self._cdata
            )

            iter = 0  # Counter for each robot shape
            # For each configuration, going through all the geometry objects.
            for oMg, geometry_objects in zip(
                self._cdata.oMg, self._cmodel.geometryObjects
            ):
                if self._CAPSULE:
                    # Only selecting the shapes of the robot and not the environement
                    if isinstance(geometry_objects.geometry, hppfcl.Cylinder):
                        # Computing the distance between the given part of the robot and the obstacle
                        for n_obs, (obs_pose, obs_shape) in enumerate(zip(self._OBSTACLE, self._OBSTACLE_SHAPE)):
                            
                            dist = pydiffcol.distance(
                                hppfcl.Capsule(geometry_objects.geometry.radius,geometry_objects.geometry.halfLength),
                                oMg,
                                obs_shape,
                                obs_pose,
                                self._req,
                                self._res,
                            )
                            # If the given part of the robot is too close to the obstacle, a residual is added. 0 Otherwise
                            if dist < self._eps_collision_avoidance:
                                # * Computing the derivative of the obstacle residual
                                # Computing the jacobians in pinocchio
                                pin.computeJointJacobians(self._rmodel, self._rdata, q_t)

                                # Computing the derivatives of the distance
                                _ = pydiffcol.distance_derivatives(
                                    hppfcl.Capsule(geometry_objects.geometry.radius,geometry_objects.geometry.halfLength),
                                    oMg,
                                    obs_shape,
                                    obs_pose,
                                    self._req,
                                    self._res,
                                    self._req_diff,
                                    self._res_diff
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
                                J = (jacobian.T @ self._res_diff.dn_loc_dM1.T).T

                            else:
                                J = np.zeros((3, self._nq))

                            self._derivative_residual_sec_part[
                                (t - 1) * self._number_of_objects * 3
                                + iter * 3 + n_obs * 3 : (t - 1) * self._number_of_objects * 3
                                + (iter + 1) * 3 + n_obs * 3,
                                t * nq : (t + 1) * nq,
                            ] = (
                                J * self._WEIGHT_OBS
                            )

                            iter += 1
                else:
                    # Only selecting the shapes of the robot and not the environement
                    if not isinstance(geometry_objects.geometry, hppfcl.Box):
                        # Computing the distance between the given part of the robot and the obstacle
                        for n_obs, (obs_pose, obs_shape) in enumerate(zip(self._OBSTACLE, self._OBSTACLE_SHAPE)):
            
                            dist = pydiffcol.distance(
                                geometry_objects.geometry,
                                oMg,
                                obs_shape,
                                obs_pose,
                                self._req,
                                self._res,
                            )
                            # If the given part of the robot is too close to the obstacle, a residual is added. 0 Otherwise
                            if dist < self._eps_collision_avoidance:
                                # * Computing the derivative of the obstacle residual
                                # Computing the jacobians in pinocchio
                                pin.computeJointJacobians(self._rmodel, self._rdata, q_t)

                                # Computing the derivatives of the distance
                                _ = pydiffcol.distance_derivatives(
                                    geometry_objects.geometry,
                                    oMg,
                                    obs_shape,
                                    obs_pose,
                                    self._req,
                                    self._res,
                                    self._req_diff,
                                    self._res_diff
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
                                J = (jacobian.T @ self._res_diff.dn_loc_dM1.T).T

                            else:
                                J = np.zeros((3, self._nq))

                            self._derivative_residual_sec_part[
                                (t - 1) * self._number_of_objects * 3
                                + iter * 3 + n_obs * 3 : (t - 1) * self._number_of_objects * 3
                                + (iter + 1) * 3 + n_obs * 3,
                                t * nq : (t + 1) * nq,
                            ] = (
                                J * self._WEIGHT_OBS
                            )

                            iter += 1

        # Creating the derivative residual array which is the concatenation of the 2 derivative residual arrays.
        self._derivative_residual = np.zeros(
            (
                len(self._residual),
                (self._T) * self._rmodel.nq,
            )
        )

        self._derivative_residual[
            : (self._T) * self._rmodel.nq + 3, :
        ] = self._derivative_residual_first_part
        self._derivative_residual[
            (self._T) * self._rmodel.nq + 3 :, :
        ] = self._derivative_residual_sec_part

        self.gradval = self._derivative_residual.T @ self._residual

        return self.gradval

    def hess(self, Q: np.ndarray):
        """Returns the hessian of the cost function with regards to the gauss newton approximation"""
        self._Q = Q
        self.cost(self._Q)
        self.grad(self._Q)
        self.hessval = self._derivative_residual.T @ self._derivative_residual

        return self.hessval

    def grad_numdiff(self, Q: np.ndarray):
        return numdiff(self.cost, Q)

    def hess_numdiff(self, Q: np.ndarray):
        return numdiff(self.grad_numdiff, Q)


if __name__ == "__main__":
    pass
