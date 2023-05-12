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


class QuadratricProblemNLP:
    def __init__(
        self,
        robot,
        rmodel: pin.Model,
        gmodel: pin.GeometryModel,
        q0: np.array,
        target: np.array,
        target_shape: hppfcl.ShapeBase,
        T: int,
        weight_q0: float,
        weight_dq: float,
        weight_term_pos: float,
    ):
        """Initialize the class with the models and datas of the robot.

        Parameters
        ----------
        robot : pin.Robot
            Model of the robot, used for robot.q0
        rmodel : pin.Model
            Model of the robot
        q0: np.array
            Initial configuration of the robot
        target: np.array
            Target position for the end effector
        target_shape: hppfcl.ShapeBase
            hppfcl.ShapeBase of the target
        T : int
            Number of steps for the trajectory
        weight_q0 : float
            Factor of penalisation of the initial cost (q_0 - q0)**2
        weight_dq : float
            Factor of penalisation of the running cost (q_t+1 - q_t)**2
        weight_term_pos : float
            Factor of penalisation of the terminal cost (p(q_T) - target)**
        """
        self._robot = robot
        self._rmodel = rmodel
        self._gmodel = gmodel
        self._rdata = rmodel.createData()
        self._gdata = gmodel.createData()

        self._q0 = q0
        self._T = T
        self._target = target
        self._target_shape = target_shape
        self._weight_q0 = weight_q0
        self._weight_dq = weight_dq
        self._weight_term_pos = weight_term_pos

        # Storing the IDs of the frame of the end effector

        self._EndeffID = self._rmodel.getFrameId("endeff")
        self._EndeffID_geom = self._gmodel.getGeometryId("endeff_geom")
        assert self._EndeffID_geom < len(self._gmodel.geometryObjects)
        assert self._EndeffID < len(self._rmodel.frames)

    def cost(self, Q: np.ndarray):
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

        ### INITIAL RESIDUAL
        ### Computing the distance between q0 and q_init to make sure the robot starts at the right place
        self._initial_residual = (
            get_q_iter_from_Q(self._Q, 0, self._rmodel.nq) - self._q0
        )

        # Penalizing the initial residual
        self._initial_residual *= self._weight_q0

        ### RUNNING RESIDUAL
        ### Running residuals are computed by diffenciating between q_th and q_th +1
        self._principal_residual = (
            get_difference_between_q_iter(Q, 0, self._rmodel.nq) * self._weight_dq
        )
        for iter in range(1, self._T):
            self._principal_residual = np.concatenate(
                (
                    self._principal_residual,
                    get_difference_between_q_iter(Q, iter, self._rmodel.nq)
                    * self._weight_dq,
                ),
                axis=None,
            )

        ### TERMINAL RESIDUAL
        ### Computing the distance between the last configuration and the target

        # Distance request for pydiffcol
        self._req = pydiffcol.DistanceRequest()
        self._res = pydiffcol.DistanceResult()

        self._req.derivative_type = pydiffcol.DerivativeType.FirstOrderRS

        # Obtaining the last configuration of Q
        q_last = get_q_iter_from_Q(self._Q, self._T, self._rmodel.nq)

        # Forward kinematics of the robot at the configuration q.
        pin.framesForwardKinematics(self._rmodel, self._rdata, q_last)
        pin.updateGeometryPlacements(
            self._rmodel, self._rdata, self._gmodel, self._gdata, q_last
        )

        # Obtaining the cartesian position of the end effector.
        self.endeff_Transform = self._rdata.oMf[self._EndeffID]
        self.endeff_Shape = self._gmodel.geometryObjects[self._EndeffID_geom].geometry

        #
        dist_endeff_target = pydiffcol.distance(
            self.endeff_Shape,
            self.endeff_Transform,
            self._target_shape,
            self._target,
            self._req,
            self._res,
        )

        self._terminal_residual = (self._weight_term_pos) * self._res.w

        ### TOTAL RESIDUAL
        self._residual = np.concatenate(
            (self._initial_residual, self._principal_residual, self._terminal_residual),
            axis=None,
        )

        ### COMPUTING COSTS
        self._initial_cost = 0.5 * sum(self._initial_residual**2)
        self._principal_cost = 0.5 * sum(self._principal_residual**2)
        self._terminal_cost = 0.5 * sum(self._terminal_residual**2)
        self.costval = self._initial_cost + self._terminal_cost + self._principal_cost

        return self.costval

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
        self._derivative_initial_residual = np.diag([self._weight_q0] * self._rmodel.nq)

        # Computing the derivative of the principal residual
        nq, T = self._rmodel.nq, self._T
        J_principal = np.zeros((T * nq, (T + 1) * nq))
        np.fill_diagonal(J_principal, -self._weight_dq)
        np.fill_diagonal(J_principal[:, nq:], self._weight_dq)

        self._derivative_principal_residual = J_principal

        # Computing the derivative of the terminal residual

        q_terminal = get_q_iter_from_Q(self._Q, self._T, self._rmodel.nq)

        # Computing the jacobians in pinocchio
        pin.computeJointJacobians(self._rmodel, self._rdata, q_terminal)

        # Computing the derivatives of the distance
        _ = pydiffcol.distance_derivatives(
            self.endeff_Shape,
            self.endeff_Transform,
            self._target_shape,
            self._target,
            self._req,
            self._res,
        )

        # Getting the frame jacobian from the end effector in the LOCAL reference frame
        jacobian = pin.computeFrameJacobian(
            self._rmodel, self._rdata, q_terminal, self._EndeffID, pin.LOCAL
        )

        # The jacobian here is the multiplication of the jacobian of the end effector and the jacobian of the distance between the end effector and the target
        J = jacobian.T @ self._res.dw_dq1.T
        self._derivative_terminal_residual = self._weight_term_pos * J.T

        # Putting them all together
        T, nq = self._T, self._rmodel.nq

        self._derivative_residual = np.zeros(
            [(self._T + 1) * self._rmodel.nq + 3, (self._T + 1) * self._rmodel.nq]
        )

        # Computing the initial residuals
        self._derivative_residual[
            : self._rmodel.nq, : self._rmodel.nq
        ] = self._derivative_initial_residual

        # Computing the principal residuals
        self._derivative_residual[
            self._rmodel.nq : -3, :
        ] = self._derivative_principal_residual

        # Computing the terminal residuals
        self._derivative_residual[
            -3:, -self._rmodel.nq :
        ] = self._derivative_terminal_residual

        self.gradval = self._derivative_residual.T @ self._residual
        
        gradval_numdiff = self.grad_numdiff(Q)
        # print(f"grad val : {np.linalg.norm(self.gradval)} \n grad val numdiff : {np.linalg.norm(gradval_numdiff)}")
        # assert np.linalg.norm(self.gradval - gradval_numdiff, np.inf) < 1e-4
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
    from utils import numdiff

    # Setting up the environnement
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper()
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    q = pin.randomConfiguration(rmodel)

    pin.framesForwardKinematics(rmodel, rdata, q)

    # THIS STEP IS MANDATORY OTHERWISE THE FRAMES AREN'T UPDATED
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)

    q0 = np.array([1, 1, 1, 1, 1, 1])
    q1 = np.array([2.1, 2.1, 2.1, 2.1, 2.1, 2.1])
    q2 = np.array([3.3, 3.3, 3.3, 3.3, 3.3, 3.3])
    q3 = np.array([4, 4, 4, 4, 4, 4])

    Q = np.concatenate((q0, q1, q2, q3))
    T = int((len(Q) - 1) / rmodel.nq)
    p = pin.SE3.Random()

    # The target shape is a ball of 5e-2 radii at the TARGET position
    TARGET_SHAPE = hppfcl.Sphere(5e-2)

    QP = QuadratricProblemNLP(
        robot,
        rmodel,
        gmodel,
        q0=q,
        target=p,
        target_shape=TARGET_SHAPE,
        T=T,
        weight_q0=5,
        weight_dq=0.1,
        weight_term_pos=10,
    )

    QP._Q = Q

    def grad_numdiff(Q: np.ndarray):
        return numdiff(QP.cost, Q)

    def hess_numdiff(Q: np.ndarray):
        return numdiff(grad_numdiff, Q)

    cost = QP.cost(Q)
    grad = QP.grad(Q)
    gradval_numdiff = grad_numdiff(Q)
    hessval_numdiff = hess_numdiff(Q)
    print(np.linalg.norm(grad - gradval_numdiff, np.inf))
    assert np.linalg.norm(grad - gradval_numdiff, np.inf) < 1e-4
