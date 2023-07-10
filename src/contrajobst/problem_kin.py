from os.path import dirname, join, abspath

import numpy as np
import pinocchio as pin
import hppfcl

from wrapper_robot import RobotWrapper
import pydiffcol

# This class is for defining the optimization problem and computing the cost function, its gradient and hessian.


class ProblemInverseKinematics:
    def __init__(
        self,
        rmodel: pin.Model,
        rdata: pin.Data,
        gmodel: pin.GeometryModel,
        gdata: pin.GeometryData,
        target: np.ndarray,
        target_shape: hppfcl.ShapeBase,
        obstacle: np.ndarray,
        obstacle_shape: hppfcl.ShapeBase,
    ):
        """Initialize the class with the models and datas of the robot.

        Parameters
        ----------
        rmodel : pin.Model
            Model of the robot
        rdata : pin.Data
            Data of the model of the robot
        gmodel : pin.GeometryModel
            Geometrical model of the robot
        gdata : pin.GeometryData
            Geometrical data of the model of the robot
        target : pin.SE3
            Pose of the target
        target_shape : hppfcl.ShapeBase
            Shape of the target

        """
        self._rmodel = rmodel
        self._rdata = rdata
        self._gmodel = gmodel
        self._gdata = gdata
        self._target = target
        self._target_shape = target_shape
        self._obstacle = obstacle
        self._obstacle_shape = obstacle_shape

        # Storing the IDs of the frames of the end effector and the target

        self._EndeffID = self._rmodel.getFrameId("panda2_leftfinger")
        self._EndeffID_geom = self._gmodel.getGeometryId("panda2_leftfinger_0")
        assert self._EndeffID < len(self._rmodel.frames)
        assert self._EndeffID_geom < len(self._gmodel.geometryObjects)

    def cost(self, q: np.ndarray):
        """Compute the cost of the configuration q. The cost is quadratic here.

        Parameters
        ----------
        q : np.ndarray
            Array of configuration of the robot, size rmodel.nq.

        Returns
        -------
        cost : float
            Cost of the configuration q.
        """

        # Distance request for pydiffcol
        self._req = pydiffcol.DistanceRequest()
        self._res = pydiffcol.DistanceResult()

        self._req.derivative_type = pydiffcol.DerivativeType.FirstOrderRS

        # Forward kinematics of the robot at the configuration q.
        pin.framesForwardKinematics(self._rmodel, self._rdata, q)
        pin.updateGeometryPlacements(
            self._rmodel, self._rdata, self._gmodel, self._gdata, q
        )
        ###* COMPUTING THE DISTANCE BETWEEN THE END EFFECTOR AND THE TARGET
        # Obtaining the cartesian position of the end effector.
        self.endeff_Transform = self._rdata.oMf[self._EndeffID]
        self.endeff_Shape = self._gmodel.geometryObjects[self._EndeffID_geom].geometry

        ###* Creating the residual array
        self._residual = np.zeros(6)

        # Computing the distance with pydiffcol
        distance_end_effector_target = pydiffcol.distance(
            self.endeff_Shape,
            self.endeff_Transform,
            self._target_shape,
            self._target,
            self._req,
            self._res,
        )

        self._residual[:3] = self._res.w

        ###* COMPUTING THE DISTANCE BETWEEN THE END EFFECTOR AND THE OBSTACLE

        distance_end_efector_obstacle = pydiffcol.distance(
            self.endeff_Shape,
            self.endeff_Transform,
            self._obstacle_shape,
            self._obstacle,
            self._req,
            self._res,
        )

        # If the end effector is close to the obstacle, a cost is added
        eps_obstacle = 1e-5
        if distance_end_efector_obstacle < eps_obstacle:
            self._residual[3:] = self._res.w

        return 0.5 * np.linalg.norm(self._residual) ** 2

    def grad(self, q: np.ndarray):
        """Compute the gradient of the cost function at a configuration q.

        Parameters
        ----------
        q : np.ndarray
            Array of configuration of the robot, size rmodel.nq.

        Returns
        -------
        gradient cost : np.ndarray
            Gradient cost of the robot at the end effector at a configuration q, size rmodel.nq.
        """

        # Computing the cost to initialize all the variables
        self.cost(q)

        # Computing the jacobians in pinocchio
        pin.computeJointJacobians(self._rmodel, self._rdata, q)

        # Getting the frame jacobian from the end effector in the LOCAL reference frame
        self._jacobian = pin.computeFrameJacobian(
            self._rmodel, self._rdata, q, self._EndeffID, pin.LOCAL
        )

        # Computing the derivatives of the distance of the end effector to the target
        _ = pydiffcol.distance_derivatives(
            self.endeff_Shape,
            self.endeff_Transform,
            self._target_shape,
            self._target,
            self._req,
            self._res,
        )
        self._derivative_residuals = self._jacobian.T @ self._res.dw_dq1.T

        # Computing the derivatives of the distance of the end effector to the target
        _ = pydiffcol.distance_derivatives(
            self.endeff_Shape,
            self.endeff_Transform,
            self._obstacle_shape,
            self._obstacle,
            self._req,
            self._res,
        )

        self._derivative_residuals = np.hstack(
            (self._derivative_residuals, self._jacobian.T @ self._res.dw_dq1.T)
        )
        return self._derivative_residuals @ self._residual

    def hessian(self, q: np.ndarray):
        """Returns hessian matrix of the end effector at a q position

        Parameters
        ----------
        q : np.ndarray
            Array of the configuration of the robot

        Returns
        -------
        Hessian matrix : np.ndaraay
            Hessian matrix at a given q configuration of the robot
        """

        self.grad(q)
        return self._derivative_residuals @ self._derivative_residuals.T


if __name__ == "__main__":
    from utils import generate_reachable_target, numdiff
    from wrapper_meshcat import MeshcatWrapper

    def grad_numdiff(q: np.ndarray):
        return numdiff(QP.cost, q)

    def hess_numdiff(q: np.ndarray):
        return numdiff(grad_numdiff, q)

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

    # Generating a target
    TARGET = pin.SE3.Identity()
    TARGET.translation = np.array([-0.25, 0, 1.6])

    # Generating an initial configuration
    q = pin.neutral(rmodel)
    pin.framesForwardKinematics(rmodel, rdata, q)

    # THIS STEP IS MANDATORY OTHERWISE THE FRAMES AREN'T UPDATED
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    # Creating the visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        TARGET,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
    )

    vis.display(q)

    # The target shape is a ball of 5e-2 radii at the TARGET position

    TARGET_SHAPE = hppfcl.Sphere(5e-2)

    QP = ProblemInverseKinematics(rmodel, rdata, cmodel, cdata, TARGET, TARGET_SHAPE)

    res = QP.cost(q)
    print(res)
    gradval = QP.grad(q)
    hessval = QP.hessian(q)

    gradval_numdiff = grad_numdiff(q)
    hessval_numdiff = hess_numdiff(q)
