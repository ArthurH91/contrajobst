import numpy as np
import hppfcl
import copy
import pinocchio as pin

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from utils import get_transform


class MeshcatWrapper:
    """Wrapper displaying a robot and a target in a meshcat server."""

    def __init__(self, grid=False, axes=False):
        """Wrapper displaying a robot and a target in a meshcat server.

        Parameters
        ----------
        grid : bool, optional
            Boolean describing whether the grid will be displayed or not, by default False
        axes : bool, optional
            Boolean describing whether the axes will be displayed or not, by default False
        """

        self._grid = grid
        self._axes = axes

    def visualize(self, TARGET: pin.SE3,OBSTACLE: pin.SE3,  RADII_TARGET=5e-2, OBSTACLE_DIM = [5e-1,5e-1,5e-2], robot=None):
        """Returns the visualiser, displaying the robot and the target if they are in input.

        Parameters
        ----------
        TARGET : pin.SE3
            pin.SE3 describing the position of the target
        RADII_TARGET : float, optional
            radii of the target which is a ball, by default 5e-2
        robot : robot, optional
            robot from example robot data for instance, by default None

        Returns
        -------
        vis : MeshcatVisualizer
            visualizer from Meshcat
        """
        if robot is not None:
            self._robot = robot

            self._TARGET = TARGET
            self._RADII_TARGET = RADII_TARGET

            self._OBSTACLE = OBSTACLE
            self._OBSTACLE_DIM = OBSTACLE_DIM

            # Creating the models of the robot
            self._rmodel = self._robot.model
            self._rcmodel = self._robot.collision_model
            self.rvmodel = self._robot.visual_model

        # Creation of the visualizer,
        self.viewer = self.create_visualizer()

        if self._TARGET is not None:
            # Creating the target, which is here a hppfclSphere
            self._renderSphere("target")

        if self._OBSTACLE is not None:
            # Creating the obstacle
            self._renderBox("obstacle")

        Viewer = pin.visualize.MeshcatVisualizer

        if robot is not None:
            self.viewer = Viewer(robot.model, robot.collision_model, robot.visual_model)
        self.viewer.initViewer(
            viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        )
        self.viewer.loadViewerModel()

        return self.viewer

    def create_visualizer(self):
        """Creation of an empty visualizer.

        Returns
        -------
        vis : Meshcat.Visualizer
            visualizer from meshcat
        """
        self.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.viewer.delete()
        if not self._grid:
            self.viewer["/Grid"].set_property("visible", False)
        if not self._axes:
            self.viewer["/Axes"].set_property("visible", False)
        return self.viewer

    def _renderSphere(self, e_name: str, color=np.array([1.0, 1.0, 1.0, 1.0])):
        """Displaying a sphere in a meshcat server.

        Parameters
        ----------
        e_name : str
            name of the object displayed
        color : np.ndarray, optional
            array describing the color of the target, by default np.array([1., 1., 1., 1.]) (ie white)
        """
        # Setting the object in the viewer
        self.viewer[e_name].set_object(
            g.Sphere(self._RADII_TARGET), self._meshcat_material(*color)
        )

        # Obtaining its position in the right format
        T = get_transform(self._TARGET)

        # Applying the transformation to the object
        self.viewer[e_name].set_transform(T)

    def _renderBox(self, e_name: str, color=np.array([1.0, 1.0, 1.0, 1.0])):
        """Displaying a sphere in a meshcat server.

        Parameters
        ----------
        e_name : str
            name of the object displayed
        color : np.ndarray, optional
            array describing the color of the target, by default np.array([1., 1., 1., 1.]) (ie white)
        """
        # Setting the object in the viewer
        self.viewer[e_name].set_object(
            g.Box(self._OBSTACLE_DIM), self._meshcat_material(*color)
        )

        # Obtaining its position in the right format
        T = get_transform(self._OBSTACLE)

        # Applying the transformation to the object
        self.viewer[e_name].set_transform(T)

                

    def _meshcat_material(self, r, g, b, a):
        """Converting RGBA color to meshcat material.

        Parameters
        ----------
        r : _type_
            _description_
        g : _type_
            _description_
        b : _type_
            _description_
        a : _type_
            _description_

        Returns
        -------
        material : meshcat.geometry.MeshPhongMaterial()
            material for meshcat
        """
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(r * 255) * 256**2 + int(g * 255) * 256 + int(b * 255)
        material.opacity = a
        return material


if __name__ == "__main__":
    from utils import generate_reachable_target
    from wrapper_robot import RobotWrapper

    # Generating the robot
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper()
    rdata = rmodel.createData()

    # Generate a reachable target
    p = generate_reachable_target(rmodel, rdata, "tool0")

    # Generate a reachable obstacle
    l_translation = p.translation/2
    l_rotation = np.identity(3)
    l = p.copy()
    l.translation = l_translation
    l.rotation = l_rotation
 
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(p, l, robot=robot)
