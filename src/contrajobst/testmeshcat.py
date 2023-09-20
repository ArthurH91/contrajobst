import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import example_robot_data as robex

# Creation of the robot
robot = robex.load("ur10")
rmodel = robot.model
rdata = rmodel.createData()


# # Create visualizers for both robots
viz = MeshcatVisualizer(rmodel, robot.visual_model, robot.visual_model)
viz2 = MeshcatVisualizer(rmodel, robot.visual_model, robot.visual_model)

# Creating the viewers for the robot at the same URL
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6001")
viz2.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6001")

# Loading the models
viz.loadViewerModel(rootNodeName="number 1")
viz2.loadViewerModel(rootNodeName="number 2")

# Displaying random configurations of the robots
viz.display(pin.randomConfiguration(rmodel))
viz2.display(pin.randomConfiguration(rmodel))
