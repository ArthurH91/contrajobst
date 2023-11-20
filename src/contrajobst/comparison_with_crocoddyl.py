from os.path import dirname, join, abspath

import numpy as np
import pinocchio as pin
import crocoddyl

from solver_newton_mt import SolverNewtonMt
from wrapper_robot import RobotWrapper
from problem_traj_without_obs import NLP_without_obs

# ### HYPERPARMS
T = 6
WEIGHT_Q0 = 5
WEIGHT_DQ = 10
WEIGHT_TERM_POS = 10

# TARGET POSE
TARGET = np.array([-0.155, -0.815, 0.456])
# INITIAL CONFIG OF THE ROBOT
INITIAL_CONFIG = np.array([0, -2.5, 2, -1.2, -1.7, 0,0])

# Creation of the robot 

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

# Initial trajectory
Q0 = np.concatenate([INITIAL_CONFIG] * (T + 1))

# 
dt = 1e-3
x0 = np.concatenate([INITIAL_CONFIG, pin.utils.zero(rmodel.nv)])
X0 = ([x0]*(T+1))
u0 = np.zeros(7)
U0 = ([u0] * T)
############################################################# HOME MADE SOVLER #########################################
# Creating the QP
NLP = NLP_without_obs(
    rmodel,
    cmodel,
    q0=INITIAL_CONFIG,
    target=TARGET,
    T=T,
    weight_q0=WEIGHT_Q0,
    weight_dq=WEIGHT_DQ,
    weight_term_pos=WEIGHT_TERM_POS,
)


########################################################## CROCODYL ####################################################

# Stat and actuation model
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFull(state)

# Running & terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

### Creation of cost terms

# State Regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# End effector frame cost

framePlacementResidual = crocoddyl.ResidualModelFrameTranslation(
    state,
    rmodel.getFrameId("panda2_joint7"),
    TARGET,
)
goalTrackingCost = crocoddyl.CostModelResidual(
    state, framePlacementResidual
)

# Adding costs to the models
runningCostModel.addCost("stateReg", xRegCost, WEIGHT_DQ**2)
terminalCostModel.addCost(
    "gripperPose", goalTrackingCost, WEIGHT_TERM_POS**2
)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
    state, actuation, runningCostModel
)
terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
    state, actuation, terminalCostModel
)

# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
runningModel = crocoddyl.IntegratedActionModelEuler(
    running_DAM, dt
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    terminal_DAM, 0.0
)

runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])


problem = crocoddyl.ShootingProblem(
    x0, [runningModel] * T, terminalModel
)
# Create solver + callbacks
ddp = crocoddyl.SolverFDDP(problem)

ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])


#### SOLVING #####

ddp.solve(X0, U0, maxiter = 500)
# Trust region solver
trust_region_solver = SolverNewtonMt(
    NLP.cost,
    NLP.grad,
    NLP.hess,
    max_iter=100,
    callback=None,
    verbose=True,
    bool_plot_results=False,
    eps=1e-9,
)
trust_region_solver(Q0[7:])


####################################################### RESULTS #############################################################
print(f"NLP.cost(Q0) : {NLP.cost(Q0)}")
print(f"ddp.problem.calc(X0,U0) : {ddp.problem.calc(X0, U0)}")



print(f"ddp.problem.calc(ddp.xs.tolist(),ddp.us.tolist()) : {ddp.problem.calc(ddp.xs.tolist(),ddp.us.tolist())}")
print(f"NLP.cost(trust_region_solver._xval_k) : {NLP.cost(trust_region_solver._xval_k)}")


