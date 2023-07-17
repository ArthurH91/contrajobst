import hppfcl
import pinocchio as pin
import numpy as np
import time
import copy


RED = np.array([249, 136, 126, 125]) / 255


def get_transform(T_: hppfcl.Transform3f):
    T = np.eye(4)
    if isinstance(T_, hppfcl.Transform3f):
        T[:3, :3] = T_.getRotation()
        T[:3, 3] = T_.getTranslation()
    elif isinstance(T_, pin.SE3):
        T[:3, :3] = T_.rotation
        T[:3, 3] = T_.translation
    else:
        raise NotADirectoryError
    return T


def get_q_iter_from_Q(Q: np.ndarray, iter: int, nq: int):
    """Returns the iter-th configuration vector q_iter in the Q array.

    Args:
        Q (np.ndarray): Optimization vector.
        iter (int): Index of the q_iter desired.
        nq (int): size of q_iter

    Returns:
        q_iter (np.ndarray): Array of the configuration of the robot at the iter-th step.
    """
    q_iter = np.array((Q[nq * iter : nq * (iter + 1)]))
    return q_iter


def get_difference_between_q_iter(Q: np.ndarray, iter: int, nq: int):
    """Returns the difference between the q_iter+1 and q_iter in the array self.Q

    Parameters
    ----------
    Q : np.ndarray
        Optimization vector.
    iter : int
        Index of the q_iter desired.
    nq : int
        Length of a configuration vector.

    Returns:
        q_iter+1 - q_iter (np.ndarray): Difference of the arrays of the configuration of the robot at the iter-th and ither +1 -th steps.

    """
    return get_q_iter_from_Q(Q, iter + 1, nq) - get_q_iter_from_Q(Q, iter, nq)


def display_last_traj(vis, Q: np.ndarray, q0: np.ndarray, T: int, dt=None):
    """Display the trajectory computed by the solver

    Parameters
    ----------
    vis : Meshcat.Visualizer
        Meshcat visualizer
    Q : np.ndarray
        Optimization vector.
    q0 : np.ndarray
        Initial configuration vector
    nq : int
        size of q_iter
    """
    for q_iter in [q0] + np.split(Q, T + 1):
        vis.display(q_iter)
        if dt is None:
            input()
        else:
            time.sleep(dt)


def display_last_traj_with_obstacle_moving(
    vis, meshcatvis, Q: np.ndarray, q0: np.ndarray, T: int, theta_list, TARGET, dt=None
):
    """Display the trajectory computed by the solver

    Parameters
    ----------
    vis : Meshcat.Visualizer
        Meshcat visualizer
    Q : np.ndarray
        Optimization vector.
    q0 : np.ndarray
        Initial configuration vector
    nq : int
        size of q_iter
    """
    for q_iter in [q0] + np.split(Q, T + 1):
        vis.display(q_iter)
        for theta in theta_list:
            OBSTACLE_translation = TARGET.translation / 2 + [
                0.2 + theta,
                0 + theta,
                0.8 + theta,
            ]
            rotation = np.identity(3)
            rotation[1, 1] = 0
            rotation[2, 2] = 0
            rotation[1, 2] = -1
            rotation[2, 1] = 1
            OBSTACLE_rotation = rotation
            OBSTACLE = TARGET.copy()
            OBSTACLE.translation = OBSTACLE_translation
            OBSTACLE.rotation = OBSTACLE_rotation
            meshcatvis["obstacle"].set_transform(get_transform(OBSTACLE))
            time.sleep(dt)
        if dt is None:
            input()
        else:
            time.sleep(dt)


def numdiff(f, x, eps=1e-8):
    """Estimate df/dx at x with finite diff of step eps

    Parameters
    ----------
    f : function handle
        Function evaluated for the finite differente of its gradient.
    x : np.ndarray
        Array at which the finite difference is calculated
    eps : float, optional
        Finite difference step, by default 1e-6

    Returns
    -------
    jacobian : np.ndarray
        Finite difference of the function f at x.
    """
    xc = np.copy(x)
    f0 = np.copy(f(x))
    res = []
    for i in range(len(x)):
        xc[i] += eps
        res.append(copy.copy(f(xc) - f0) / eps)
        xc[i] = x[i]
    return np.array(res).T


def generate_reachable_target(
    rmodel, rdata=None, frameName="endeff", returnConfiguration=False
):
    """
    Sample a random configuration, then returns the forward kinematics
    for this configuration rdata.oMf[frameId].
    If rdata is None, create it on the flight (warning emitted)
    """
    q_target = pin.randomConfiguration(rmodel)

    # Creation of a temporary model.Data, to have access to the forward kinematics.
    if rdata is None:
        rdata = rmodel.createData()
        print("Warning: pin.Data create for a simple kinematic, please avoid")

    # Updating the model.Data with the framesForwardKinematics
    pin.framesForwardKinematics(rmodel, rdata, q_target)

    # Get and check Frame Id
    fid = rmodel.getFrameId(frameName)
    assert fid < len(rmodel.frames)

    if returnConfiguration:
        return rdata.oMf[fid].copy(), q_target
    return rdata.oMf[fid].copy()


if __name__ == "__main__":
    import example_robot_data as robex

    robot = robex.load("ur10")
    p = generate_reachable_target(robot.model, robot.data, "tool0")

    assert np.all(np.isfinite(p.translation))
