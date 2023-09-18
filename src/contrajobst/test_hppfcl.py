import numpy as np
import copy

import hppfcl
import pinocchio as pin
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

RED = np.array([249, 136, 126, 125]) / 255
RED_FULL = np.array([249, 136, 126, 255]) / 255

GREEN = np.array([170, 236, 149, 125]) / 255
GREEN_FULL = np.array([170, 236, 149, 255]) / 255

BLUE = np.array([144, 169, 183, 125]) / 255
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

YELLOW = np.array([1, 1, 0, 0.5])
YELLOW_FULL = np.array([1, 1, 0, 1.0])

BLACK = np.array([0, 0, 0, 0.5])
BLACK_FULL = np.array([0, 0, 0, 1.0])


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


def create_visualizer():
    """Creation of an empty visualizer.

    Returns
    -------
    vis : Meshcat.Visualizer
        visualizer from meshcat
    """
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viewer.delete()
    return viewer


def rgbToHex(color):
    if len(color) == 4:
        c = color[:3]
        opacity = color[3]
    else:
        c = color
        opacity = 1.0
    hex_color = "0x%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
    return hex_color, opacity


def meshcat_material(r, g, b, a):
    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(r * 255) * 256**2 + int(g * 255) * 256 + int(b * 255)
    material.opacity = a
    return material


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


red = meshcat_material(RED[0], RED[1], RED[2], RED[3])
green = meshcat_material(GREEN[0], GREEN[1], GREEN[2], GREEN[3])
yellow = meshcat_material(YELLOW[0], YELLOW[1], YELLOW[2], YELLOW[3])
blue = meshcat_material(BLUE[0], BLUE[1], BLUE[2], BLUE[3])

if __name__ == "__main__":
    vis = create_visualizer()

    # Shapes
    r1 = 0.5
    shape1 = hppfcl.Sphere(r1)
    r2 = np.array([0.5, 0.3, 1])
    shape2 = hppfcl.Box(r2)

    T1 = pin.SE3.Identity()

    vis["sphere1"].set_object(g.Sphere(r1), red)
    T2 = pin.SE3.Random()

    vis["box"].set_object(g.Box(r2), blue)

    vis["box"].set_transform(get_transform(T2))

    # Computing distance/contact points
    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    dist = hppfcl.distance(shape1, T1, shape2, T2, req, res)
    print(dist)

    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()

    # Vis contact points

    r_w = 0.1

    vis["cp1"].set_object(g.Sphere(r_w), green)
    vis["cp1"].set_transform(tf.translation_matrix(cp1))

    vis["cp2"].set_object(g.Sphere(r_w), yellow)
    vis["cp2"].set_transform(tf.translation_matrix(cp2))

    req_col = hppfcl.CollisionRequest()
    res_col = hppfcl.CollisionResult()
