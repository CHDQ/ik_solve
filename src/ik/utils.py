import math

import numpy as np
import transformations


def vector_norm(vector: np.ndarray):
    return vector / (np.linalg.norm(vector) + 1e-7)


def vector_rot_q(initial, target):
    initial_norm = vector_norm(initial)
    target_norm = vector_norm(target)
    normal_vector = np.cross(initial_norm, target_norm)
    theta = math.acos(np.dot(initial_norm, target_norm))
    cos_half = math.cos(theta / 2)
    sin_half = math.sin(theta / 2)
    rotate_q = [cos_half, normal_vector[0] * sin_half, normal_vector[1] * sin_half, normal_vector[2] * sin_half]
    return rotate_q


def q2r(q, axes="rzxy"):
    return transformations.euler_from_quaternion(q, axes)


def r2q(rad, axes="rzxy"):
    return transformations.quaternion_from_euler(*rad, axes)


def axis_rot2q(vector, theta):
    return transformations.quaternion_from_matrix(transformations.rotation_matrix(theta, vector))


def sub_q(q0, q1):
    q0 = np.array(q0)
    q1 = np.array(q1)
    q0 = q0 / (np.linalg.norm(q0) + 1e-7)
    q1 = q1 / (np.linalg.norm(q1) + 1e-7)
    return transformations.quaternion_multiply(transformations.quaternion_conjugate(q1), q0)


def mul_q(q0, q1):
    return transformations.quaternion_multiply(q0, q1)


def rot_point(q, p):
    p = np.append(p, 1)
    return np.dot(transformations.quaternion_matrix(q), p)[:-1]


def q2matrix(q):
    return transformations.quaternion_matrix(q)


def matrix2q(matrix):
    return transformations.quaternion_from_matrix(matrix)


def r2matrix(euler, axis="sxyz"):
    return transformations.euler_matrix(*euler, axis)
