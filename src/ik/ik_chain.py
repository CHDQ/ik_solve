import math

import numpy as np
import transformations

from src.ik.utils import vector_norm


class Constraint:
    def __init__(self, x: list = None, y: list = None, z: list = None):
        self.x = x
        self.y = y
        self.z = z
        self.lower_limit = None
        self.upper_limit = None
        self.init_limit()

    def init_limit(self):
        self.lower_limit = np.array([self.x[0], self.y[0], self.z[0]])
        self.upper_limit = np.array([self.x[1], self.y[1], self.z[1]])


class BoneNode:
    def __init__(self, position, orientation, node_length=1, constraint: Constraint = None):
        self.init_orientation = vector_norm(orientation)
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.node_length = node_length
        self.constraint = constraint
        self.transform_matrix = np.eye(4)
        self.ik_position = self.position
        self.ik_vector = self.orientation

    def update_trans_matrix(self):
        """
        计算ik之后更新旋转矩阵
        :return:
        """
        rot_norm = vector_norm(np.cross(self.orientation, self.ik_vector))
        m_zero = np.eye(4)
        m_zero[:3, -1] = - self.position
        translate = np.eye(4)
        translate[:3, -1] = self.ik_position
        rotate = np.eye(4)
        if not np.any(np.isnan(rot_norm)):
            cos_a = np.dot(vector_norm(self.orientation), vector_norm(self.ik_vector))
            euler = math.acos(np.clip(cos_a, -1, 1))
            sin_half = math.sin(euler / 2)
            cos_half = math.cos(euler / 2)
            q = [cos_half, rot_norm[0] * sin_half, rot_norm[1] * sin_half, rot_norm[2] * sin_half]
            rotate = np.array(transformations.quaternion_matrix(q))
        self.transform_matrix = np.dot(translate, np.dot(rotate, m_zero))
        self.position = self.ik_position
        self.orientation = self.ik_vector


class IKChain:
    def __init__(self, *nodes: BoneNode):
        self.node_list = list(nodes)

    def append(self, *nodes: BoneNode):
        for node in nodes:
            self.node_list.append(node)
