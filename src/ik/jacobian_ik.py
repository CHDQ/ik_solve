import torch

from src.ik.ik_chain import IKChain
from src.ik.ik_sovler import IKSolver
from src.ik.utils import vector_rot_q, q2matrix, r2q, vector_norm
import numpy as np


class JacobianIK(IKSolver):
    def __init__(self, chain: IKChain, tolerance: float = 0.01, max_iter: int = 20):
        super(JacobianIK, self).__init__(chain, tolerance, max_iter)
        self.l_rot_matrix = None
        self.init_joint_tensor()
        self.step = 0.05

    def init_joint_tensor(self):
        l_rot_matrix = []
        for node in self.chain.node_list:
            l_rot_matrix.append(q2matrix(vector_rot_q(node.init_orientation, node.orientation)))
        self.l_rot_matrix = np.array(l_rot_matrix)

    def calc_bone_pos(self):
        head = [[0, 0, 0]]
        orientation_list = []
        g_rotate = []
        for i, node in enumerate(self.chain.node_list):
            rot = np.eye(4)
            if i > 0:
                rot = g_rotate[i - 1]
            rot = np.dot(self.l_rot_matrix[i], rot)
            g_rotate.append(rot)
            orientation = vector_norm(np.dot(rot, np.append(node.init_orientation, 1))[:3])
            orientation_list.append(orientation)
            pos = node.node_length * orientation + head[i]
            head.append(pos)
        return np.array(head[:-1]).copy(), np.array(head[1:]).copy(), np.array(orientation_list)

    def add_rot(self, theta_list):
        for i, theta in enumerate(theta_list):
            self.l_rot_matrix[i] = np.dot(q2matrix(r2q(theta)), self.l_rot_matrix[i])

    def jacobian_ik(self, target):
        head, tail, _ = self.calc_bone_pos()
        vector = tail - head
        velocity = self.l_rot_matrix[:, :3, :3]
        j_t = np.cross(vector, velocity).reshape(-1, 3)
        theta = np.dot(j_t, target - tail[-1]) * self.step
        return theta.reshape(-1, 3)

    def update_visible(self, head, orientation_list):
        for i, node in enumerate(self.chain.node_list):
            node.ik_position = head[i]
            node.ik_vector = orientation_list[i]
            node.update_trans_matrix()

    def solve(self, target):
        target = np.array(target)
        loss = 100
        epoch = 0
        head = None
        orientation_list = None
        while loss > self.tolerance and epoch < self.max_iter:
            theta_list = self.jacobian_ik(target)
            self.add_rot(theta_list)
            head, tail, orientation_list = self.calc_bone_pos()
            loss = np.linalg.norm(tail[-1] - target)
            epoch = epoch + 1
            print("\r loss: %s, epoch: %s" % (loss, epoch), end="")
        self.update_visible(head, orientation_list)
        print()
