import math
from functools import partial

import torch
from torch.autograd.functional import jacobian

from src.ik.ik_chain import IKChain
from src.ik.ik_sovler import IKSolver
from src.ik.utils import vector_rot_q, q2matrix, r2q, vector_norm, matrix2q, q2r, mul_q, r2matrix
import numpy as np


class JacobianIK(IKSolver):
    def __init__(self, chain: IKChain, tolerance: float = 0.01, max_iter: int = 200):
        super(JacobianIK, self).__init__(chain, tolerance, max_iter)
        self.rot_matrix = None
        self.rot_euler = None
        self.device = torch.device("cpu")
        self.init_joint_tensor()
        self.step = 0.005

    def init_joint_tensor(self):
        rot_matrix = []
        rot_euler = []
        for node in self.chain.node_list:
            rot_matrix.append(q2matrix(vector_rot_q(node.init_orientation, node.orientation)))
            rot_euler.append(q2r(vector_rot_q(node.init_orientation, node.orientation)))
        self.rot_matrix = np.array(rot_matrix)
        self.rot_euler = np.array(rot_euler)

    def calc_bone_pos(self):
        head = [[0, 0, 0]]
        orientation_list = []
        for i, node in enumerate(self.chain.node_list):
            rot = self.rot_matrix[i]
            orientation = vector_norm(np.dot(rot, np.append(node.init_orientation, 1))[:3])
            orientation_list.append(orientation)
            pos = node.node_length * orientation + head[i]
            head.append(pos)
        return np.array(head[:-1]).copy(), np.array(head[1:]).copy(), np.array(orientation_list)

    def constrain(self, i):
        cons = self.chain.node_list[i].constraint
        if cons is None:
            return
        self.rot_euler[i] = np.clip(self.rot_euler[i], a_min=cons.lower_limit, a_max=cons.upper_limit)

    def add_rot(self, theta_list):
        for i, theta in enumerate(theta_list):
            self.rot_euler[i] = self.rot_euler[i] + theta_list[i]
            self.constrain(i)
            theta = torch.from_numpy(self.rot_euler[i]).float()
            matrix = r2matrix(theta)
            self.rot_matrix[i] = matrix

    def rotate(self, theta, point):
        """
        单独骨骼旋转 theta->(x,y,z) point-(x,y,z)
        :param theta:
        :param point:
        :return:
        """
        theta = torch.from_numpy(theta).float()
        theta.requires_grad = True
        matrix = r2matrix(theta)
        p = torch.ones(4, dtype=torch.float)
        p[:3] = torch.from_numpy(point)
        y = torch.matmul(matrix, p)
        j_t = torch.tensor([])
        Weight = torch.eye(4, dtype=torch.float)
        for i, weight in enumerate(Weight[:3]):
            j_t = torch.cat((j_t, torch.autograd.grad(y, theta, grad_outputs=weight, retain_graph=True)[0]), 0)
        return j_t.view((3, -1)).T.numpy()

    def jacobian_ik(self, target):
        head, tail, _ = self.calc_bone_pos()
        vector = (target - tail)
        j = np.zeros((3, len(head) * 3), dtype=np.float)
        for i, m in enumerate(self.rot_matrix):
            for k in range(3):
                j[:, k + i * 3] = np.cross(vector_norm(m[:3, k]), vector[i])
        j_inv = np.linalg.pinv(j)
        theta = np.matmul(j_inv, torch.from_numpy(target - tail[-1]).float()) * self.step
        return theta.reshape(-1, 3).numpy()

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
            print("\r loss: %s, epoch: %s" % (loss.item(), epoch), end="")
        self.update_visible(head, orientation_list)
        print()
