import math

import torch

from src.ik.ik_chain import IKChain
from src.ik.ik_sovler import IKSolver
from src.ik.utils import vector_rot_q, q2matrix, r2q, vector_norm, matrix2q, q2r, mul_q
import numpy as np


class JacobianIK(IKSolver):
    def __init__(self, chain: IKChain, tolerance: float = 0.01, max_iter: int = 20):
        super(JacobianIK, self).__init__(chain, tolerance, max_iter)
        self.rot_matrix = None
        self.rot_euler = None
        self.init_joint_tensor()
        self.step = 0.05

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

    def add_rot(self, theta_list):
        for i, theta in enumerate(theta_list):
            self.rot_euler[i] = self.rot_euler[i] + theta_list[i]
            matrix = q2matrix(r2q(self.rot_euler[i]))
            theta = torch.from_numpy(self.rot_euler[i]).float()
            matrix = self.euler2matrix(theta).detach().numpy()
            self.rot_matrix[i] = matrix

    def euler2matrix(self, theta):
        rx = torch.eye(3, dtype=torch.float)
        rx[1, 1] = torch.cos(theta[0])
        rx[1, 2] = -torch.sin(theta[0])
        rx[2, 1] = torch.sin(theta[0])
        rx[2, 2] = torch.cos(theta[0])
        ry = torch.eye(3, dtype=torch.float)
        ry[0, 0] = torch.cos(theta[1])
        ry[0, 2] = torch.sin(theta[1])
        ry[2, 0] = -torch.sin(theta[1])
        ry[2, 2] = torch.cos(theta[1])
        rz = torch.eye(3, dtype=torch.float)
        rz[0, 0] = torch.cos(theta[2])
        rz[0, 1] = -torch.sin(theta[2])
        rz[1, 0] = torch.sin(theta[2])
        rz[1, 1] = torch.cos(theta[2])
        matrix = torch.eye(4, dtype=torch.float)
        matrix[:3, :3] = torch.matmul(ry, torch.matmul(rx, rz))
        return matrix

    def rotate(self, theta, point):
        theta = torch.from_numpy(theta).float()
        theta.requires_grad = True
        matrix = self.euler2matrix(theta)
        p = torch.ones(4, dtype=torch.float)
        p[:3] = torch.from_numpy(point)
        y = torch.matmul(matrix, p)
        j_t = torch.tensor([])
        Weight = torch.eye(4, dtype=torch.float)
        for i, weight in enumerate(Weight[:3]):
            j_t = torch.cat((j_t, torch.autograd.grad(y, theta, grad_outputs=weight, retain_graph=True)[0]), 0)
        # print(j_t.view((3, -1)))
        return j_t.view((3, -1)).T.numpy()

    def jacobian_ik(self, target):
        head, tail, _ = self.calc_bone_pos()
        vector = (tail[-1] - head)[:, np.newaxis, ...]
        j = None
        for i in range(len(self.rot_euler)):
            if j is None:
                j = self.rotate(self.rot_euler[i], vector[i])
            else:
                j = np.concatenate((j, self.rotate(self.rot_euler[i], vector[i])), axis=1)
        # j = np.cross(velocity, vector).transpose((0, 2, 1)).reshape((3, -1))
        j_p = np.linalg.pinv(j)
        theta = np.dot(j_p, (target - tail[-1])) * self.step * -1
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
        self.max_iter = 200
        while loss > self.tolerance and epoch < self.max_iter:
            theta_list = self.jacobian_ik(target)
            self.add_rot(theta_list)
            head, tail, orientation_list = self.calc_bone_pos()
            loss = np.linalg.norm(tail[-1] - target)
            epoch = epoch + 1
            print("\r loss: %s, epoch: %s" % (loss, epoch), end="")
        self.update_visible(head, orientation_list)
        print()
