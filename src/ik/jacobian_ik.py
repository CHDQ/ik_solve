import math
from functools import partial

import torch
from torch.autograd.functional import jacobian

from src.ik.ik_chain import IKChain
from src.ik.ik_sovler import IKSolver
from src.ik.utils import vector_rot_q, q2matrix, r2q, vector_norm, matrix2q, q2r, mul_q
import numpy as np


class JacobianIK(IKSolver):
    def __init__(self, chain: IKChain, tolerance: float = 0.01, max_iter: int = 20):
        super(JacobianIK, self).__init__(chain, tolerance, max_iter)
        self.rot_matrix = None
        self.rot_euler = None
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
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

    def constrain(self, i):
        cons = self.chain.node_list[i].constraint
        if cons is None:
            return
        self.rot_euler[i] = np.clip(self.rot_euler[i], a_min=np.gradient(cons.lower_limit), a_max=np.gradient(cons.upper_limit))

    def add_rot(self, theta_list):
        for i, theta in enumerate(theta_list):
            self.rot_euler[i] = self.rot_euler[i] + theta_list[i]
            self.constrain(i)
            theta = torch.from_numpy(self.rot_euler[i]).float()
            matrix = self.euler2matrix(theta).detach().numpy()
            self.rot_matrix[i] = matrix

    def euler2matrix(self, theta):
        """
        旋转角度转旋转矩阵  theta ->(x,y,z)
        :param theta:
        :return:
        """
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
        """
        单独骨骼旋转 theta->(x,y,z) point-(x,y,z)
        :param theta:
        :param point:
        :return:
        """
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
        return j_t.view((3, -1)).T.numpy()

    def b_euler2matrix(self, theta):
        """
        批量角度转旋转矩阵 batch*(x,y,z)
        :param theta:
        :return:
        """
        s = len(theta)
        matrix = torch.ones(s, 4, 4, dtype=torch.float, device=self.device) * torch.eye(4, dtype=torch.float, device=self.device)
        rx = torch.ones(s, 3, 3, dtype=torch.float, device=self.device) * torch.eye(3, dtype=torch.float, device=self.device)
        ry = torch.ones(s, 3, 3, dtype=torch.float, device=self.device) * torch.eye(3, dtype=torch.float, device=self.device)
        rz = torch.ones(s, 3, 3, dtype=torch.float, device=self.device) * torch.eye(3, dtype=torch.float, device=self.device)
        c0 = torch.cos(theta[:, 0])
        c1 = torch.cos(theta[:, 1])
        c2 = torch.cos(theta[:, 2])
        s0 = torch.sin(theta[:, 0])
        s1 = torch.sin(theta[:, 1])
        s2 = torch.sin(theta[:, 2])
        rx[:, 1, 1] = c0
        rx[:, 1, 2] = -1 * s0
        rx[:, 2, 1] = s0
        rx[:, 2, 2] = c0
        ry[:, 0, 0] = c1
        ry[:, 0, 2] = s1
        ry[:, 2, 0] = -1 * s1
        ry[:, 2, 2] = c1
        rz[:, 0, 0] = c2
        rz[:, 0, 1] = -1 * s2
        rz[:, 1, 0] = s2
        rz[:, 1, 1] = c2
        matrix[:, :3, :3] = torch.matmul(ry, torch.matmul(rx, rz))
        return matrix

    def b_rotate(self, point, theta):
        f"""
        批量旋转{point}->batch*(x,y,z) theta->batch*(x,y,z)
        :param point: 
        :param theta: 
        :return: 
        """
        matrix = self.b_euler2matrix(theta)
        p = torch.ones(len(point), 4, dtype=torch.float, device=self.device)
        p[:, :3] = torch.from_numpy(point).to(self.device)
        y = torch.matmul(matrix, p.unsqueeze(2)).squeeze()
        return y

    def calc_jacobian(self, theta, point):
        """
        利用pytorch求解jacobian矩阵
        :param theta:
        :param point:
        :return:
        """
        theta = torch.from_numpy(theta).float()
        theta = theta.to(self.device)
        fn = partial(self.b_rotate, point)
        j_t = jacobian(fn, theta)
        j_t = j_t.transpose(0, 1)[:3, ...]
        b = torch.ones(j_t.shape[:-1], dtype=torch.long) * torch.eye(3)
        b = b.unsqueeze(-1)
        b = b.expand(-1, -1, -1, 3)
        j_t = torch.masked_select(j_t, b.bool())
        j = j_t.reshape(len(theta), 3, 3).T
        j = j.reshape(3, 3 * len(theta))
        return j

    def jacobian_ik(self, target):
        head, tail, _ = self.calc_bone_pos()
        vector = (tail[-1] - head)
        # 循环骨骼求jacobian
        ############################################################
        # vector = (tail[-1] - head)[:, np.newaxis, ...]
        # j = None
        # for i in range(len(self.rot_euler)):
        #     if j is None:
        #         j = self.rotate(self.rot_euler[i], vector[i])
        #     else:
        #         j = np.concatenate((j, self.rotate(self.rot_euler[i], vector[i])), axis=1)
        # vector = vector.squeeze(1)
        #############################################################
        # pytorch 批量求jacobian
        j = self.calc_jacobian(self.rot_euler, vector)
        j_p = torch.pinverse(j).cpu()
        v = torch.tensor((target - tail[-1])).float()
        theta = torch.matmul(j_p, v)
        dp = self.calc_constrain_d(-theta.reshape(-1, 3) * self.step)
        y = torch.matmul((torch.eye(j.shape[1]) - torch.matmul(j_p, j)), dp)
        theta = theta + y
        theta = -theta * self.step
        return theta.reshape(-1, 3).numpy()

    def calc_constrain_d(self, theta, alpha=math.radians(2), k=0.1):
        """
        参考论文
        A Realistic Joint Limit Algorithm for Kinematically Redundant Manipulators
        :param theta:
        :param alpha:
        :param k:
        :return:
        """
        d = torch.tensor([], dtype=torch.float, device=self.device)
        for i, node in enumerate(self.chain.node_list):
            constrain = node.constraint
            euler = torch.tensor(self.rot_euler[i])
            e = constrain.upper_limit - constrain.lower_limit
            molecular_min = euler + theta[i] - (constrain.lower_limit + alpha)
            molecular_max = euler + theta[i] - (constrain.upper_limit - alpha)
            dt_min = 2 * molecular_min / e
            dt_max = 2 * molecular_max / e
            p1 = 1 / 2 - torch.sign(molecular_min).float() / 2
            p2 = 1 / 2 + torch.sign(molecular_max).float() / 2
            d = torch.cat((d,( p1 * dt_min + p2 * dt_max).float()))
        return -d * k

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
        self.max_iter = 100
        while loss > self.tolerance and epoch < self.max_iter:
            theta_list = self.jacobian_ik(target)
            self.add_rot(theta_list)
            head, tail, orientation_list = self.calc_bone_pos()
            loss = np.linalg.norm(tail[-1] - target)
            epoch = epoch + 1
            print("\r" + "\t" * 10, end="")
            print("\r loss: %s, epoch: %s" % (loss, epoch), end="")
        self.update_visible(head, orientation_list)
        print()
