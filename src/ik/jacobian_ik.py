import math

import numpy as np
from tqdm import tqdm

from src.ik.ik_chain import IKChain
from src.ik.ik_sovler import IKSolver
from src.ik.utils import *


class JacobianIK(IKSolver):
    def __init__(self, chain: IKChain, tolerance: float = 0.001, max_iter: int = 200, step: int = 0.05):
        super(JacobianIK, self).__init__(chain, tolerance, max_iter)
        self.step = step

    def orientation2matrix(self):
        joints = []
        for i, node in enumerate(reversed(self.chain.node_list)):
            q = vector_rot_q(node.init_orientation, node.ik_vector)
            matrix = q2matrix(q)
            joints.append({
                "matrix": matrix,
                "index": len(self.chain.node_list) - i - 1,
                "head": node.ik_position,
                "orientation": node.init_orientation,
                "tail": node.ik_vector * node.node_length + node.ik_position,
                "length": node.node_length,
                "angle": node.ik_joint
            })
        return joints

    def compute_jacobian(self, joints):
        j = np.zeros((3, 3 * len(self.chain.node_list)), dtype=np.float)
        joint_pos = joints[0]["tail"]
        for i, joint in enumerate(joints):
            current_pos = joint["head"]
            matrix = joint["matrix"]
            diff = (joint_pos - current_pos)
            for m in range(3):
                j[:, i * 3 + m] = np.cross(vector_norm(matrix[:3, m]), diff)
        return j

    def calc_jacobian_ik_task(self, target):
        joints = self.orientation2matrix()
        jac = self.compute_jacobian(joints)
        jac_pinv = np.linalg.pinv(jac)
        # jac_pinv = jac.T
        theta = np.matmul(jac_pinv, vector_norm(target - joints[0]["tail"]) * self.step)
        return theta.reshape((-1, 3))

    def update_ik(self, angle):
        joints = self.orientation2matrix()
        tail = None
        angle = list(reversed(angle))
        for i, joint in enumerate(reversed(joints)):
            theta = joint["angle"] + angle[i]
            self.chain.node_list[joint["index"]].ik_joint = theta
            matrix = r2matrix(theta)
            vector = vector_norm(np.matmul(matrix[:3, :3], joint["length"] * joint["orientation"]))
            self.chain.node_list[joint["index"]].ik_vector = vector
            if i > 0:
                parent_node = self.chain.node_list[i - 1]
                self.chain.node_list[i].ik_position = parent_node.ik_position + parent_node.ik_vector * parent_node.node_length
            tail = self.chain.node_list[i].ik_position + vector * joint["length"]
        return tail

    def update_ui(self):
        for i in self.chain.node_list:
            i.update_trans_matrix()

    def solve(self, target):
        with tqdm(range(self.max_iter), ncols=80) as it:
            for epoch in it:
                angle = self.calc_jacobian_ik_task(target)
                tail = self.update_ik(angle)
                loss = np.linalg.norm(np.abs(target - tail))
                it.set_postfix(epoch=epoch, loss=loss)
                if loss < self.tolerance:
                    break
        self.update_ui()
