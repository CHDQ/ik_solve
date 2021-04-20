import numpy as np
from tqdm import tqdm

from src.ik.ik_chain import IKChain
from src.ik.ik_sovler import IKSolver
from src.ik.utils import *


class JacobianIK(IKSolver):
    def __init__(self, chain: IKChain, tolerance: float = 0.01, max_iter: int = 200):
        super(JacobianIK, self).__init__(chain, tolerance, max_iter)
        self.step = 0.005

    def orientation2matrix(self):
        joints = []
        for i, node in enumerate(reversed(self.chain.node_list)):
            q = vector_rot_q(node.init_orientation, node.ik_vector)
            matrix = q2matrix(q)
            joints.append({
                "matrix": matrix,
                "index": self.chain.node_list[len(self.chain.node_list) - i - 1],
                "head": node.ik_position,
                "tail": np.matmul(matrix[:3, :3], node.ik_position) + node.ik_position,
                "length": node.node_length,
                "angle": matrix2r(matrix)
            })
        return joints

    def compute_jacobian(self, joints):
        tail = joints[0]["tail"]
        j = np.zeros((3, 3 * len(self.chain.node_list)), dtype=np.float)
        for i, matrix in enumerate(joints):
            diff = tail - self.chain.node_list[joints[i]["index"]]
            for m in range(3):
                j[:, i * 3 + m] = np.cross(vector_norm(matrix[:, m]).T, diff)
        return j

    def calc_jacobian_ik_task(self, target):
        joints = self.orientation2matrix()
        jac = self.compute_jacobian(joints)
        jac_pinv = np.linalg.pinv(jac)
        theta = np.matmul(jac_pinv, target - joints[0]["tail"])
        return theta.reshape((-1, 3))

    def update_ik(self, theta):
        joints = self.orientation2matrix()
        tail = None
        for i, joint in enumerate(reversed(joints)):
            theta = joint["angle"] + theta[i]
            matrix = r2matrix(theta)
            vector = vector_norm(np.matmul(matrix[:3, :3], joint["length"]))
            self.chain.node_list[joint["index"]].ik_vector = vector
            if i > 0:
                self.chain.node_list[joint["index"]].ik_position = self.chain.node_list[joint["index"] - 1].ik_position + vector * joint["length"]
            tail = self.chain.node_list[joint["index"]].ik_position
        return tail

    def update_ui(self):
        pass

    def solve(self, target):
        with tqdm(range(self.max_iter)) as it:
            for epoch in it:
                angle = self.calc_jacobian_ik_task(target)
                tail = self.update_ik(angle)
                loss = np.linalg.norm(np.abs(target, tail))
                it.set_postfix(epoch=epoch, loss=loss)
                if loss < self.tolerance:
                    break
        self.update_ui()
