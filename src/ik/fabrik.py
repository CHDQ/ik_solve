from functools import reduce

from src.ik.ik_chain import *


class FABRIK:
    def __init__(self, chain: IKChain, tolerance: float = 0.01, max_iter: int = 20):
        self.chain = chain  # ik链
        self.tolerance = tolerance  # 容忍度  距离接近范围内就认为到达目标
        self.max_iter = max_iter  # 最大迭代次数
        self.chain_length = reduce(lambda x, y: x + y.node_length, [0, ] + self.chain.node_list)
        self.chain_size = len(self.chain.node_list)

    def solve(self, target):
        """

        :param target: position
        :return:
        """
        t_s_length = np.linalg.norm(target - self.chain.node_list[0].position)
        if t_s_length > self.chain_length:
            vector = target - self.chain.node_list[0].position
            vector = vector_norm(vector)
            self.chain.node_list[0].ik_vector = self.constrain(self.chain.node_list[0].init_orientation, vector,
                                                               self.chain.node_list[0].constraint)
            self.chain.node_list[0].update_trans_matrix()
            for i in range(1, len(self.chain.node_list)):
                p_node = self.chain.node_list[i - 1]
                c_node = self.chain.node_list[i]
                c_node.ik_position = p_node.position + p_node.ik_vector * p_node.node_length
                c_node.ik_vector = self.constrain(c_node.init_orientation, vector, c_node.constraint)
                c_node.update_trans_matrix()
        else:
            chains = self.chain.node_list.copy()
            chains.append(BoneNode(target, np.ones((3,))))
            chains.insert(0, BoneNode(chains[0].position, chains[0].orientation))
            for i in range(self.max_iter):
                self.backward(chains)
                distance = self.forward(chains)
                if distance < self.tolerance:
                    break
            for c_node in chains[1: self.chain_size + 1]:
                c_node.update_trans_matrix()

    def forward(self, chains) -> float:
        for i in range(1, self.chain_size + 2):
            if i == 1:
                chains[i].ik_position = chains[i - 1].ik_position
                continue
            orientation = vector_norm(chains[i].ik_position - chains[i - 1].ik_position)
            orientation = self.constrain(chains[i - 1].init_orientation, orientation, chains[i - 1].constraint)
            chains[i - 1].ik_vector = orientation
            if i < self.chain_size + 1:
                chains[i].ik_position = chains[i - 1].ik_position + orientation * chains[i - 1].node_length
        return np.linalg.norm(chains[-1].ik_position - chains[-2].ik_position - chains[-2].node_length)

    def backward(self, chains):
        for i in range(self.chain_size, 0, -1):
            orientation = vector_norm(chains[i].ik_position - chains[i + 1].ik_position)
            orientation = -self.constrain(chains[i].init_orientation, -orientation, chains[i].constraint)
            chains[i].ik_vector = orientation
            chains[i].ik_position = chains[i + 1].ik_position + orientation * chains[i].node_length

    @staticmethod
    def constrain(init_orientation: np.ndarray, ik_vector: np.ndarray, constraint: Constraint) -> np.ndarray:
        if constraint is None:
            return ik_vector
        rot_norm = vector_norm(np.cross(init_orientation, ik_vector))
        if np.any(np.isnan(rot_norm)):
            return ik_vector
        euler = math.acos(np.clip(np.dot(init_orientation, ik_vector), -1, 1))
        sin_half = math.sin(euler / 2)
        cos_half = math.cos(euler / 2)
        rotate_q = [cos_half, rot_norm[0] * sin_half, rot_norm[1] * sin_half, rot_norm[2] * sin_half]
        euler_rad = list(transformations.euler_from_quaternion(rotate_q))
        flag = False
        for i, item in enumerate("xyz"):
            scope = getattr(constraint, item)
            if scope is None:
                continue
            if scope[0] > euler_rad[i]:
                euler_rad[i] = scope[0]
                flag = True
                continue
            if scope[1] < euler_rad[i]:
                euler_rad[i] = scope[1]
                flag = True
        if not flag:
            return ik_vector
        return vector_norm(np.dot(transformations.euler_matrix(*euler_rad), np.append(init_orientation, 1))[:-1])
