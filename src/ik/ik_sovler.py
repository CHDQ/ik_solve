from src.ik.ik_chain import IKChain


class IKSolver:
    def __init__(self, chain: IKChain, tolerance: float = 0.01, max_iter: int = 20):
        self.chain = chain  # ik链
        self.tolerance = tolerance  # 容忍度  距离接近范围内就认为到达目标
        self.max_iter = max_iter  # 最大迭代次数

    def solve(self, target):
        pass
