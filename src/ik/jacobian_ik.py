from src.ik.ik_chain import IKChain
from src.ik.ik_sovler import IKSolver


class JacobianIK(IKSolver):
    def __init__(self, chain: IKChain, tolerance: float = 0.01, max_iter: int = 200):
        super(JacobianIK, self).__init__(chain, tolerance, max_iter)
        self.step = 0.005
