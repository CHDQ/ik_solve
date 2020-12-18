import random
import time

import numpy as np
import pyvista as pv

from src.ik.fabrik import FABRIK
from src.ik.ik_chan import *

update_grid = False


class BoneVisible:
    def __init__(self, ik_solver: FABRIK):
        self.chain = ik_solver.chain
        self.ik_solver = ik_solver
        self.arrow = []
        self.target = None
        self.plotter = None
        self.init_vista()

    def init_vista(self):
        plotter = pv.Plotter()
        self.plotter = plotter
        for node in self.chain.node_list:
            arrow = pv.Arrow(node.position, node.orientation, scale=node.node_length)
            self.arrow.append(arrow)
            plotter.add_mesh(arrow, show_edges=True, color=self.random_color())
        tail = self.chain.node_list[-1]
        target_position = tail.position + tail.orientation * tail.node_length
        self.target = pv.Sphere(radius=0.1, center=target_position)
        plotter.add_mesh(self.target, color="tan")
        self.set_shortcuts()

    def set_shortcuts(self):
        self.plotter.add_key_event("1", self.move_target("z"))
        self.plotter.add_key_event("2", self.move_target("z", reverse=True))
        self.plotter.add_key_event("4", self.move_target("x"))
        self.plotter.add_key_event("5", self.move_target("x", reverse=True))
        self.plotter.add_key_event("7", self.move_target("y"))
        self.plotter.add_key_event("8", self.move_target("y", reverse=True))

    def update(self):
        self.ik_solver.solve(self.target.center)
        for i, node in enumerate(self.chain.node_list):
            self.arrow[i].transform(node.transform_matrix)
            node.transform_matrix = np.eye(4)
        self.plotter.show_grid()

    def move_target(self, axis="x", step=0.05, reverse=False):
        xyz = ["x", "y", "z"]
        step = step * -1 if reverse else step

        def handle():
            move_step = [0, 0, 0]
            move_step[xyz.index(axis)] = step
            self.target.translate(move_step)
            self.update()
            self.plotter.show_grid()

        return handle

    def visible(self):
        self.plotter.show_grid()
        self.plotter.show(auto_close=True)

    def random_color(self):
        color = [random.random(), random.random(), random.random()]
        return color


if __name__ == "__main__":
    ik_chain = IKChain()
    constraint = Constraint([math.radians(-100), math.radians(100)], [math.radians(-100), math.radians(100)],
                            [math.radians(-100), math.radians(100)])
    ik_chain.append(BoneNode([0, 0, 0], [1, 0, 0], constraint=constraint))
    ik_chain.append(BoneNode([1, 0, 0], [1, 0, 0], constraint=constraint))
    ik_chain.append(BoneNode([2, 0, 0], [1, 0, 0], constraint=constraint))
    ik = FABRIK(ik_chain)
    BoneVisible(ik).visible()
