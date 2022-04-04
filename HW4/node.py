from tkinter import Canvas
from utils import distance

POINT_RADIUS = 1

class Node:
    def __init__(self, position):
        self._position = position
        self._cost = 0
        self._parent = None
        self._nodes = []
        self._outside = False

    def get_cost(self):
        return self._cost

    def set_cost(self, cost):
        self._cost = cost

    def set_parent(self, node: "Node"):
        self._parent = node

    def set_outside_state(self, state: bool):
        self._outside = state

    def remove_node(self, node: 'Node'):
        self._nodes.remove(node)

    def reconnect(self, node: 'Node'):
        self._parent.remove_node(self)
        self._parent = node
        self._parent.add_node(self)

    def add_node(self, node: "Node"):
        node.set_parent(self)
        self._nodes.append(node)

    def print(self, canvas: Canvas):
        # x0,y0,x1,y1
        # draw all tree   
        canvas.create_oval(self._position[0] - POINT_RADIUS, self._position[1] - POINT_RADIUS,
                           self._position[0] + POINT_RADIUS, self._position[1] + POINT_RADIUS,
                           fill="red" if self._outside else "green")

        for node in self._nodes:
            self.print_line(canvas, self._position, node.get_position())
            node.print(canvas)

    def print_path_to_root(self, canvas):
        if self._parent:
            self.print_line(canvas, self._position, self._parent.get_position(), fill="red", width=2.0)
            self._parent.print_path_to_root(canvas)

    def get_position(self):
        return self._position

    @staticmethod
    def print_line(canvas, first_position, last_position, fill="black", width=1.0):
        # x0,y0,x1,y1
        canvas.create_line(first_position[0], first_position[1],
                           last_position[0], last_position[1],
                           fill=fill, width=width)

    @staticmethod
    def nodes_distance(first_node, second_node):
        return distance(first_node.get_position(), second_node.get_position())