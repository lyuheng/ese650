from tkinter import Canvas
from math import sqrt
from node import Node


class Obstacle:
    def __init__(self, radius, center):
        self._radius = radius
        self._center = center

    def is_point_in(self, node: Node, safety=1.0):
        position = node.get_position()
        return (((position[0] - self._center[0]) ** 2 + (position[1] - self._center[1]) ** 2)
                < ((self._radius ** 2) * safety))

    def does_line_intersect(self, start_position, end_position, safety=1.5):
        """
        return if a line intersect with this obstacle
        """
        radius = self._radius * safety
        x_c, y_c = self._center
        x_l0, y_l0 = start_position
        x_l1, y_l1 = end_position
        x_l1 = x_l1 - x_l0
        y_l1 = y_l1 - y_l0

        c = (x_c ** 2) + (y_c ** 2) + (x_l0 ** 2) + (y_l0 ** 2) - (2 * ((x_c * x_l0) + (y_c * y_l0))) - (radius ** 2)
        b = 2 * ((x_l0 * x_l1) + (y_l0 * y_l1) - (x_l1 * x_c) - (y_l1 * y_c))
        a = (y_l1 ** 2) + (x_l1 ** 2)

        det = (b ** 2) - (4 * a * c)

        if det >= 0:
            if a != 0:
                square_det = sqrt(det)
                if 0 <= (-b + square_det) / (2 * a) <= 1:
                    return True
                if 0 <= (-b - square_det) / (2 * a) <= 1:
                    return True
                return False
            else:
                return True
        else:
            return False

    def print(self, canvas: Canvas):
        canvas.create_oval(self._center[0] - self._radius, self._center[1] - self._radius,
                           self._center[0] + self._radius, self._center[1] + self._radius,
                           fill="red")
