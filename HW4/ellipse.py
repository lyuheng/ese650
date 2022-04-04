from random import uniform
from math import pi, sqrt, atan2, cos, sin
from utils import distance

from tkinter import Tk, Frame, Button, Canvas, LEFT, ALL
POINT_RADIUS = 1

CANVAS_SIZE = [640, 480]

class Ellipse:
    def __init__(self, first_focus: [], second_focus: [], semi_major_axis: float):
        self._first_focus = first_focus
        self._second_focus = second_focus
        #  得到椭圆的中心坐标
        self._center = [(first_focus[0] + second_focus[0])/2, (first_focus[1] + second_focus[1])/2]
        #  得到椭圆半轴
        self._semi_major_axis = semi_major_axis
        self._semi_minor_axis = sqrt((self._semi_major_axis ** 2) - (distance(self._center, second_focus) ** 2))
        self._semi_major_axis2 = semi_major_axis * 2
        # 得到椭圆的倾斜度
        self._ellipse_angle = atan2(self._center[0] - first_focus[0], self._center[1] - first_focus[1])

    def is_point_in(self, point: []) -> []:
        """
        True: in ellipse
        False: out of ellipse
        """
        return ((distance(self._first_focus, point)
                 + distance(self._second_focus, point)) <= self._semi_major_axis2)

    def random_point_in(self) -> []:
        """
        random sample point in this ellipse
        ''
        """
        angle = uniform(- pi, pi)
        semi_major_axis = uniform(0, self._semi_major_axis)
        semi_minor_axis = uniform(0, self._semi_minor_axis)
        x_rotated = semi_major_axis * cos(angle)
        y_rotated = semi_minor_axis * sin(angle)

        x = self._center[0] + cos(self._ellipse_angle) * (x_rotated) - sin(self._ellipse_angle) * y_rotated
        y = self._center[1] + cos(self._ellipse_angle) * (y_rotated) + sin(self._ellipse_angle) * x_rotated
        return [x, y]


class ShowRandomPointsInEllipse:
    """
    random sample point in this ellipse
    """
    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.canvas = Canvas(master, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])
        self.canvas.pack()

        ellipse = Ellipse([10, 10], [200, 200], 170)

        for _ in range(10000):
            position = ellipse.random_point_in()
            self.canvas.create_oval(position[0] - POINT_RADIUS, position[1] - POINT_RADIUS,
                                    position[0] + POINT_RADIUS, position[1] + POINT_RADIUS,
                                    fill="green")

if __name__ == '__main__':
    root = Tk()
    my_gui = ShowRandomPointsInEllipse(root)
    root.mainloop()

        


