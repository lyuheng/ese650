from tkinter import Canvas
from math import sqrt
from utils import distance
from node import Node

import numpy as np

class Obstacle:
    def __init__(self, lefttop, rightdown):
        self.lefttop = lefttop
        self.rightdown = rightdown
        self.length = rightdown[0] - lefttop[0]
        self.width = rightdown[1] - lefttop[1]
        
    def is_point_in(self, node: Node, safety_radius):
        pos = node.get_position()

        if pos[0] >= self.lefttop[0] and pos[0] <= self.rightdown[0] and \
            pos[1] >= self.lefttop[1]-safety_radius and pos[1] <= self.rightdown[1]+safety_radius:
            return True
        
        if pos[1]>=self.lefttop[1] and pos[1]<=self.rightdown[1] and \
            pos[0] >= self.lefttop[0]-safety_radius and pos[0] <= self.rightdown[0]+safety_radius:
            return True

        if distance(pos,self.lefttop) <= safety_radius or distance(pos,self.rightdown) <= safety_radius or \
            distance(pos,[self.lefttop[0],self.lefttop[1]+self.width]) <= safety_radius or \
                distance(pos, [self.rightdown[0],self.rightdown[1]-self.width]) <= safety_radius:
                return True
        if pos[0] <= 18 or pos[0] >=582 or pos[1] <= 18 or pos[1]>=582:
             return True
        return False

    def does_line_intersect(self, start_position, end_position, safety_radius):
        """
        return if a line intersect with this obstacle
        do discretization, step = 3
        """
        # TODO: use line intersection
        dis = distance(start_position,end_position)
        dire = [end_position[0]-start_position[0], end_position[1]-start_position[1]]
        if dis < 0.1:
            start_node = Node(start_position)
            if self.is_point_in(start_node, safety_radius) == True:
                return True
            end_node = Node(end_position)
            if self.is_point_in(end_node, safety_radius) == True:
                return True
            return False

        dire = [dire[0]/dis, dire[1]/dis]
        
        num = (int)(np.floor(dis/0.1)) # step=0.1
        for i in range(num+1):
            new_p = [start_position[0]+dire[0]*0.1*i, start_position[1]+dire[1]*0.1*i]
            new_node = Node(new_p)
            if self.is_point_in(new_node, safety_radius) == True:
                return True
        return False


    def print(self, canvas: Canvas):
        # canvas.create_oval(self._center[0] - self._radius, self._center[1] - self._radius,
        #                    self._center[0] + self._radius, self._center[1] + self._radius,
        #                    fill="red")
        canvas.create_rectangle(self.lefttop[0], self.lefttop[1],
                            self.rightdown[0], self.rightdown[1], fill="red")
