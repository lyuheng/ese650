from tkinter import Tk, Frame, Button, Canvas, Label, StringVar, LEFT, ALL
from random import randint
from math import inf
from node import Node
from obs_rec import Obstacle
import numpy as np
import matplotlib.pyplot as plt
#from ellipse import Ellipse

CANVAS_SIZE = [600,600]
MAX_LOOP = 10000
#ARRIVAL_RADIUS = 0.5


class RRStarAlgo:
    def __init__(self, init_position, final_position, safety_radius: float=18):
        self._init_position = init_position
        self._final_position = final_position
        self._safety_radius = safety_radius
        self._tree = Node(init_position)
        self._nodes = [self._tree]
        self._obstacles = [Obstacle([120,420],[300,450]), Obstacle([420,30],[450,420])]
        self._arrival = None
        self._counter = 0

    def step(self):
        new_node = self.create_random_node()
        if not self.is_valid_node(new_node):
            return

        best_node, near_nodes = self.choose_parent(new_node)
        if not best_node:
            return

        self.insert_node(best_node, new_node)

        if not near_nodes == []:
            near_nodes.remove(best_node)
            self.rewire(near_nodes, new_node)

        if self.is_final_node(new_node):
            self.update_arrival_node(new_node)

    def choose_parent(self, new_node):
        best_node = None
        near_nodes = self.find_near(new_node, 800*(np.log(len(self._nodes))/len(self._nodes))**(1/2) )
        best_cost = None

        for node in near_nodes:
            cost = node.get_cost() + Node.nodes_distance(node, new_node)

            if (best_cost is None) or (cost < best_cost):

                if self.does_line_intersect(new_node.get_position(), node.get_position()):
                    continue

                best_cost = cost
                best_node = node

        if not best_node:
            nearest_node = self.find_nearest(new_node)
            if not self.does_line_intersect(nearest_node.get_position(), new_node.get_position()):
                best_node = nearest_node

        return best_node, near_nodes

    def insert_node(self, best_node, new_node):
        new_node.set_cost(best_node.get_cost() + Node.nodes_distance(best_node, new_node))
        best_node.add_node(new_node)
        self._nodes.append(new_node)


    def rewire(self, near_nodes, new_node):
        if near_nodes:
            new_code_cost = new_node.get_cost()
            for node in near_nodes:
                if node.get_cost() > (new_code_cost + Node.nodes_distance(node, new_node)):
                    if self.does_line_intersect(new_node.get_position(), node.get_position()):
                        continue

                    node.set_cost(new_code_cost + Node.nodes_distance(node, new_node))
                    node.reconnect(new_node)

    def does_line_intersect(self, start_position, end_position):
        for obstacle in self._obstacles:
            if obstacle.does_line_intersect(start_position, end_position, self._safety_radius):
                return True
        return False

    def print_tree(self, canvas: Canvas):
        self._tree.print(canvas)

    def print_obstacles(self, canvas: Canvas):
        for obstacle in self._obstacles:
            obstacle.print(canvas)

    def print_arrival(self, canvas: Canvas):
        if self._arrival:
            position = self._final_position
            canvas.create_oval(position[0] - 15, position[1] - 15,
                               position[0] + 15, position[1] + 15,
                               fill="blue")

    def print_solution(self, canvas: Canvas):
        if self._arrival:
            self._arrival.print_path_to_root(canvas)

    def find_nearest(self, node: Node):
        return min(self._nodes, key=lambda x: Node.nodes_distance(x, node))

    def find_near(self, node: Node, radius):
        return list(filter(lambda x: Node.nodes_distance(x, node) < radius, self._nodes))

    def is_valid_node(self, new_node: Node):
        for obstacle in self._obstacles:
            if obstacle.is_point_in(new_node, self._safety_radius):
                return False

        # if self._ellipse:
        #     if not self._ellipse.is_point_in(new_node.get_position()):
        #         return False

        return True

    def is_final_node(self, node: Node):
        position = node.get_position()
        return (((position[0] - self._final_position[0]) ** 2 + (position[1] - self._final_position[1]) ** 2)
                < (15 ** 2))

    def update_arrival_node(self, node: Node):
        if self._arrival:
            if self._arrival.get_cost() > node.get_cost():
                self._arrival = node
        else:
            self._arrival = node


    def create_random_node(self):
        return Node([randint(0, CANVAS_SIZE[0]), randint(0, CANVAS_SIZE[1])])

    """
    def update_rec(self):
        ###!!!
        self._ellipse = Ellipse(self._init_position, self._final_position, self._arrival.get_cost() / 2)
        self._ellipse_cost_limit = self._arrival.get_cost()

        self._outside_nodes = self._outside_nodes + list(filter(lambda node: False == self._ellipse.is_point_in(node.get_position()), self._nodes))
        for node in self._outside_nodes:
            node.set_outside_state(True)
        self._nodes = list(filter(lambda node: self._ellipse.is_point_in(node.get_position()), self._nodes))
    """


class RRTStarUI:
    def __init__(self, master):
        self.master = master
        master.title("RRT*")

        self.canvas = Canvas(master, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1])
        self.canvas.pack()

        self.canvas.create_rectangle(0, 0, CANVAS_SIZE[0], CANVAS_SIZE[1], fill="grey")

        self.frame = Frame(master)
        self.frame.pack()

        self.close_button = Button(self.frame, text="Close", command=master.quit)
        self.close_button.pack(side=LEFT)

        self.cost_string = StringVar()
        self.cost_string.set("Infinity")
        self.cost_label = Label(self.frame, textvariable=self.cost_string)
        self.cost_label.pack(side=LEFT)

        self._loop_counter = 0

        self.algo = RRStarAlgo([300,300], [540,60])

        self.res = [] # divided 30
        self.cyclic_call()

    def cyclic_call(self):

        for _ in range(1):
            self.algo.step()
            self._loop_counter += 1
        """
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(0, 0, CANVAS_SIZE[0], CANVAS_SIZE[1], fill="grey")
        self.algo.print_tree(self.canvas)
        self.algo.print_obstacles(self.canvas)
        self.algo.print_arrival(self.canvas)
        self.algo.print_solution(self.canvas)
        """
        
        if self._loop_counter < MAX_LOOP:
            if self._loop_counter % 200 == 0:
                print(self._loop_counter)
            if self.algo._arrival:
                self.res.append(self.algo._arrival.get_cost()/30)
                self.cost_string.set(str(self.algo._arrival.get_cost()))
            else:
                self.res.append(0)
            self.master.after(1, self.cyclic_call)

if __name__ == '__main__':
    """
    root = Tk()
    my_gui = RRTStarUI(root)
    root.mainloop()
    np.save('cost.npy', np.array(my_gui.res))
    """
    cost = np.load('cost.npy')
    s = np.where(cost == 0)[0][-1]+1
    print(s)
    c = cost[s:]
    plt.xlim(left=s, right=10000)
    plt.xticks([543,2000,4000,6000,8000,10000])
    plt.xlabel('iteration')
    plt.ylabel('optimal cost')
    plt.plot(c)
    plt.show()
