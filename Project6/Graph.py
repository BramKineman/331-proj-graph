"""
Name:
CSE 331 FS20 (Onsay)
"""

import heapq
import itertools
import math
import queue
import random
import time
from typing import TypeVar, Callable, Tuple, List, Set

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')    # Graph Class Instance


class Vertex:
    """ Class representing a Vertex object within a Graph """

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, idx: str, x: float = 0, y: float = 0) -> None:
        """
        DO NOT MODIFY
        Initializes a Vertex
        :param idx: A unique string identifier used for hashing the vertex
        :param x: The x coordinate of this vertex (used in a_star)
        :param y: The y coordinate of this vertex (used in a_star)
        """
        self.id = idx
        self.adj = {}             # dictionary {id : weight} of outgoing edges
        self.visited = False      # boolean flag used in search algorithms
        self.x, self.y = x, y     # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        DO NOT MODIFY
        Equality operator for Graph Vertex class
        :param other: vertex to compare
        """
        if self.id != other.id:
            return False
        elif self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        elif self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        elif self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        elif set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        Represents Vertex object as string.
        :return: string representing Vertex object
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]

        return f"<id: '{self.id}'" + ", Adjacencies: " + "".join(lst) + ">"

    def __str__(self) -> str:
        """
        DO NOT MODIFY
        Represents Vertex object as string.
        :return: string representing Vertex object
        """
        return repr(self)

    def __hash__(self) -> int:
        """
        DO NOT MODIFY
        Hashes Vertex into a set; used in unit tests
        :return: hash value of Vertex
        """
        return hash(self.id)

#============== Modify Vertex Methods Below ==============#

    def degree(self) -> int:
        """
        Determines the number of outgoing edges from the current vertex.
        :return: The integer number of edges leaving the current vertex.
        """
        return len(self.adj)

    def get_edges(self) -> Set[Tuple[str, float]]:
        """
        Determines the edges as tuples containing the id of the destination vertex and
        the weight of the edge.
        :return: A set of tuples representing outgoing edges from the current vertex.
        """
        my_set = set()
        for vertex_id, weight in self.adj.items():
            # create a set with Set[Tuple[id, weight]]
            current_tuple = vertex_id, weight
            my_set.add(current_tuple)
        return my_set

    def euclidean_distance(self, other: Vertex) -> float:
        """
        Determines the euclidean distance between the current vertex and other vertex.
        :param other: Vertex to determine distance to.
        :return: Euclidean distance between two vertices as a float.
        """
        return math.sqrt(math.pow((other.x - self.x), 2) + math.pow(other.y - self.y, 2))

    def taxicab_distance(self, other: Vertex) -> float:
        """
        Determines the taxicab distance between the current vertex and other vertex.
        :param other: Vertex to determine distance to.
        :return: taxicab distance between two vertices as a float.
        """
        return abs(self.y - other.y) + abs(self.x - other.x)


class Graph:
    """ Class implementing the Graph ADT using an Adjacency Map structure """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csv: str = "") -> None:
        """
        DO NOT MODIFY
        Instantiates a Graph class instance
        :param: plt_show : if true, render plot when plot() is called; else, ignore calls to plot()
        :param: matrix : optional matrix parameter used for fast construction
        :param: csv : optional filepath to a csv containing a matrix
        """
        matrix = matrix if matrix else np.loadtxt(csv, delimiter=',', dtype=str).tolist() if csv else None
        self.size = 0   # number of vertices
        self.vertices = {}  # dictionary of {id : Vertex}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)

    def __eq__(self, other: Graph) -> bool:
        """
        DO NOT MODIFY
        Overloads equality operator for Graph class
        :param other: graph to compare
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        else:
            for vertex_id, vertex in self.vertices.items():
                other_vertex = other.get_vertex(vertex_id)
                if other_vertex is None:
                    print(f"Vertices not equal: '{vertex_id}' not in other graph")
                    return False

                adj_set = set(vertex.adj.items())
                other_adj_set = set(other_vertex.adj.items())

                if not adj_set == other_adj_set:
                    print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                    print(f"Adjacency symmetric difference = "
                          f"{str(adj_set.symmetric_difference(other_adj_set))}")
                    return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        Represents Graph object as string.
        :return: String representation of graph for debugging
        """
        return "Size: " + str(self.size) + ", Vertices: " + str(list(self.vertices.items()))

    def __str__(self) -> str:
        """
        DO NOT MODIFY
        Represents Graph object as string.
        :return: String representation of graph for debugging
        """
        return repr(self)

    def plot(self) -> None:
        """
        DO NOT MODIFY
        Creates a plot a visual representation of the graph using matplotlib
        :return: None
        """
        if self.plot_show:

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_edges())
            max_weight = max([edge[2] for edge in self.get_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_edges()):
                origin = self.get_vertex(edge[0])
                destination = self.get_vertex(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_vertices()])
            y = np.array([vertex.y for vertex in self.get_vertices()])
            labels = np.array([vertex.id for vertex in self.get_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03*max(x), y[j] - 0.03*max(y), labels[j])

            # show plot
            plt.show()
            # delay execution to enable animation
            time.sleep(self.plot_delay)

#============== Modify Graph Methods Below ==============#

    def reset_vertices(self) -> None:
        """
        Resets visited flags of all vertices within the graph.
        :return: None
        """
        for vertex_id, vertex in self.vertices.items():
            vertex.visited = False

    def get_vertex(self, vertex_id: str) -> Vertex:
        """
        Finds the vertex object with id vertex_id if it exists.
        :param vertex_id: id of the vertex to find.
        :return: Vertex with vertex_id if it exists, None otherwise.
        """
        for each_vertex_id, vertex in self.vertices.items():
            if each_vertex_id == vertex_id:
                return vertex
        return None

    def get_vertices(self) -> Set[Vertex]:
        """
        Creates and returns a set of all vertex objects in the graph.
        :return: Set of all vertex objects in the graph.
        """
        returned_set = set()
        for vertex_id, vertex in self.vertices.items():
            returned_set.add(vertex)
        return returned_set

    def get_edge(self, start_id: str, dest_id: str) -> Tuple[str, str, float]:
        """
        Determines the edge connecting the vertex with id start_id to the vertex with id dest_id
        in the form of a tuple with [start_id, dest_id, weight]
        :param start_id: string id of the starting vertex
        :param dest_id: string id of the destination vertex
        :return: Tuple containing [start_id, dest_id, weight]
        """
        # first, search vertices for vertex with start_id.
        start_vertex = self.vertices.get(start_id)
        weight = None
        if start_vertex is not None:
            weight = start_vertex.adj.get(dest_id)
        if weight is not None:
            return start_id, dest_id, weight
        return None

    def get_edges(self) -> Set[Tuple[str, str, float]]:
        """
        Creates and returns a set of tuples representing all edges within the graph.
        Each tuple has [start_id, other_id, weight]
        :return: Set of tuples representing all edges within the graph.
        """
        returned_set = set()

        for each_vertex_in_graph, each_vertex in self.vertices.items():
            for connected_vertex, weight in each_vertex.adj.items():
                my_tuple = each_vertex_in_graph, connected_vertex, weight
                returned_set.add(my_tuple)

        return returned_set

    def add_to_graph(self, start_id: str, dest_id: str = None, weight: float = 0) -> None:
        """
        Adds a vertex / vertices / edge to a graph.
        :param start_id: ID of the vertex being added.
        :param dest_id: ID of vertices that are connected to added vertex.
        :param weight: Weight of the edge to be added.
        :return: None
        """
        # if start_id does not exist in graph, add to graph

        # if dest_id does not exist in graph and dest_id is not None, add to graph

        # if edge is not None, add to graph



    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Description.
        :param matrix:
        :return:
        """
        pass

    def graph2matrix(self) -> Matrix:
        """
        Description.
        :return:
        """
        pass

    def bfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        Description.
        :param start_id:
        :param target_id:
        :return:
        """
        pass

    def dfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        Description.
        :param start_id:
        :param target_id:
        :return:
        """

        def dfs_inner(current_id: str, target_id: str, path: List[str] = [])\
                -> Tuple[List[str], float]:
            """
            Description.
            :param current_id:
            :param target_id:
            :param path:
            :return:
            """
            pass

        pass

    def a_star(self, start_id: str, target_id: str, metric: Callable[[Vertex, Vertex], float])\
            -> Tuple[List[str], float]:
        """
        Description.
        :param start_id:
        :param target_id:
        :param metric:
        :return:
        """
        pass

    def make_equivalence_relation(self) -> int:
        """
        Description.
        :return:
        """
        pass


class AStarPriorityQueue:
    """
    Priority Queue built upon heapq module with support for priority key updates
    Created by Andrew McDonald
    Inspired by https://docs.python.org/3/library/heapq.html
    """

    __slots__ = ['data', 'locator', 'counter']

    def __init__(self) -> None:
        """
        Construct an AStarPriorityQueue object
        """
        self.data = []                        # underlying data list of priority queue
        self.locator = {}                     # dictionary to locate vertices within priority queue
        self.counter = itertools.count()      # used to break ties in prioritization

    def __repr__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        lst = [f"[{priority}, {vertex}], " if vertex is not None else "" for
               priority, count, vertex in self.data]
        return "".join(lst)[:-1]

    def __str__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        return repr(self)

    def empty(self) -> bool:
        """
        Determine whether priority queue is empty
        :return: True if queue is empty, else false
        """
        return len(self.data) == 0

    def push(self, priority: float, vertex: Vertex) -> None:
        """
        Push a vertex onto the priority queue with a given priority
        :param priority: priority key upon which to order vertex
        :param vertex: Vertex object to be stored in the priority queue
        :return: None
        """
        # list is stored by reference, so updating will update all refs
        node = [priority, next(self.counter), vertex]
        self.locator[vertex.id] = node
        heapq.heappush(self.data, node)

    def pop(self) -> Tuple[float, Vertex]:
        """
        Remove and return the (priority, vertex) tuple with lowest priority key
        :return: (priority, vertex) tuple where priority is key,
        and vertex is Vertex object stored in priority queue
        """
        vertex = None
        while vertex is None:
            # keep popping until we have valid entry
            priority, count, vertex = heapq.heappop(self.data)
        del self.locator[vertex.id]            # remove from locator dict
        vertex.visited = True                  # indicate that this vertex was visited
        while len(self.data) > 0 and self.data[0][2] is None:
            heapq.heappop(self.data)          # delete trailing Nones
        return priority, vertex

    def update(self, new_priority: float, vertex: Vertex) -> None:
        """
        Update given Vertex object in the priority queue to have new priority
        :param new_priority: new priority on which to order vertex
        :param vertex: Vertex object for which priority is to be updated
        :return: None
        """
        node = self.locator.pop(vertex.id)      # delete from dictionary
        node[-1] = None                         # invalidate old node
        self.push(new_priority, vertex)         # push new node
