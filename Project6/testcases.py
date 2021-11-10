import unittest, string, math, random, cProfile

from Project6.Fall20.Graph import Graph, Vertex, AStarPriorityQueue
# Project6.Fall20

class GraphTests(unittest.TestCase):

    """
    Begin Vertex Tests
    """

    def test_degree(self):

        # (1) test a-->b and a-->c
        vertex = Vertex('a')
        vertex.adj['b'] = 1
        self.assertEqual(vertex.degree(), 1)
        vertex.adj['c'] = 3
        assert vertex.degree() == 2

        # (2) test a-->letter for all letters in alphabet
        vertex = Vertex('a')
        for i, char in enumerate(string.ascii_lowercase):
            self.assertEqual(vertex.degree(), i)
            vertex.adj[char] = i

    def test_get_edges_vertex(self):

        # (1) test a-->b and a-->c
        vertex = Vertex('a')
        solution = {('b', 1), ('c', 2)}
        vertex.adj['b'] = 1
        vertex.adj['c'] = 2
        subject = vertex.get_edges()
        self.assertEqual(subject, solution)

        # (2) test empty case
        vertex = Vertex('a')
        solution = set()
        subject = vertex.get_edges()
        self.assertEqual(subject, solution)

        # (3) test a-->letter for all letters in alphabet
        for i, char in enumerate(string.ascii_lowercase):
            vertex.adj[char] = i
            solution.add((char, i))
        subject = vertex.get_edges()
        self.assertEqual(subject, solution)

    def test_distances(self):

        # (1) test pythagorean triple
        vertex_a = Vertex('a')
        vertex_b = Vertex('b', 3, 4)

        subject = vertex_a.euclidean_distance(vertex_b)
        self.assertEqual(subject, 5)
        subject = vertex_b.euclidean_distance(vertex_a)
        self.assertEqual(subject, 5)
        subject = vertex_a.taxicab_distance(vertex_b)
        self.assertEqual(subject, 7)
        subject = vertex_b.taxicab_distance(vertex_a)
        self.assertEqual(subject, 7)

        # (2) test linear difference
        vertex_a = Vertex('a')
        vertex_b = Vertex('b', 0, 10)

        subject = vertex_a.euclidean_distance(vertex_b)
        self.assertEqual(subject, 10)
        subject = vertex_b.euclidean_distance(vertex_a)
        self.assertEqual(subject, 10)
        subject = vertex_a.taxicab_distance(vertex_b)
        self.assertEqual(subject, 10)
        subject = vertex_b.taxicab_distance(vertex_a)
        self.assertEqual(subject, 10)

        # (3) test zero distance
        vertex_a = Vertex('a')
        vertex_b = Vertex('b')

        subject = vertex_a.euclidean_distance(vertex_b)
        self.assertEqual(subject, 0)
        subject = vertex_b.euclidean_distance(vertex_a)
        self.assertEqual(subject, 0)
        subject = vertex_a.taxicab_distance(vertex_b)
        self.assertEqual(subject, 0)
        subject = vertex_b.taxicab_distance(vertex_a)
        self.assertEqual(subject, 0)

        # (4) test floating point distance
        vertex_a = Vertex('a')
        vertex_b = Vertex('b', 5, 5)

        subject = vertex_a.euclidean_distance(vertex_b)
        self.assertAlmostEqual(subject, 5 * math.sqrt(2))
        subject = vertex_b.euclidean_distance(vertex_a)
        self.assertAlmostEqual(subject, 5 * math.sqrt(2))
        subject = vertex_a.taxicab_distance(vertex_b)
        self.assertEqual(subject, 10)
        subject = vertex_b.taxicab_distance(vertex_a)
        self.assertEqual(subject, 10)

    """
    End Vertex Tests
    """

    """
    Begin Graph Tests
    """

    def test_reset_vertices(self):

        graph = Graph()

        # (1) visit all vertices then reset
        graph.vertices['a'] = Vertex('a')
        graph.vertices['b'] = Vertex('b')

        for vertex in graph.vertices.values():
            vertex.visited = True
        graph.reset_vertices()
        for vertex in graph.vertices.values():
            self.assertFalse(vertex.visited)

    def test_get_vertex(self):

        graph = Graph()

        # (1) test basic vertex object
        vertex_a = Vertex('a')
        graph.vertices['a'] = vertex_a
        subject = graph.get_vertex('a')
        self.assertEqual(subject, vertex_a)

        # (2) test empty case
        subject = graph.get_vertex('b')
        self.assertIsNone(subject)

        # (3) test case with adjacencies
        vertex_b = Vertex('b')
        for i, char in enumerate(string.ascii_lowercase):
            vertex_b.adj[char] = i
        graph.vertices['b'] = vertex_b
        subject = graph.get_vertex('b')
        self.assertEqual(subject, vertex_b)

    def test_get_vertices(self):

        graph = Graph()
        solution = set()

        # (1) test empty graph
        subject = graph.get_vertices()
        self.assertEqual(subject, solution)

        # (2) test single vertex
        vertex = Vertex('$')
        graph.vertices['$'] = vertex
        solution.add(vertex)
        subject = graph.get_vertices()
        self.assertEqual(subject, solution)

        # (3) test multiple vertices
        graph = Graph()
        solution = set()
        for i, char in enumerate(string.ascii_lowercase):
            vertex = Vertex(char)
            graph.vertices[char] = vertex
            solution.add(vertex)
        subject = graph.get_vertices()
        self.assertEqual(subject, solution)

    def test_get_edge(self):

        graph = Graph()

        # (1) neither end vertex exists
        subject = graph.get_edge('a', 'b')
        self.assertIsNone(subject)

        # (2) one end vertex exists
        graph.vertices['a'] = Vertex('a')
        subject = graph.get_edge('a', 'b')
        self.assertIsNone(subject)

        # (3) both end vertices exist, but no edge
        graph.vertices['a'] = Vertex('a')
        graph.vertices['b'] = Vertex('b')
        subject = graph.get_edge('a', 'b')
        self.assertIsNone(subject)

        # (4) a -> b exists but b -> a does not
        graph.vertices.get('a').adj['b'] = 331
        subject = graph.get_edge('a', 'b')
        self.assertEqual(subject, ('a', 'b', 331))
        subject = graph.get_edge('b', 'a')
        self.assertIsNone(subject)

        # (5) connect all vertices to center vertex and return all edges
        graph.vertices['$'] = Vertex('$')
        for i, char in enumerate(string.ascii_lowercase):
            graph.vertices[char] = Vertex(char)
            graph.vertices.get('$').adj[char] = i
        for i, char in enumerate(string.ascii_lowercase):
            subject = graph.get_edge('$', char)
            self.assertEqual(subject, ('$', char, i))

    def test_get_edges(self):

        graph = Graph()

        # (1) test empty graph
        subject = graph.get_edges()
        self.assertEqual(subject, set())

        # (2) test graph with vertices but no edges
        graph.vertices['a'] = Vertex('a')
        graph.vertices['b'] = Vertex('b')
        subject = graph.get_edges()
        self.assertEqual(subject, set())

        # (3) test graph with one edge
        graph.vertices.get('a').adj['b'] = 331
        subject = graph.get_edges()
        self.assertEqual(subject, {('a', 'b', 331)})

        # (4) test graph with two edges
        graph = Graph()
        graph.vertices['a'] = Vertex('a')
        graph.vertices['b'] = Vertex('b')
        graph.vertices.get('a').adj['b'] = 331
        graph.vertices.get('b').adj['a'] = 1855
        subject = graph.get_edges()
        solution = {('a', 'b', 331), ('b', 'a', 1855)}
        self.assertEqual(subject, solution)

        # (5) test entire alphabet graph
        graph = Graph()
        solution = set()
        for i, char in enumerate(string.ascii_lowercase):
            graph.vertices[char] = Vertex(char)
            for j, jar in enumerate(string.ascii_lowercase):
                if i != j:
                    graph.vertices.get(char).adj[jar] = 26 * i + j
                    solution.add((char, jar, 26 * i + j))

        subject = graph.get_edges()
        self.assertEqual(subject, solution)

    def test_add_to_graph(self):

        graph = Graph()

        # (1) test creation of first vertex
        graph.add_to_graph('a')
        self.assertEqual(graph.size, 1)
        subject = graph.get_vertices()
        self.assertEqual(subject, {Vertex('a')})

        # (2) test creation of second vertex
        graph.add_to_graph('b')
        self.assertEqual(graph.size, 2)
        subject = graph.get_vertices()
        self.assertEqual(subject, {Vertex('a'), Vertex('b')})

        # (3) test creation of edge a-->b between existing vertices
        graph.add_to_graph('a', 'b', 331)
        self.assertEqual(graph.size, 2)
        subject = graph.get_edges()
        self.assertEqual(subject, {('a', 'b', 331)})

        # (4) test creation of edge b-->a between existing vertices in opposite direction
        graph.add_to_graph('b', 'a', 1855)
        self.assertEqual(graph.size, 2)
        subject = graph.get_edges()
        self.assertEqual(subject, {('a', 'b', 331), ('b', 'a', 1855)})

        # (5) test update of existing edge weight
        graph.add_to_graph('a', 'b', 2020)
        self.assertEqual(graph.size, 2)
        subject = graph.get_edges()
        self.assertEqual(subject, {('a', 'b', 2020), ('b', 'a', 1855)})

        # (6) test creation of edge between existing start and non-existing destination
        graph.add_to_graph('a', 'c', 123)
        self.assertEqual(graph.size, 3)
        subject = graph.get_edges()
        self.assertEqual(subject, {('a', 'b', 2020), ('b', 'a', 1855), ('a', 'c', 123)})

        # (7) test creation of edge between non-existent start and existent destination
        graph.add_to_graph('d', 'c', 345)
        self.assertEqual(graph.size, 4)
        subject = graph.get_edges()
        self.assertEqual(subject, {('a', 'b', 2020), ('b', 'a', 1855), ('a', 'c', 123), ('d', 'c', 345)})

        # (8) test creation of edge between non-existent start and non-existent destination
        graph.add_to_graph('x', 'y', 999)
        self.assertEqual(graph.size, 6)
        subject = graph.get_edges()
        self.assertEqual(subject,
                         {('a', 'b', 2020), ('b', 'a', 1855), ('a', 'c', 123), ('d', 'c', 345), ('x', 'y', 999)})

        # (9) test on entire alphabet graph
        graph = Graph()
        solution = set()
        for i, char in enumerate(string.ascii_lowercase):
            for j, jar in enumerate(string.ascii_lowercase):
                if i != j:
                    graph.add_to_graph(char, jar, 26 * i + j)
                    solution.add((char, jar, 26 * i + j))

        self.assertEqual(graph.size, 26)
        subject = graph.get_edges()
        self.assertEqual(subject, solution)

    def test_matrix2graph(self):

        graph = Graph()

        # (1) test empty matrix
        matrix = [[]]
        graph.matrix2graph(matrix)
        self.assertEqual(graph.size, 0)
        v_subject = graph.get_vertices()
        self.assertEqual(v_subject, set())
        e_subject = graph.get_edges()
        self.assertEqual(e_subject, set())

        # (2) test single vertex with no connection
        matrix = [[None, 'a'],
                  ['a', None]]
        graph.matrix2graph(matrix)
        self.assertEqual(graph.size, 1)
        v_subject = graph.get_vertices()
        self.assertEqual(v_subject, {Vertex('a')})
        e_subject = graph.get_edges()
        self.assertEqual(e_subject, set())

        # (3) test single vertex with connection
        graph = Graph()
        matrix = [[None, 'a'],
                  ['a', 331]]
        graph.matrix2graph(matrix)
        self.assertEqual(graph.size, 1)
        e_subject = graph.get_edges()
        self.assertEqual(e_subject, {('a', 'a', 331)})

        # (4) test two vertices with no connection
        graph = Graph()
        matrix = [[None, 'a', 'b'],
                  ['a', None, None],
                  ['b', None, None]]
        graph.matrix2graph(matrix)
        self.assertEqual(graph.size, 2)
        v_subject = graph.get_vertices()
        self.assertEqual(v_subject, {Vertex('a'), Vertex('b')})
        e_subject = graph.get_edges()
        self.assertEqual(e_subject, set())

        # (5) test two vertices with two-way connection
        graph = Graph()
        matrix = [[None, 'a', 'b'],
                  ['a', None, 100],
                  ['b', 200, None]]
        graph.matrix2graph(matrix)
        self.assertEqual(graph.size, 2)
        e_subject = graph.get_edges()
        self.assertEqual(e_subject, {('a', 'b', 100), ('b', 'a', 200)})

        # (6) test on entire alphabet graph
        matrix = [[None]]
        e_solution = set()
        for i, char in enumerate(string.ascii_lowercase):
            matrix.append([char])
            for j, jar in enumerate(string.ascii_lowercase):
                if i == 0:
                    matrix[0].append(jar)
                if i != j:
                    matrix[i+1].append(26 * i + j)
                    e_solution.add((char, jar, 26 * i + j))
                else:
                    matrix[i+1].append(None)
        graph.matrix2graph(matrix)
        e_subject = graph.get_edges()
        self.assertEqual(e_subject, e_solution)

    def test_graph2matrix(self):

        graph = Graph()

        # (1) test empty graph
        subject = graph.graph2matrix()
        self.assertIsNone(subject)

        # (2) test single vertex with no connection
        matrix = [[None, 'a'],
                  ['a', None]]
        graph.matrix2graph(matrix)
        subject = graph.graph2matrix()
        self.assertEqual(subject, matrix)

        # (3) test single vertex with connection
        graph = Graph()
        matrix = [[None, 'a'],
                  ['a', 331]]
        graph.matrix2graph(matrix)
        subject = graph.graph2matrix()
        self.assertEqual(subject, matrix)

        # (4) test two vertices with no connection
        graph = Graph()
        matrix = [[None, 'a', 'b'],
                  ['a', None, None],
                  ['b', None, None]]
        graph.matrix2graph(matrix)
        subject = graph.graph2matrix()
        self.assertEqual(subject, matrix)

        # (5) test two vertices with 2-way connection
        graph = Graph()
        matrix = [[None, 'a', 'b'],
                  ['a', None, 100],
                  ['b', 200, None]]
        graph.matrix2graph(matrix)
        subject = graph.graph2matrix()
        self.assertEqual(subject, matrix)

        # (6) test on entire alphabet graph
        matrix = [[None]]
        for i, char in enumerate(string.ascii_lowercase):
            matrix.append([char])
            for j, jar in enumerate(string.ascii_lowercase):
                if i == 0:
                    matrix[0].append(jar)
                if i != j:
                    matrix[i+1].append(26 * i + j)
                else:
                    matrix[i+1].append(None)
        graph.matrix2graph(matrix)
        subject = graph.graph2matrix()
        self.assertEqual(subject, matrix)

    def test_bfs(self):

        graph = Graph()

        # (1) test on empty graph
        subject = graph.bfs('a', 'b')
        self.assertEqual(subject, ([], 0))

        # (2) test on graph missing start or dest
        graph.add_to_graph('a')
        subject = graph.bfs('a', 'b')
        self.assertEqual(subject, ([], 0))
        subject = graph.bfs('b', 'a')
        self.assertEqual(subject, ([], 0))

        # (3) test on graph with both vertices but no path
        graph.add_to_graph('b')
        subject = graph.bfs('a', 'b')
        self.assertEqual(subject, ([], 0))

        # (4) test on single edge
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        subject = graph.bfs('a', 'b')
        self.assertEqual(subject, (['a', 'b'], 331))

        # (5) test on two edges
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        subject = graph.bfs('a', 'c')
        self.assertEqual(subject, (['a', 'b', 'c'], 431))

        # (6) test on edge triangle and ensure one-edge path is taken
        # (bfs guarantees fewest-edge path, not least-weighted path)
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        graph.add_to_graph('a', 'c', 999)
        subject = graph.bfs('a', 'c')
        self.assertEqual(subject, (['a', 'c'], 999))

        # (7) test on grid figure-8 and ensure fewest-edge path is taken
        graph = Graph()
        graph.add_to_graph('bottomleft', 'midleft', 1)
        graph.add_to_graph('midleft', 'bottomleft', 1)
        graph.add_to_graph('bottomleft', 'bottomright', 1)
        graph.add_to_graph('bottomright', 'bottomleft', 1)
        graph.add_to_graph('bottomright', 'midright', 1)
        graph.add_to_graph('midright', 'bottomright', 1)
        graph.add_to_graph('midleft', 'midright', 1)
        graph.add_to_graph('midright', 'midleft', 1)
        graph.add_to_graph('topleft', 'midleft', 1)
        graph.add_to_graph('midleft', 'topleft', 1)
        graph.add_to_graph('topleft', 'topright', 1)
        graph.add_to_graph('topright', 'topleft', 1)
        graph.add_to_graph('topright', 'midright', 1)
        graph.add_to_graph('midright', 'topright', 1)

        subject = graph.bfs('bottomleft', 'topleft')
        self.assertEqual(subject, (['bottomleft', 'midleft', 'topleft'], 2))

        graph.reset_vertices()      # mark all unvisited
        subject = graph.bfs('bottomright', 'topright')
        self.assertEqual(subject, (['bottomright', 'midright', 'topright'], 2))

        graph.reset_vertices()      # mark all unvisited
        subject = graph.bfs('bottomleft', 'topright')
        self.assertIn(subject[0], [['bottomleft', 'midleft', 'topleft', 'topright'],
                                   ['bottomleft', 'midleft', 'midright', 'topright'],
                                   ['bottomleft', 'bottomright', 'midright', 'topright']])
        self.assertEqual(subject[1], 3)

        # (8) test on example graph from Onsay's slides, starting from vertex A
        # see bfs_graph.png
        graph = Graph()
        graph.add_to_graph('a', 'b', 2)
        graph.add_to_graph('a', 'c', 2)
        graph.add_to_graph('a', 'e', 2)
        graph.add_to_graph('c', 'f', 2)
        graph.add_to_graph('b', 'd', 2)
        graph.add_to_graph('e', 'h', 2)
        graph.add_to_graph('e', 'g', 2)
        graph.add_to_graph('h', 'i', 2)
        graph.add_to_graph('g', 'i', 2)

        subject = graph.bfs('a', 'd')
        self.assertEqual(subject, (['a', 'b', 'd'], 4))

        graph.reset_vertices()      # mark all unvisited
        subject = graph.bfs('a', 'f')
        self.assertEqual(subject, (['a', 'c', 'f'], 4))

        graph.reset_vertices()      # mark all unvisited
        subject = graph.bfs('a', 'h')
        self.assertEqual(subject, (['a', 'e', 'h'], 4))

        graph.reset_vertices()      # mark all unvisited
        subject = graph.bfs('a', 'g')
        self.assertEqual(subject, (['a', 'e', 'g'], 4))

        graph.reset_vertices()      # mark all unvisited
        subject = graph.bfs('a', 'i')
        self.assertIn(subject[0], [['a', 'e', 'h', 'i'], ['a', 'e', 'g', 'i']])
        self.assertEqual(subject[1], 6)

        # (9) test path which does not exist
        graph.reset_vertices()      # mark all unvisited
        graph.add_to_graph('z')
        subject = graph.bfs('a', 'z')
        self.assertEqual(subject, ([], 0))

    def test_dfs(self):

        graph = Graph()

        # (1) test on empty graph
        subject = graph.dfs('a', 'b')
        self.assertEqual(subject, ([], 0))

        # (2) test on graph missing start or dest
        graph.add_to_graph('a')
        subject = graph.dfs('a', 'b')
        self.assertEqual(subject, ([], 0))
        subject = graph.dfs('b', 'a')
        self.assertEqual(subject, ([], 0))

        # (3) test on graph with both vertices but no path
        graph.add_to_graph('b')
        subject = graph.dfs('a', 'b')
        self.assertEqual(subject, ([], 0))

        # (4) test on single edge
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        subject = graph.dfs('a', 'b')
        self.assertEqual(subject, (['a', 'b'], 331))

        # (5) test on two edges
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        subject = graph.dfs('a', 'c')
        self.assertEqual(subject, (['a', 'b', 'c'], 431))

        # (6) test on linear chain with backtracking distractors
        # see linear_graph.png
        graph = Graph()
        graph.add_to_graph('a', 'b', 1)
        graph.add_to_graph('b', 'a', 2)
        graph.add_to_graph('b', 'c', 1)
        graph.add_to_graph('c', 'b', 2)
        graph.add_to_graph('c', 'd', 1)
        graph.add_to_graph('d', 'c', 2)
        graph.add_to_graph('d', 'e', 1)
        graph.add_to_graph('e', 'd', 2)
        subject = graph.dfs('a', 'e')
        self.assertEqual(subject, (['a', 'b', 'c', 'd', 'e'], 4))

        graph.reset_vertices()      # mark all unvisited
        subject = graph.dfs('e', 'a')
        self.assertEqual(subject, (['e', 'd', 'c', 'b', 'a'], 8))

        # (7) test on linear chain with cycle
        # see cyclic_graph.png
        graph = Graph()
        graph.add_to_graph('a', 'b', 10)
        graph.add_to_graph('b', 'a', 100)
        graph.add_to_graph('b', 'm', 2)
        graph.add_to_graph('m', 'b', 20)
        graph.add_to_graph('m', 'c', 2)
        graph.add_to_graph('c', 'm', 20)
        graph.add_to_graph('b', 'n', 4)
        graph.add_to_graph('n', 'b', 40)
        graph.add_to_graph('n', 'c', 4)
        graph.add_to_graph('c', 'n', 40)
        graph.add_to_graph('c', 'd', 10)
        graph.add_to_graph('d', 'c', 100)

        subject = graph.dfs('a', 'd')
        self.assertIn(subject, [(['a', 'b', 'm', 'c', 'd'], 24),
                                (['a', 'b', 'n', 'c', 'd'], 28)])

        graph.reset_vertices()      # mark all unvisited
        subject = graph.dfs('d', 'a')
        self.assertIn(subject, [(['d', 'c', 'm', 'b', 'a'], 240),
                                (['d', 'c', 'n', 'b', 'a'], 280)])

        # (8) test path which does not exist on graph
        graph.reset_vertices()
        graph.add_to_graph('z')
        subject = graph.dfs('a', 'z')
        self.assertEqual(subject, ([], 0))

    def test_graph_comprehensive(self):

        # construct random matrix
        random.seed(331)
        vertices = [s for s in string.ascii_lowercase]
        matrix = [[None] + vertices]
        probability = 0.1  # probability that two vertices are connected
        for i in range(1, len(matrix[0])):
            row = [matrix[0][i]]
            for j in range(1, len(matrix[0])):
                weight = (random.randint(1, 10))  # choose a random weight between 1 and 9
                connect = (random.random() < probability)  # connect if random draw in (0,1) < probability
                if i == j or not connect:  # such that p=0 never connects and p=1 always connects
                    weight = None  # do not connect vertex to self, either
                row.append(weight)
            matrix.append(row)

        # (1) test matrix2graph and graph2matrix
        graph = Graph()
        [graph.add_to_graph(letter) for letter in string.ascii_lowercase]  # prespecify order of vertices in dict
        graph.matrix2graph(matrix)
        subject = graph.graph2matrix()
        self.assertEqual(subject, matrix)

        # (2) test get_vertices by comparing certain invariants (ordering is not guaranteed under set)
        subject = graph.get_vertices()

        subject_ids = set([vertex.id for vertex in subject])
        solution_ids = set([letter for letter in string.ascii_lowercase])
        self.assertEqual(subject_ids, solution_ids)

        subject_degrees = {}
        for vertex in subject:
            degree = vertex.degree()
            if degree in subject_degrees:
                subject_degrees[degree] += 1
            else:
                subject_degrees[degree] = 1
        solution_degrees = {2: 8, 3: 7, 6: 1, 1: 5, 4: 3, 5: 1, 0: 1}
        self.assertEqual(subject_degrees, solution_degrees)

        # (3) test get_edges
        subject = graph.get_edges()
        solution = {('a', 's', 9), ('a', 't', 8), ('a', 'z', 6), ('b', 'l', 4), ('b', 'v', 9), ('c', 'k', 10),
                    ('c', 'u', 7), ('d', 'e', 1), ('d', 'i', 10), ('e', 'o', 5), ('e', 'q', 4), ('e', 'v', 8),
                    ('f', 'j', 6), ('g', 'h', 7), ('g', 'o', 4), ('g', 'r', 10), ('h', 'd', 4), ('h', 'k', 3),
                    ('h', 'l', 10), ('j', 'b', 5), ('j', 'o', 7), ('j', 'q', 7), ('j', 'r', 3), ('j', 'w', 6),
                    ('j', 'y', 2), ('k', 'z', 3), ('l', 'g', 7), ('l', 'h', 9), ('l', 'i', 10), ('l', 'w', 6),
                    ('m', 'a', 2), ('m', 'f', 10), ('m', 'j', 3), ('n', 't', 6), ('o', 'd', 5), ('o', 'l', 9),
                    ('p', 'q', 1), ('q', 'o', 4), ('q', 'z', 7), ('r', 'c', 4), ('s', 'i', 7), ('s', 'j', 8),
                    ('s', 'k', 8), ('s', 'w', 10), ('t', 'd', 1), ('t', 'g', 2), ('t', 'q', 7), ('t', 'u', 9),
                    ('u', 'g', 8), ('u', 'y', 3), ('v', 'i', 8), ('v', 'x', 5), ('w', 'c', 3), ('w', 'd', 2),
                    ('x', 'h', 7), ('x', 'j', 4), ('x', 'u', 3), ('y', 'e', 7), ('y', 'l', 4), ('y', 'n', 3),
                    ('z', 'j', 1), ('z', 'n', 4), ('z', 'q', 7), ('z', 's', 4), ('z', 'y', 5)}
        self.assertEqual(subject, solution)

        # define helper function to check validity of bfs/dfs result
        def is_valid_path(graph, search_result):
            path, dist = search_result
            length = 0
            for i in range(len(path) - 1):
                start, end = path[i], path[i + 1]
                edge = graph.get_edge(start, end)
                if edge is None:
                    return False  # path contains some edge not in the graph
                length += edge[2]
            return length == dist  # path consists of valid edges: return whether length matches

        # (4) check bfs/dfs on all pairs of vertices in graph
        for start in vertices:
            for end in vertices:
                if start != end:
                    # (5.1) test bfs
                    subject = graph.bfs(start, end)
                    self.assertTrue(is_valid_path(graph, subject))
                    graph.reset_vertices()

                    # (5.2) test dfs
                    subject = graph.bfs(start, end)
                    self.assertTrue(is_valid_path(graph, subject))
                    graph.reset_vertices()

    def test_a_star(self):

        def build_msu_graph(plt_show=False):
            """Helper function for use in constructing A* example"""

            graph = Graph(plt_show)
            vertices = [Vertex('A', 0, 0.01), Vertex('B', 2, 0), Vertex('C', 4, 0),
                        Vertex('D', 6, 0), Vertex('E', 9, 0), Vertex('F', 12, 0),
                        Vertex('G', 2, 5), Vertex('H', 6, 4), Vertex('I', 12, 4),
                        Vertex('J', 5, 9), Vertex('K', 8, 8), Vertex('L', 12, 8),
                        Vertex('M', 8, 10), Vertex('Breslin Center', 0, 2), Vertex('Spartan Stadium', 4, 2),
                        Vertex('Wells Hall', 9, 2), Vertex('Engineering Building', 9, -2),
                        Vertex('Library', 7, 6), Vertex('Union', 8, 11), Vertex('The Rock', 14, 8)]
            for vertex in vertices:
                graph.vertices[vertex.id] = vertex

            edges = [('A', 'B', 8), ('B', 'C', 8), ('C', 'D', 8), ('D', 'E', 12),
                     ('E', 'F', 12), ('B', 'G', 5), ('D', 'H', 4), ('F', 'I', 16),
                     ('G', 'H', 5), ('H', 'I', 6), ('G', 'J', 5), ('I', 'L', 16),
                     ('J', 'K', 4), ('K', 'L', 4), ('J', 'M', 4), ('M', 'L', 4),
                     ('Breslin Center', 'A', 0), ('Spartan Stadium', 'C', 0),
                     ('Wells Hall', 'E', 0), ('Engineering Building', 'E', 0),
                     ('Library', 'K', 0), ('Union', 'M', 0), ('The Rock', 'L', 0)]
            for edge in edges:
                # add edge in both directions
                graph.add_to_graph(edge[0], edge[1], edge[2])
                graph.add_to_graph(edge[1], edge[0], edge[2])

            return graph

        #
        # (A) Grid graph tests
        #
        graph = Graph()

        # (1) test on nxn grid from corner to corner: should shoot diagonal
        # (shortest path is unique, so each heuristic will return the same path)
        grid_size = 5
        for x in range(grid_size):
            for y in range(grid_size):
                idx = f"{x},{y}"
                graph.vertices[idx] = Vertex(idx, x, y)
        for x in range(grid_size):
            for y in range(grid_size):
                if x < grid_size - 1:
                    graph.add_to_graph(f"{x},{y}", f"{x+1},{y}", 1)
                    graph.add_to_graph(f"{x+1},{y}", f"{x},{y}", 1)
                if y < grid_size - 1:
                    graph.add_to_graph(f"{x},{y}", f"{x},{y+1}", 1)
                    graph.add_to_graph(f"{x},{y+1}", f"{x},{y}", 1)
                if x < grid_size - 1 and y < grid_size - 1:
                    graph.add_to_graph(f"{x},{y}", f"{x+1},{y+1}", math.sqrt(2))
                    graph.add_to_graph(f"{x+1},{y+1}", f"{x},{y}", math.sqrt(2))

        for metric in [Vertex.euclidean_distance, Vertex.taxicab_distance]:
            subject = graph.a_star('0,0', '4,4', metric)
            self.assertEqual(subject[0], ['0,0', '1,1', '2,2', '3,3', '4,4'])
            self.assertAlmostEqual(subject[1], (grid_size - 1) * math.sqrt(2))
            graph.reset_vertices()

        # (2) test on nxn grid with penalty for shooting diagonal
        # (shortest path is not unique, so each heuristic will return a different path)
        for x in range(grid_size-1):
            for y in range(grid_size-1):
                graph.add_to_graph(f"{x},{y}", f"{x+1},{y+1}", 3)
                graph.add_to_graph(f"{x+1},{y+1}", f"{x},{y}", 3)

        subject = graph.a_star('0,0', '4,4', Vertex.euclidean_distance)
        self.assertEqual(subject, (['0,0', '1,0', '1,1', '2,1', '2,2', '3,2', '3,3', '4,3', '4,4'], 8))
        graph.reset_vertices()
        subject = graph.a_star('0,0', '4,4', Vertex.taxicab_distance)
        self.assertEqual(subject, (['0,0', '1,0', '2,0', '3,0', '4,0', '4,1', '4,2', '4,3', '4,4'], 8))
        graph.reset_vertices()

        #
        # (B) MSU graph tests
        #
        graph = build_msu_graph(plt_show=False)
        for metric in [Vertex.euclidean_distance, Vertex.taxicab_distance]:

            # (3) test Breslin to Union shortest path in both directions
            # (shortest path is unique, so each heuristic will return the same path)
            subject = graph.a_star('Breslin Center', 'Union', metric)
            solution = (['Breslin Center', 'A', 'B', 'G', 'J', 'M', 'Union'], 22)
            self.assertEqual(subject[0], solution[0])
            self.assertAlmostEqual(subject[1], solution[1])
            graph.reset_vertices()

            subject = graph.a_star('Union', 'Breslin Center', metric)
            solution = (['Union', 'M', 'J', 'G', 'B', 'A', 'Breslin Center'], 22)
            self.assertEqual(subject[0], solution[0])
            self.assertAlmostEqual(subject[1], solution[1])
            graph.reset_vertices()

            # (4) test Breslin to EB shortest path - bypass slow Shaw Ln
            # (shortest path is unique, so each heuristic will return the same path)
            subject = graph.a_star('Breslin Center', 'Engineering Building', metric)
            solution = (['Breslin Center', 'A', 'B', 'G', 'H', 'D', 'E', 'Engineering Building'], 34)
            self.assertEqual(subject[0], solution[0])
            self.assertAlmostEqual(subject[1], solution[1])
            graph.reset_vertices()

            subject = graph.a_star('Engineering Building', 'Breslin Center', metric)
            solution = (['Engineering Building', 'E', 'D', 'H', 'G', 'B', 'A', 'Breslin Center'], 34)
            self.assertEqual(subject[0], solution[0])
            self.assertAlmostEqual(subject[1], solution[1])
            graph.reset_vertices()

            # (5) test EB to The Rock shortest path - bypass slow Farm Ln
            # (shortest path is unique, so each heuristic will return the same path)
            subject = graph.a_star('Engineering Building', 'The Rock', metric)
            solution = (['Engineering Building', 'E', 'D', 'H', 'G', 'J', 'K', 'L', 'The Rock'], 34)
            self.assertEqual(subject[0], solution[0])
            self.assertAlmostEqual(subject[1], solution[1])
            graph.reset_vertices()

            subject = graph.a_star('The Rock', 'Engineering Building', metric)
            solution = (['The Rock', 'L', 'K', 'J', 'G', 'H', 'D', 'E', 'Engineering Building'], 34)
            self.assertEqual(subject[0], solution[0])
            self.assertAlmostEqual(subject[1], solution[1])
            graph.reset_vertices()

            # (6) test Union to Library - despite equal path lengths, A* heuristic will always prefer search to the left
            # (both heuristics will prefer the same path)
            subject = graph.a_star('Union', 'Library', metric)
            solution = (['Union', 'M', 'J', 'K', 'Library'], 8)
            self.assertEqual(subject[0], solution[0])
            self.assertAlmostEqual(subject[1], solution[1])
            graph.reset_vertices()

            subject = graph.a_star('Library', 'Union', metric)
            solution = (['Library', 'K', 'J', 'M', 'Union'], 8)
            self.assertEqual(subject[0], solution[0])
            self.assertAlmostEqual(subject[1], solution[1])
            graph.reset_vertices()

    def test_a_star_comprehensive(self):
        #
        # (C) Random graph tests
        #

        # (1) initialize vertices of Euclidean and Taxicab weighted random graphs
        random.seed(331)
        probability = 0.5                                       # probability that two vertices are connected
        e_graph, t_graph = Graph(), Graph()
        vertices = []
        for s in string.ascii_lowercase:
            x, y = random.randint(0, 100), random.randint(0, 100)
            vertex = Vertex(s, x, y)
            vertices.append(vertex)
            e_graph.vertices[s], t_graph.vertices[s] = vertex, vertex
            e_graph.size += 1
            t_graph.size += 1

        # (2) construct adjacency matrix with edges weighted by appropriate distance metric
        e_matrix = [[None] + [s for s in string.ascii_lowercase]]
        t_matrix = [[None] + [s for s in string.ascii_lowercase]]
        for i in range(1, len(e_matrix[0])):
            e_row = [e_matrix[0][i]]
            t_row = [t_matrix[0][i]]
            for j in range(1, len(e_matrix[0])):
                connect = (random.random() < probability)           # connect if random draw in (0,1) < probability
                e_weighted_dist, t_weighted_dist = None, None
                if i != j and connect:
                    e_dist = vertices[i-1].euclidean_distance(vertices[j-1])
                    t_dist = vertices[i-1].taxicab_distance(vertices[j-1])
                    weight = (random.randint(1, 10))                         # choose a random weight between 1 and 9
                    e_weighted_dist = e_dist * weight                        # create realistic weighted dist
                    t_weighted_dist = t_dist * weight                        # create realistic weighted dist
                e_row.append(e_weighted_dist)
                t_row.append(t_weighted_dist)
            e_matrix.append(e_row)
            t_matrix.append(t_row)
        e_graph.matrix2graph(e_matrix)
        t_graph.matrix2graph(t_matrix)

        # (3) define helper function to check validity of search result
        def is_valid_path(graph, search_result):
            path, dist = search_result
            length = 0
            for i in range(len(path) - 1):
                start, end = path[i], path[i + 1]
                edge = graph.get_edge(start, end)
                if edge is None:
                    return False  # path contains some edge not in the graph
                length += edge[2]
            return length == dist  # path consists of valid edges: return whether length matches

        # (4) test all 26 x 26 pairwise A* traversals across random matrix and ensure they return valid paths w/o error
        for start in vertices:
            for end in vertices:
                if start != end:
                    subject = e_graph.a_star(start.id, end.id, Vertex.euclidean_distance)
                    self.assertTrue(is_valid_path(e_graph, subject))
                    e_graph.reset_vertices()

                    subject = t_graph.a_star(start.id, end.id, Vertex.taxicab_distance)
                    self.assertTrue(is_valid_path(t_graph, subject))
                    t_graph.reset_vertices()

    def test_equivalence_relation(self):

        # (1) test empty Graph
        graph = Graph()
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 0)
        self.assertEqual(graph, Graph())

        # (2) test single vertex, not self connected
        matrix = [[None, 'a'],
                  ['a', None]]
        graph.matrix2graph(matrix)
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 1)
        new_matrix = graph.graph2matrix()
        self.assertEqual(new_matrix, [[None, 'a'],
                                      ['a', 1]])

        # (3) test to ensure known equivalence relation is not modified
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 0)

        # (4) bigger matrix, symmetry check
        graph = Graph()
        matrix = [[None, 'a', 'b'],
                  ['a', None, None],
                  ['b', 1, None]]
        graph.matrix2graph(matrix)
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 3)
        new_matrix = graph.graph2matrix()
        self.assertEqual(new_matrix, [[None, 'a', 'b'],
                                      ['a', 1, 1],
                                      ['b', 1, 1]])

    def test_equivalence_relation_comprehensive(self):

        # Note, random_graph_equirelation files are graphs with 52 vertices. They're big.
        # These graphs have varying levels of connectedness. 0.5 for graph 1, 0.1 for graph 2, 0.8 for graph 3.
        # Graph four added later, 0.01 chance of connection.

        # (1) make eqr on 0.5 probability random graph
        graph = Graph(csv='random_graph_equirelation_1.csv')
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 1363)
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 0)
        solution = Graph(csv='random_graph_equirelation_1_solution.csv')
        self.assertEqual(graph, solution)

        # (2) make eqr on 0.1 probability random graph
        graph = Graph(csv='random_graph_equirelation_2.csv')
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 2445)
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 0)
        solution = Graph(csv='random_graph_equirelation_2_solution.csv')
        self.assertEqual(graph, solution)

        # (3) make eqr on 0.8 probability random graph
        graph = Graph(csv='random_graph_equirelation_3.csv')
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 544)
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 0)
        solution = Graph(csv='random_graph_equirelation_3_solution.csv')
        self.assertEqual(graph, solution)

        # (4) make eqr on 0.01 probability random graph
        graph = Graph(csv='random_graph_equirelation_4.csv')
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 267)
        count = graph.make_equivalence_relation()
        self.assertEqual(count, 0)
        solution = Graph(csv='random_graph_equirelation_4_solution.csv')
        self.assertEqual(graph, solution)

    def test_profile(self):

        # use this testcase to evaluate your code's performance
        # replace function call inside string with any testcase to analyze
        print(cProfile.runctx("self.test_a_star_comprehensive()", globals(), locals()))

    """
    End Graph Tests
    """

    """
    Begin AStarPriorityQueue Tests (NOT FOR MIMIR - just for local development)
    """

    def test_apq(self):

        pq = AStarPriorityQueue()

        # test basics and ensure vertices are visited when popped
        for i, char in enumerate(string.ascii_lowercase):
            pq.push(-i, Vertex(char))
        for i, char in enumerate(reversed(string.ascii_lowercase)):
            priority, vertex = pq.pop()
            assert priority == -25+i
            assert vertex.id == char
            assert vertex.visited

        # test updating and ensure vertices are visited when popped
        for i, char in enumerate(string.ascii_lowercase):
            pq.push(i, Vertex(char))
        pq.update(-1, Vertex('z'))
        pq.update(100, Vertex('a'))
        priority, vertex = pq.pop()
        assert (priority, vertex.id, vertex.visited) == (-1, 'z', True)
        for i, char in enumerate(string.ascii_lowercase):
            if char != 'a' and char != 'z':
                priority, vertex = pq.pop()
                assert (priority, vertex.id, vertex.visited) == (i, char, True)
        priority, vertex = pq.pop()
        assert (priority, vertex.id, vertex.visited) == (100, 'a', True)
        assert pq.empty()   # assert trailing vertices are popped off properly

        # test to ensure Nones are popped off of the end (see SS20 Piazza post for concerns)
        pq = AStarPriorityQueue()
        for i, char in enumerate(string.ascii_lowercase):
            pq.push(i, Vertex(char))
        for i, char in enumerate(string.ascii_lowercase):
            pq.update(-i, Vertex(char))     # will force earlier nodes to be set to None and pushed to end of list
        for i, char in enumerate(reversed(string.ascii_lowercase)):
            priority, vertex = pq.pop()
            assert priority == -25+i
            assert vertex.id == char
            assert vertex.visited
        assert pq.empty()  # assert trailing vertices are popped off properly

    """
    End AStarPriorityQueue Tests
    """


if __name__ == '__main__':
    unittest.main()
