from .tensor import *

class Graph:
    """
        Represents the graph data structure.

        Attributes:
            vertices (list): A list of vertices in the graph.
            edges (list): A list of edges in the graph.
            weights (list): A list of weights corresponding to the edges.
            matrix (Matrix): An adjacency matrix representing the graph.
            directed (bool): A flag indicating whether the graph is directed or undirected.
    """

    def __init__(self, vertices=None, edges=None, weights=None, matrix=None, directed: bool = False):

        if not ((isinstance(vertices, Union[tuple, list]) or (vertices is None))
                and (isinstance(edges, Union[tuple, list]) or (edges is None))
                and (isinstance(weights, Union[tuple, list]) or (weights is None))):
            raise ArgTypeError()

        if edges is not None:
            for pair in edges:
                if len(pair) != 2:
                    raise AmountError()
                if vertices is not None:
                    if pair[0] not in vertices or pair[1] not in vertices:
                        raise KeyError()

        if weights is not None:
            for val in weights:
                if not (isinstance(val, Union[int, float, Decimal, Infinity, Undefined])):
                    raise ArgTypeError("Must be a numerical value.")

        self.directed = directed

        if matrix is None:
            # let "vertices" and "edges" be only lists or tuples.
            # weights and edges can also be None.
            if vertices is None:
                self.vertices = []
            else:
                self.vertices = [k for k in vertices]

            if edges is None:
                self.edges = []
            else:
                self.edges = [list(k) for k in edges]

            if weights is None:
                self.weights = [1 for k in self.edges]
            else:
                self.weights = [k for k in weights]

            general = [[0 for k in range(len(self.vertices))] for l in range(len(self.vertices))]
            for k in range(len(self.vertices)):
                if weights is None:
                    for pair in edges:
                        i = self.vertices.index(pair[0])
                        j = self.vertices.index(pair[1])
                        general[i][j] = 1

                else:
                    counter = 0
                    for pair in edges:
                        i = self.vertices.index(pair[0])
                        j = self.vertices.index(pair[1])
                        general[i][j] = weights[counter]
                        counter += 1
                    del counter
            if self.vertices:
                self.matrix = Matrix(*[Vector(*k) for k in general])
            else:
                self.matrix = Matrix(Vector())

        else:
            dim = matrix.dimension.split("x")
            if dim[0] != dim[1]:
                raise DimensionError(2)
            if vertices is not None:
                if int(dim[0]) != len(vertices):
                    raise DimensionError(0)
                self.vertices = [k for k in vertices]  # You can still name your vertices like "a", "b", "c"...
            else:
                self.vertices = [k for k in range(len(matrix.values))]
            # "edges" and "weights" option will be ignored here

            self.weights = []
            self.edges = []
            for k in range(int(dim[0])):
                for l in range(int(dim[1])):
                    val = matrix[k][l]
                    if val != 0:
                        self.edges.append([self.vertices[k], self.vertices[l]])
                        self.weights.append(val)

            self.matrix = matrix.copy()

    def __str__(self):
        m = maximum(self.weights)
        beginning = maximum([len(str(k)) for k in self.vertices]) + 1
        length = len(str(m)) + 1
        length = beginning if beginning >= length else length

        result = beginning * " "
        each = []
        for item in self.vertices:
            the_string = str(item)
            emptyness = length - len(the_string)
            emptyness = emptyness if emptyness >= 0 else 0
            val = " " * emptyness + the_string
            result += val
            each.append(val)
        result += "\n"

        for i in range(len(each)):
            emptyness = beginning - len(str(self.vertices[i]))
            emptyness = emptyness if emptyness >= 0 else 0

            result += " " * emptyness + str(self.vertices[i])
            for j in range(len(each)):
                val = str(self.matrix[i][j])
                emptyness = length - len(val)
                emptyness = emptyness if emptyness >= 0 else 0
                result += " " * emptyness + val
            result += "\n"

        return result

    def addedge(self, label, weight=1):
        """
            Adds an edge with the specified label and weight to the graph.

            Args:
                label (Union[tuple, list]): A tuple or list representing the endpoints of the edge.
                weight (Union[int, float, Decimal, Infinity, Undefined], optional): The weight of the edge. Defaults to 1.

            Returns:
                Graph: The updated graph object.

            Raises:
                ArgTypeError: If the label is not a tuple or list.
                AmountError: If the label does not contain exactly two elements.
                ArgTypeError: If the weight is not a numerical value.
        """
        if not (isinstance(label, Union[tuple, list])): raise ArgTypeError()
        if len(label) != 2: raise AmountError()
        if not (isinstance(weight, Union[int, float, Decimal, Infinity, Undefined])):
            raise ArgTypeError("Must be a numerical value.")

        i = self.vertices.index(label[0])
        j = self.vertices.index(label[1])
        self.matrix[i][j] = weight

        self.edges.append(list(label))
        self.weights.append(weight)

        return self

    def popedge(self, label):
        """
            Removes the edge with the specified label from the graph.

            Args:
                label (Union[tuple, list]): A tuple or list representing the endpoints of the edge to be removed.

            Returns:
                Union[tuple, list]: The removed edge label.

            Raises:
                ArgTypeError: If the label is not a tuple or list.
                AmountError: If the label does not contain exactly two elements.
        """
        if not (isinstance(label, Union[tuple, list])): raise ArgTypeError()
        if len(label) != 2: raise AmountError()

        k = self.edges.index(label)
        self.edges.pop(k)
        self.weights.pop(k)

        i = self.vertices.index(label[0])
        j = self.vertices.index(label[1])
        self.matrix[i][j] = 0
        return label

    def addvertex(self, v):
        """
            Adds a vertex to the graph.

            Args:
                v: The vertex to be added.

            Returns:
                Graph: The updated graph object.
        """
        self.vertices.append(v)

        general = [[0 for k in range(len(self.vertices))] for l in range(len(self.vertices))]
        for k in range(len(self.vertices)):
            if not self.weights:
                for pair in self.edges:
                    i = self.vertices.index(pair[0])
                    j = self.vertices.index(pair[1])
                    general[i][j] = 1
            else:
                counter = 0
                for pair in self.edges:
                    i = self.vertices.index(pair[0])
                    j = self.vertices.index(pair[1])
                    general[i][j] = self.weights[counter]
                    counter += 1
                del counter

        self.matrix = Matrix(*[Vector(*k) for k in general])
        return self

    def popvertex(self, v):
        """
            Removes the specified vertex and its incident edges from the graph.

            Args:
                v: The vertex to be removed.

            Returns:
                The removed vertex.

            Raises:
                KeyError: If the specified vertex is not in the graph.
        """
        k = self.vertices.index(v)
        self.vertices.pop(k)

        while True:
            control = True
            for item in self.edges:
                if v in item:
                    control = False
                    i = self.edges.index(item)
                    self.edges.pop(i)
                    self.weights.pop(i)
                    break

            if control:
                break

        general = [[0 for k in range(len(self.vertices))] for l in range(len(self.vertices))]
        for k in range(len(self.vertices)):
            if not self.weights:
                for pair in self.edges:
                    i = self.vertices.index(pair[0])
                    j = self.vertices.index(pair[1])
                    general[i][j] = 1
            else:
                counter = 0
                for pair in self.edges:
                    i = self.vertices.index(pair[0])
                    j = self.vertices.index(pair[1])
                    general[i][j] = self.weights[counter]
                    counter += 1
                del counter

        self.matrix = Matrix(*[Vector(*k) for k in general])
        return v

    def getdegree(self, vertex):
        """
            Returns the degree of the specified vertex.

            Args:
                vertex: The vertex whose degree is to be determined.

            Returns:
                int: The degree of the specified vertex.

            Raises:
                KeyError: If the specified vertex is not in the graph.
        """
        if vertex not in self.vertices: raise KeyError()

        d = 0
        for item in self.edges:
            if vertex in item:
                d += 1
        return d

    def getindegree(self, vertex):
        """
            Returns the indegree of the specified vertex in the graph.

            Args:
                vertex: The vertex whose indegree is to be determined.

            Returns:
                int: The indegree of the specified vertex.

            Raises:
                KeyError: If the specified vertex is not in the graph.
        """
        if vertex not in self.vertices: raise KeyError()

        d = 0
        if self.directed:
            for item in self.edges:
                if vertex == item[1]:
                    d += 1
        else:
            for item in self.edges:
                if vertex in item:
                    d += 1
        return d

    def getoutdegree(self, vertex):
        """
            Returns the outdegree of the specified vertex in the graph.

            Args:
                vertex: The vertex whose outdegree is to be determined.

            Returns:
                int: The outdegree of the specified vertex.

            Raises:
                KeyError: If the specified vertex is not in the graph.
        """
        if vertex not in self.vertices: raise KeyError()

        d = 0
        if self.directed:
            for item in self.edges:
                if vertex == item[0]:
                    d += 1
        else:
            for item in self.edges:
                if vertex in item:
                    d += 1
        return d

    def getdegrees(self):
        """
            Returns a dictionary containing the degrees of all vertices in the graph.

            Returns:
                dict: A dictionary where keys are degrees and values are the count of vertices with that degree.
        """
        result = {}
        for v in self.vertices:
            d = self.getdegree(v)
            if d not in result:
                result[d] = 1
            else:
                result[d] += 1

        return result

    def getweight(self, label):
        """
            Returns the weight of the edge with the specified label.

            Args:
                label (Union[tuple, list]): A tuple or list representing the endpoints of the edge.

            Returns:
                Union[int, float, Decimal, Infinity, Undefined]: The weight of the specified edge.

            Raises:
                ArgTypeError: If the label is not a tuple or list.
                AmountError: If the label does not contain exactly two elements.
        """
        if not (isinstance(label, Union[tuple, list])): raise ArgTypeError()
        if len(label) != 2: raise AmountError()

        i = self.edges.index(label)
        return self.weights[i]

    def isIsomorphic(g, h):
        """
            Determines whether two graphs are isomorphic.

            Args:
                g (Graph): The first graph.
                h (Graph): The second graph.

            Returns:
                bool: True if the graphs are isomorphic, False otherwise.

            Raises:
                ArgTypeError: If either g or h is not a Graph object.
        """
        if not (isinstance(g, Graph) or isinstance(h, Graph)): raise ArgTypeError("Must be a Graph.")
        return g.getdegrees() == h.getdegrees()

    def isEuler(self):
        """
            Determines whether the graph is an Euler graph.

            Returns:
                bool: True if the graph is an Euler graph, False otherwise.
        """
        degrees = self.getdegrees()
        for k in degrees:
            if k % 2:
                return False
        return True