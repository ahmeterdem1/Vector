from .math.infinity import sin, cos, sqrt, Infinity
from .utils import *
from secrets import randbits
from typing import Union, List, Callable
from decimal import Decimal

ADD = 0
MUL = 1
DIV = 2
POW = 3

# below are unary operations in the context of the computational graph
SQRT = 4
EXP = 5
SIN = 6
COS = 7
ARCSIN = 8
LOG2 = 9
LN = 10
SIG = 11
ARCSINH = 12
ARCTAN = 13
ARCCOS = 14
TAN = 15
COSH = 16
ARCCOSH = 17
ARCTANH = 18
SINH = 19
TANH = 20
LOG10 = 21
LN1P = 22
LOGADDEXP = 23

log2E = 1.4426950408889392
ln2 = 0.6931471805599569
ln10 = 2.30258509299
E = 2.718281828459045

def log2(x: Union[int, float, Decimal, Infinity, Undefined], resolution: int = 15):
    """
        Computes the base-2 logarithm of the given number.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The base-2 logarithm of the given number.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is less than or equal to 0, or if 'resolution' is not a positive integer.
    """
    if not isinstance(x, Union[int, float, Decimal, Infinity, Undefined, Variable]):
        raise ArgTypeError("Must be a numerical value.")
    if x <= 0:
        raise RangeError()
    if resolution < 1:
        raise RangeError()

    count = 0
    factor = 1
    if x < 1:
        factor = -1
        x = 1 / x

    while x > 2:
        x = x / 2
        count += 1

    for i in range(1, resolution + 1):
        x = x * x
        if x >= 2:
            count += 1 / (2**i)
            x = x / 2

    return factor * count

def ln(x: Union[int, float, Decimal, Infinity, Undefined], resolution: int = 15):
    """
        Computes the base-e logarithm of the given number.

        Args:
            x: The number.
            resolution (int, optional): The resolution for the approximation. Defaults to 15.

        Returns:
            float: The base-e logarithm of the given number.

        Raises:
            ArgTypeError: If 'x' is not a numerical value.
            RangeError: If 'x' is less than or equal to 0, or if 'resolution' is not a positive integer.
        """

    if isinstance(x, Variable):
        res = Variable(log2(x.value, resolution) / log2E)
        res.operation = LN
        res.backward = (x, Variable(1),)
        return res

    return log2(x, resolution) / log2E

class BinaryNode:

    def __init__(self, val, var):
        self.data = val  # CONSTANT
        self.variable = var
        self.left = None
        self.right = None
        self.parent = None

    def alone(self):
        return (self.left is None or not self.left.flag) and (self.right is None or not self.right.flag)

class Variable:

    """
        A basic numerical variable, that forms a computational graph as
        an extra.

    """

    def __init__(self, val):
        self.value = val
        self.forward = None
        self.backward = None
        self.operation = None  # Operation to apply to self.backward
        self.id = id(self)  # To prevent repetitive function calls during backpropagation

        self.pass_id = 0
        self.grad = 0

    def __str__(self):
        return self.value.__str__()

    def __repr__(self):
        return self.value.__repr__()

    def __getitem__(self, item):
        return self.value.__getitem__(item)

    def __setitem__(self, key, value):
        return self.value.__setitem__(key, value)

    def disconnect(self):
        """
            Disconnects the node from its predecessors. After this
            operation, self will become a leaf on the graph, with
            no preceding nodes. Always recalculate the graph after
            calling this method on its any member, if you are going
            to calculate gradient.

            This function does not fully disconnect the node from the
            graph.
        """
        self.backward = None
        self.operation = None

    def search(self, ID):
        """
            Find a specific node on the computational graph. Finds
            the node by its native object ID. Assumes the node is
            unique on the graph. Returns the latest occurrence if
            the node appears multiple times on the graph.

            Args:
                ID (int): Object ID generated by CPython to look for
                    in the graph.

            Returns:
                The Variable object of the found node if it is found.
                Otherwise returns None. Check for "None" case, for
                error handling if needed.

        """
        if self.id == ID:
            return self
        else:
            frontier = [self]
            next_froniter: list
            back: Union[tuple, None]

            while frontier:
                next_frontier = []
                for node in frontier:
                    back = node.backward
                    if back is None:
                        continue

                    if back[0].id == ID:
                        return back[0]
                    if back[1].id == ID:
                        return back[1]
                    next_frontier.append(back[0])
                    next_frontier.append(back[1])
                frontier = next_frontier

    def __propagate(self, ID):
        """
            Forward propagate the operations to calculate
            the gradient. If the variable to take the derivative
            respect to appears only once on the computational
            graph, this function is faster for calculating the
            derivative than the backpropagation.

            Args:
                ID (int): Object ID generated by CPython of "self".
                    If not "self", may cause errors.

            Returns:
                The derivative of the computation respect to object
                with the ID.

            Notes:
                This method needs to be called on the node that hosts
                the variable to take the derivative respect to. This
                method is designed more primitively, compared to other
                methods and functions on this class and its helpers.

        """
        if self.forward is None:
            if self.operation == ADD:
                return 1

            if self.operation == MUL:
                if self.backward[0].id == ID:
                    return self.backward[1].value
                else:
                    return self.backward[0].value

            if self.operation == POW:
                if self.backward[0].id == ID:
                    return self.backward[1].value * (
                                self.backward[0].value ** (self.backward[1].value - 1))
                else:
                    return self.value * ln(self.backward[0].value) * self.propagate(self.id)

            if self.operation == DIV:
                if self.backward[0].id == ID:
                    return 1 / self.backward[1].value
                else:
                    return -self.backward[0].value / (self.backward[1].value ** 2)

        if self.operation == ADD:
            return self.forward.propagate(self.id)

        if self.operation == MUL:
            if self.backward[0].id == ID:
                return self.backward[1].value * self.forward.propagate(self.id)
            else:
                return self.backward[0].value * self.forward.propagate(self.id)

        if self.operation == POW:
            if self.backward[0].id == ID:
                return self.backward[1].value * (self.backward[0].value ** (self.backward[1].value - 1)) * self.forward.propagate(self.id)
            else:
                return self.value * ln(self.backward[0].value) * self.propagate(self.id)

        if self.operation == DIV:
            if self.backward[0].id == ID:
                return self.forward.propagate(self.id) / self.backward[1].value
            else:
                return -self.backward[0].value * self.forward.propagate(self.id) / (self.backward[1].value ** 2)

        return self.forward.propagate(self.id)

    def __backpropagate(self, ID):
        """
            Backpropagate the computational graph to compute the
            gradient.

            Args:
                 ID (int): The Object ID generated by CPython of
                    the node to take the gradient respect to.

            Returns:
                The gradient as a value. If you modify Variable
                objects manually, there is a chance that this
                method returns None. Recursive calls of this
                method on the graph may raise exceptions because
                of that.
        """

        if self.id == ID:
            return 1

        if self.backward is None:
            return 0

        if self.operation == ADD:
            return self.backward[0].backpropagate(ID) + self.backward[1].backpropagate(ID)

        if self.operation == MUL:
            return self.backward[0].backpropagate(ID) * self.backward[1].value + \
                self.backward[0].value * self.backward[1].backpropagate(ID)

        if self.operation == POW:
            return self.backward[0].backpropagate(ID) * self.backward[1].value * (self.backward[0].value ** (self.backward[1].value - 1)) + \
                ln(self.backward[0].value) * (self.backward[0].value ** self.backward[1].value) * self.backward[1].backpropagate(ID)

        if self.operation == DIV:
            return self.backward[0].backpropagate(ID) / self.backward[1].value - self.backward[1].backpropagate(ID) * self.backward[0].value / (self.backward[1].value ** 2)

    def derive(self):

        """
            Calculates the derivative of the current node.

            Returns:
                A 2-tuple of 2-tuples. Based on the assumption that
                each node is a binary node, therefore has 2 children.

                - First element of the first tuple, is the derivative
                of the current node, respect to its first child.

                - Second element of the first tuple, is the derivative
                of the current node, respect to its second child.

                - The second tuple, in order, contains the children of
                the current node.

        """

        # TODO: Optimize below if-else logic

        if self.operation == ADD:
            return (1, 1), self.backward

        if self.operation == MUL:
            return (self.backward[1].value, self.backward[0].value), self.backward

        if self.operation == POW:
            return ((self.backward[1].value * (self.backward[0].value ** (self.backward[1].value - 1)),
                    ln(self.backward[0].value) * (self.backward[0].value ** self.backward[1].value)),
                    self.backward)

        if self.operation == DIV:
            return (1/self.backward[1].value, -self.backward[0].value / (self.backward[1].value ** 2)), self.backward

        if self.operation == SQRT:
            return (1/(2*self.value), 0), self.backward

        if self.operation == EXP:
            return (self.value, 0), self.backward

        if self.operation == SIN:
            return (cos(self.backward[0].value), 0), self.backward

        if self.operation == COS:
            return (-sin(self.backward[0].value), 0), self.backward

        if self.operation == ARCSIN:
            return (1/sqrt(1 - (self.backward[0].value ** 2)), 0), self.backward

        if self.operation == LOG2:
            return (1/(self.backward[0].value * ln2), 0), self.backward

        if self.operation == LN:
            return (1/self.backward[0].value, 0), self.backward

        if self.operation == SIG:
            return (self.backward[0].value * (1 - self.backward[0].value), 0), self.backward

        if self.operation == ARCSINH:
            return (1 / sqrt(1 + self.backward[0].value ** 2), 0), self.backward

        if self.operation == ARCTAN:
            return (1 / (1 + self.backward[0].value ** 2)), self.backward

        if self.operation == ARCCOS:
            return (-1 / sqrt(1 - self.backward[0].value ** 2)), self.backward

        if self.operation == TAN or self.operation == TANH:
            return (1 - self.value ** 2, 0), self.backward

        if self.operation == COSH:
            return (E ** self.backward[0].value - E ** -self.backward[0].value, 0), self.backward

        if self.operation == ARCCOSH:
            return (1 / sqrt(self.backward[0].value ** 2 - 1), 0), self.backward

        if self.operation == ARCTANH:
            return (1 / (1 - self.backward[0].value ** 2), 0), self.backward

        if self.operation == SINH:
            return (E ** self.backward[0].value + E ** -self.backward[0].value, 0), self.backward

        if self.operation == LOG10:
            return (1 / (self.backward[0].value * ln10), 0), self.backward

        if self.operation == LN1P:
            return (1 / (self.backward[0].value + 1), 0), self.backward

        if self.operation == LOGADDEXP:
            return ((E ** self.backward[0].value) / self.value, (E ** self.backward[1].value) / self.value), self.backward

        return (1, 1), (0, 0)

    def set(self, val):
        """
            Change, reset the value that is wrapped by Variable.
            This method is not used for changing inner values held
            by the self.value itself, for example when wrapping a
            Vector.

            Args:
                val: New value to reassign to the Variable node, self.

        """
        if self.backward is None:
            self.value = val
        else:
            raise ArithmeticError()

    def __calculate(self):
        """
            Redo the calculation contained in the self-node.

            Returns:
                Returns the result of the contained calculation. If there
                is no calculation wrapped, returns the contained value.
        """
        if self.operation == ADD:
            self.value = self.backward[0].value + self.backward[1].value
            return self.value
        if self.operation == MUL:
            self.value = self.backward[0].value - self.backward[1].value
            return self.value
        if self.operation == DIV:
            self.value = self.backward[0].value / self.backward[1].value
            return self.value
        if self.operation == POW:
            self.value = self.backward[0].value ** self.backward[1].value
            return self.value
        return self.value

    def calculate(self):
        """
            (Re)calculate the computational graph, backpropagating
            from the self node. This method potentially changes the
            value stored in self.value, numerically.

            Returns:
                The value calculated to be stored in node, self.
                This method may potentially raise an ArithmeticError
                if Variable objects are manipulated manually.

        """
        frontier = [self]
        next_frontier: list
        back: Union[tuple, None]
        leaves = []
        next_leaves: list

        while frontier:
            next_frontier = []
            for node in frontier:
                back = node.backward
                if back is None:
                    leaves.append(node)
                    continue

                next_frontier.append(back[0])
                next_frontier.append(back[1])
            frontier = next_frontier

        while leaves:
            next_leaves = []
            for leaf in leaves:
                leaf.__calculate()
                # If forward of leaf is self, it is not included into the next leaves
                if leaf.forward is not None or (leaf.forward.id != self.id):
                    next_leaves.append(leaf.forward)

            leaves = next_leaves

        return self.__calculate()  # Both updates self.value and returns it

    @staticmethod
    def buildBinaryOperation(v1, v2, f: Callable, opcode: int):
        """
            Add a particular operation to the computational graph,
            defined by f and opcode, given arguments to operation
            v1 and v2.

            Args:
                v1 (Variable): First argument to operation as a Variable
                    object.

                v2 (Variable): Second argument to operation as a Variable
                    object.

                f (Callable): Binary operation to perform on v1 and v2 as
                    arguments, in order.

                opcode (int): Operation code to flag the node as. Must be one
                    of in [0, 11].

            Returns:
                Variable: Result of the operation, added as a parent to v1 and v2,
                    having as children v1 and v2, flagged with given opcode.
        """
        res = f(v1.value, v2.value)
        res = Variable(res)
        res.backward = (v1, v2)
        res.operation = opcode
        v1.forward = res
        v2.forward = res
        return res

    @staticmethod
    def buildUnaryOperation(v, f: Callable, opcode: int):
        res = f(v.value)
        res = Variable(res)
        res.backward = (v, Variable(1))
        res.operation = opcode
        v.forward = res
        res.backward[1].forward = res
        return res

    def __add__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(self.value + arg.value)
        self.forward = result
        arg.forward = result

        result.backward = (self, arg)
        result.operation = ADD
        return result

    def __radd__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(self.value + arg.value)
        self.forward = result
        arg.forward = result

        result.backward = (self, arg)
        result.operation = ADD
        return result

    def __sub__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(self.value - arg.value)
        self.forward = result
        arg.forward = result

        result.backward = (self, -arg)
        result.operation = ADD
        return result

    def __rsub__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(arg.value - self.value)
        self.forward = result
        arg.forward = result

        result.backward = (arg, -self)
        result.operation = ADD
        return result

    def __mul__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(self.value * arg.value)
        self.forward = result
        arg.forward = result

        result.backward = (self, arg)
        result.operation = MUL
        return result

    def __rmul__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(self.value * arg.value)
        self.forward = result
        arg.forward = result

        result.backward = (arg, self)
        result.operation = MUL
        return result

    def __truediv__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(self.value / arg.value)
        self.forward = result
        arg.forward = result

        result.backward = (self, arg)
        result.operation = DIV
        return result

    def __rtruediv__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(arg.value / self.value)
        self.forward = result
        arg.forward = result

        result.backward = (arg, self)
        result.operation = DIV
        return result

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def __pow__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(self.value ** arg.value)
        self.forward = result
        arg.forward = result

        result.backward = (self, arg)
        result.operation = POW
        return result

    def __rpow__(self, arg):
        if not isinstance(arg, Variable):
            arg = Variable(arg)

        result = Variable(self.value ** arg.value)
        self.forward = result
        arg.forward = result

        result.backward = (arg, self)
        result.operation = POW
        return result

    def __ge__(self, arg):
        if isinstance(arg, Variable):
            return self.value >= arg.value
        return self.value >= arg

    def __gt__(self, arg):
        if isinstance(arg, Variable):
            return self.value > arg.value
        return self.value > arg

    def __le__(self, arg):
        if isinstance(arg, Variable):
            return self.value <= arg.value
        return self.value <= arg

    def __lt__(self, arg):
        if isinstance(arg, Variable):
            return self.value <= arg.value
        return self.value <= arg

    def __eq__(self, arg):
        if isinstance(arg, Variable):
            return self.value == arg.value
        return self.value == arg

    def __ne__(self, arg):
        if isinstance(arg, Variable):
            return self.value != arg.value
        return self.value != arg

    def __or__(self, arg):
        if isinstance(arg, Variable):
            return self.value or arg.value
        return self.value or arg

    def __ror__(self, arg):
        if isinstance(arg, Variable):
            return self.value or arg.value
        return self.value or arg

    def __and__(self, arg):
        if isinstance(arg, Variable):
            return self.value and arg.value
        return self.value and arg

    def __rand__(self, arg):
        if isinstance(arg, Variable):
            return self.value and arg.value
        return self.value and arg

    def __xor__(self, arg):
        if isinstance(arg, Variable):
            return self.value ^ arg.value
        return self.value ^ arg

    def __rxor__(self, arg):
        if isinstance(arg, Variable):
            return self.value ^ arg.value
        return self.value ^ arg


def grad(node: Variable, args: Union[List[Variable], None] = None):
    """
        Takes the gradient of the given computational graph,
        respect to given Variable list.

        Args:
            node (Variable): The last Variable that results
                from the computation, that the gradient will
                be calculated on.

            args (List[Variable]): A list of Variable's that
                the gradient will be calculated respect to.

        Returns:
            A list of values representing the gradients. The
            order of the values correspond to the "args".
            Therefore this list is also a gradient vector/list
            when given multiple arguments.

        Raises:
            ArgTypeError: When args argument is left as None.
    """
    if args is None:
        raise ArgTypeError("Cannot take derivative respect to None.")

    tree_traverser: BinaryNode
    root = BinaryNode(1, node)
    node_frontier = [root]
    replace_node_frontier: list

    id_node_dict = {arg.id: 0 for arg in args}
    ids = id_node_dict.keys()


    # Build the pseudo-computational graph for the values of the derivatives
    while node_frontier:
        replace_node_frontier = []

        for temp_node in node_frontier:

            if temp_node.variable.id in ids:
                id_node_dict[temp_node.variable.id] += temp_node.data

            values, children = temp_node.variable.derive()

            if not isinstance(children[0], int):

                temp_node.left = BinaryNode(values[0] * temp_node.data, children[0])
                temp_node.left.parent = temp_node
                temp_node.right = BinaryNode(values[1] * temp_node.data, children[1])
                temp_node.right.parent = temp_node
                replace_node_frontier.append(temp_node.left)
                replace_node_frontier.append(temp_node.right)

        node_frontier = replace_node_frontier



    return list(id_node_dict.values())

def autograd(node: Variable, args: Union[List[Variable], None] = None):

    """
        This function operates mostly the same as "grad()". See the
        documentation on "grad()" for more information.

        However, the internal algorithm of this function is more than 2 times
        faster than "grad()". The disadvantage being, this function
        cannot be parallelized with threads or processes because of
        how its algorithm works. Each thread would require its own
        separate fully functional computational graph.

        This function also accepts "None" as an argument to take the derivative
        against. When left "None", the full gradient values for all of the
        nodes on the computational graph are calculated and saved in their
        ".grad" property. Then user can choose to retrieve this information
        from the nodes manually, or in parallel, or in some other way. More
        flexibility is provided in the implementation of this function by
        this way.

        At each backwards pass, a unique "pass id" is generated. Pass id, is
        a randomly generated 64-bit integer. Check the "pass_id" property of
        the node that you gave to this function to retrieve it. If pass id's
        don't match, gradients are incorrect/old for a given node, and should
        be discarded or recalculated. This can only happen when a node is fully
        disconnected from the rest of the graph.


    """

    if args is None:
        args = []

    node_frontier = [node]
    node.grad = 1
    node.pass_id = randbits(64)
    PASS = node.pass_id
    replace_node_frontier: list

    while node_frontier:
        replace_node_frontier = []
        for temp in node_frontier:

            values, back = temp.derive()
            if isinstance(back[0], int):
                continue

            if back[0].pass_id != PASS:
                back[0].pass_id = PASS
                back[0].grad = values[0] * temp.grad
            else:
                back[0].grad += values[0] * temp.grad

            if back[1].pass_id != PASS:
                back[1].pass_id = PASS
                back[1].grad = values[1] * temp.grad
            else:
                back[1].grad += values[1] * temp.grad

            replace_node_frontier.append(back[0])
            replace_node_frontier.append(back[1])

        node_frontier = replace_node_frontier

    return [arg.grad if arg.pass_id == PASS else 0 for arg in args]


