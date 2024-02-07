class DimensionError(Exception):
    """Exception raised for errors related to matrix dimensions.

        Attributes:
            code (int): Error code indicating the specific dimension-related issue.
    """
    def __init__(self, code):
        """Initialize DimensionError with a specific error code.

                Args:
                    code (int): Error code indicating the specific dimension-related issue.
                        - 0: Dimensions must match.
                        - 1: Number of dimensions cannot be zero.
                        - 2: Matrix must be a square.
        """
        if code == 0:
            super().__init__("Dimensions must match")
        elif code == 1:
            super().__init__("Number of dimensions cannot be zero")
        elif code == 2:
            super().__init__("Matrix must be a square")

class AmountError(Exception):
    """Exception raised for errors related to incorrect number of arguments.

        This exception is raised when the number of arguments provided does not match
        the expected amount.

        Attributes:
            message (str): Explanation of the error.
    """
    def __init__(self):
        """Initialize AmountError with a message indicating incorrect number of arguments."""
        super().__init__("Not the correct amount of args")

class RangeError(Exception):
    """Exception raised for errors related to arguments being out of range.

        This exception is raised when one or more arguments fall outside the acceptable range.

        Attributes:
            hint (str): Additional information or hint regarding the out-of-range condition.
    """
    def __init__(self, hint: str = ""):
        """Initialize RangeError with an optional hint or additional information.

                Args:
                    hint (str, optional): Additional information or hint regarding the out-of-range condition.
        """
        super().__init__(f"Argument(s) out of range{(': ' + hint) if hint else ''}")

class ArgTypeError(Exception):
    """Exception raised for errors related to arguments having incorrect types.

        This exception is raised when one or more elements of the arguments have the wrong type.

        Attributes:
            hint (str): Additional information or hint regarding the incorrect type condition.
    """
    def __init__(self, hint: str = ""):
        """Initialize ArgTypeError with an optional hint or additional information.

                Args:
                    hint (str, optional): Additional information or hint regarding the incorrect type condition.
        """
        super().__init__(f"Argument elements are of the wrong type{(': ' + hint) if hint else ''}")