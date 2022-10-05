"""Responsible for the creation and iteration of mathematical set points.

Classes:
    PointsSet: Represents a set of points.
    Domain (PointsSet): Represents the domain points of a differential equation.
    Range (PointsSet): Represents the range points of a differential equation.
"""
import tensorflow as tf


class PointsSet:
    """Represents a set of points. Superclass for Domain and Range classes.

    Attributes:
        points (tensorflow.Variable): Tensor with the set of points

    Methods:
        get(index=-1): Returns a specific vector of the tensor
    """

    def __init__(self):
        self.points = None

    def get(self, index=-1):
        """Returns a specific vector of a tensor, from a Domain or Range object.
        If no argument is given, it will return the whole tensor.

        Arguments:
            index (int): Index of vector

        Returns:
            tf_variable (tensorflow.Variable): Vector from tensor

        Examples:
            >>> x = Domain([-1,0],[1,2],100)
            >>> x.get(1)
            Get the second domain variable, x2∈(0,2)
            >>> x.get()
            Get the whole domain tensor
        """

        # Returning the whole tensor
        if index == -1:
            tf_variable = self.points
        # Returning a column
        else:
            tf_variable = tf.Variable(self.points[:, index:index+1])
        return tf_variable


class Domain(PointsSet):
    """Represents the domain points of the differential equation, commonly symbolized as x.

    Attributes:
        points (tensorflow.Variable): Tensor with the points of a domain set
    """

    def __init__(self, minval=0, maxval=0, num_domain=0, tensor=None):
        """Initializes a Domain object in two possible ways:
        1. By creating a tensor from domain boundaries and a number of points.
        2. By taking in a tensor that already represents a domain set.
        
        Arguments:
            minval     (int/list): Lower boundary coordinates
            maxval     (int/list): Higher boundary coordinates
            num_domain (int/list): Number of domain points
            tensor  (tensorflow.Variable): Domain set tensor, meant for direct initialization

        Examples:
            >>> x = Domain(-1,1,100)
            1D domain with 100 points: x∈(-1,1).
            >>> x = Domain([-1,0],[1,2],100)
            2D domain with 100 points: x1∈(-1,1), x2∈(0,2).
            >>> x = Domain(tensor)
            Direct initialization with tensor.
        """
        super().__init__()
        # Initialization by an existing tensor
        if tensor is not None:
            self.points = tensor
        # Initialization with boundary coordinates
        else:
            # Set the domain space according to the boundary coordinates
            domain_space = tf.random_uniform_initializer(minval, maxval)
            # Create a tensor from the domain space with the preferred number of points
            self.points = tf.Variable(domain_space(shape=[num_domain, 1]), dtype=tf.float32)


class Range(PointsSet):
    """Represents the range points of the differential equation, commonly symbolized as y.

    Attributes:
        points (tensorflow.Variable): Tensor with the points of a range set
    """

    def __init__(self, range_tensor):
        """Initializes a Range object with a tensor that represents a range set.

        Arguments:
            range_tensor (tensorflow.Variable): Range set tensor
        """
        super().__init__()
        self.points = range_tensor
