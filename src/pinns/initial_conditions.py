"""Responsible for the creation and handling of initial conditions.

Classes:
    IC: Represents the initial or boundary condition of a differential equation

Methods:
    reshape_input(x_in, tensor_size): Reshapes the integer variables of the domain to have
                                      the same length as the tensor variables.
"""
import pinns
import tensorflow as tf
from inspect import isfunction


def reshape_input(x_in, tensor_size):
    """Reshapes the integer variables of the domain to have the same length as
    the tensor variables.

    Arguments:
        x_in (list): Domain of initial condition
        tensor_size (int): Length of domains tensors

    Returns:
        full_tensor (tensorflow.Variable): Domain containing only tensors of the same length
    """
    # The first element of the list is checked separately, to create the initial tf.Variable.
    # The rest of the elements are concatenated to this tf.Variable.
    # If the list element is a tensor, create the tf.Variable with it.
    if type(x_in[0]) is not float and type(x_in[0]) is not int:
        full_tensor = x_in[0]
    # If the list element is a number, repeat it tensor_size times to make a vector,
    # and then transform this vector to a tf.Variable.
    else:
        temp_vector = tf.fill([tensor_size, 1], tf.cast(x_in[0], tf.float32))
        temp_variable = tf.Variable(temp_vector, trainable=True, dtype=tf.float32)
        full_tensor = temp_variable

    # Same process is followed for the rest of the list elements
    for i in x_in[1:]:
        # If the list element is a tensor, concatenate it to the full_tensor as it is.
        if type(i) is not float and type(i) is not int:
            full_tensor = tf.concat([full_tensor, i], 1)
        # If the list element is a number, repeat it tensor_size times to make a vector,
        # transform this vector to a tf.Variable, and concatenate it to the full_tensor.
        else:
            temp_vector = tf.fill([tensor_size, 1], tf.cast(i, tf.float32))
            temp_variable = tf.Variable(temp_vector, trainable=True, dtype=tf.float32)
            full_tensor = tf.concat([full_tensor, temp_variable], 1)
    return full_tensor


class IC:
    """Represents the initial or boundary condition of a differential equation.

    Attributes:
        x_ic (tensorflow.Variable): Domain of initial condition
        f (function): Function on the right side of the equation
        der_i (list): Indices of domain variables with respect to which y is derived
        yi (int): Index of range variable that is being differentiated
    """

    def __init__(self, x_ic, f, y_der=0, der_i=None, yi=0):
        """Initializes an IC object. Transforms all variables of the domain to have the same
        length and wrapps them in a tensorflow.Variable. Transforms the argument for the
        right side of the equation to a function.

        Arguments:
            x_ic (int/float/list): Domain of initial condition
            f (int/function): Function or number on the right side of the equation
            y_der (int): Order of y on the left side of the equation
            der_i (list): Indices of domain variables with respect to which y is derived
            yi (int): Index of range variable that is being differentiated

        Examples:
            >>> x = pinns.Domain(-1,1,100)
            >>> ic1 = IC(x_ic=0,f=1,y_der=1)
            The simple initial condition y'(0)=1
            >>> def y_out(x_ic,y):
            ...     return y.get(0) + x_ic.get(1)
            >>> ic2 = IC(x_ic=[0,x.get(1)],f=y_out,y_der=1,der_i=[1],yi=1)
            The initial condition dy1x1(0,x2)=y0+x2
        """
        self.yi = yi
        if der_i is None:
            self.der_i = []
        else:
            self.der_i = der_i

        # Input variable x_ic is a list (multiple independent variables).
        if type(x_ic) is list:
            tensor_found = False
            tensor_size = 0
            # Check the values of the list in case there is a tensor.
            for i in x_ic:
                # If there is at least one tensor in the list, keep its length for
                # the reshaping process
                if type(i) is not float and type(i) is not int:
                    tensor_found = True
                    tensor_size = i.shape[0]
                    break
            # If there is a tensor in the list, use the reshape_input method to turn all
            # the numbers of the list to tf.Variable tensors with the same length as the tensor.
            if tensor_found:
                self.x_ic = reshape_input(x_ic, tensor_size)
            # If there are no tensors and only numbers, transform the list directly to a tf.Variable.
            else:
                self.x_ic = tf.Variable([x_ic], trainable=True, dtype=tf.float32)

        # Input variable x_ic has one variable
        else:
            # Normally with one independent variable the input shouldn't be a tensor
            # or a Domain object, but a number. Nevertheless, if the former that occurs,
            # take it as it is, and if the latter occurs, make it a tf.Variable first.
            if type(x_ic) is not float and type(x_ic) is not int:
                self.x_ic = x_ic if type(x_ic) is not pinns.Domain else x_ic.get(-1)
            # If the input is a number, transform it to a tf.Variable
            else:
                self.x_ic = tf.Variable([[x_ic]], trainable=True, dtype=tf.float32)
            # All derivatives occur with respect to the one existing variable, so the
            # derivative indices are all initialized to 0.
            self.der_i = [0] * y_der

        # If the right side of the equation argument (f) is not a function, transform it to one
        if not isfunction(f):
            def y_ic(_x_in, _y):
                return f
            self.f = y_ic
        # If the right side of the equation argument (f) is a function, keep it as it is.
        else:
            self.f = f
