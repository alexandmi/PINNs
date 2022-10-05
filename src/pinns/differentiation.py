"""Responsible for the differentiation of functions.

Classes:
    Gradients: Creates and stores the derivatives of first and second order.
"""
import pinns


class Gradients:
    """Implements the differentiation for functions of first and second order, creating
    the Jacobian and the Hessian matrices respectfully.

    Attributes:
        J (3D dictionary): Saved Jacobian matrices (first derivatives) for specific combinations
                        of vectors y and x and for a specific dependent variable of y.
        H (4D dictionary): Saved Hessian matrices (second derivatives) for specific combinations
                        of vectors y and x and for a specific dependent variable of y.
        tape (tensorflow.GradientTape): Necessary object for differentiation with Tensorflow
                                        in eager execution mode.

    Methods:
        jacobian(y, x, yi=0, xi=0):
        hessian(y, x, yi=0, xi=0, xj=0):
        set_tape(tape):
        clear_grads_and_tape():
    """
    J = {}
    H = {}
    tape = None

    # Method for jacobian derivatives
    @classmethod
    def jacobian(cls, y, x, yi=0, xi=0):
        """Calculates the derivative of a dependent variable (y) with respect to all of its
        independent variables (x), creating the Jacobian matrix. It returns the derivative
        with respect to an independent variable specified by the user. If the derivative had been
        calculated before in training cycle, it returns it without differentiating.

        Arguments:
            y (pinns.Range/tensorflow.Variable): Vector of dependent variables
            x (pinns.Domain/tensorflow.Variable): Vector of independent variables
            yi (int): Index of dependent variable for differentiation
            xi (int): Index of independent variable for differentiation

        Returns:
            J[key][yi][:, xi: xi + 1]: Derivative of yi with respect to xi

        Examples:
            >>> grad = Gradients.jacobian(y,x,0,1)
            For first gradient dy0/dx1
        """

        # If y is a Range object transform it to a tf.Variable so it can be processed
        if type(y) is pinns.Range:
            y = y.get()
        # If x is a Domain object transform it to a tf.Variable so it can be processed
        if type(x) is pinns.Domain:
            x = x.get()
        # Generate the hash key for the combination of these two vectors
        key = (y.ref(), x.ref())

        # In case of multiple dependent variables, take only the one specified by the user
        if y.shape[1] > 1:
            y = y[:, yi: yi + 1]

        # If there has not been a differentiation of this y and x combination, create
        # a dictionary with their key that will hold the derivatives of the yi variable,
        # then calculate them and append them there.
        if key not in cls.J:
            y_index = {yi: cls.tape.gradient(y, x)}
            cls.J[key] = y_index

        # If for this key, the yi variable of vector y has not been differentiated yet,
        # calculate the derivatives and append them there.
        if yi not in cls.J[key]:
            cls.J[key][yi] = cls.tape.gradient(y, x)

        # Return the gradient of the variable with index xi.
        return cls.J[key][yi][:, xi: xi + 1]

    # Method for hessian derivatives
    @classmethod
    def hessian(cls, y, x, yi=0, xi=0, xj=0):
        """Calculates the second derivative of a dependent variable (y) with respect to
        all of its independent variables (x) for the second derivative. For the calculation
        of the first derivative it calls the jacobian method. It returns the second derivative
        with respect to independent variables specified by the user. If the derivative had been
        calculated before in training cycle, it returns it without differentiating.

        Arguments:
            y (pinns.Domain/tensorflow.Variable): Vector of dependent variables
            x (pinns.Domain/tensorflow.Variable): Vector of independent variables
            yi (int): Index of dependent variable for differentiation
            xi (int): Index of independent variable for the first derivative
            xj (int): Index of independent variable for the second derivative

        Returns:
            H[key][yi][xi][:, xj: xj + 1]: Derivative of yi with respect to xi and xj

        Examples:
            >>> second_grad = Gradients.hessian(y,x,1,0,1)
            For second gradient dy1/d^2(x0x1)
            >>> first_grad = Gradients.jacobian(y,x,0,1)
            >>> third_grad = Gradients.hessian(first_grad,x,0,2,3)
            For third gradient dy0/d^3(x1x2x3)
        """

        # If y is a Range object transform it to a tf.Variable so it can be processed
        if type(y) is pinns.Range:
            y = y.get()
        # If x is a Domain object transform it to a tf.Variable so it can be processed
        if type(x) is pinns.Domain:
            x = x.get()
        # Generate the hash key for the combination of these two vectors
        key = (y.ref(), x.ref())

        # If there has not been any differentiation of this y and x combination,
        # meaning their key is not in the J dictionary, call the jacobian method
        # to calculate and save their first derivatives.
        if (key not in cls.J) or (yi not in cls.J[key]):
            cls.jacobian(y, x, yi, xi)

        # Take the first derivative from the J dictionary.
        grad = cls.J[key][yi][:, xi: xi + 1]

        # If there has not been a 2nd order gradient of this y and x combination,
        # calculate it using the 1st order gradient from the J directory, which was
        # with respect to the xi variable. Append it to the directory of the xi variable,
        # then append the xi directory to the yi variable directory, and then the yi
        # directory to the key directory.
        if key not in cls.H:
            x_index = {xi: cls.tape.gradient(grad, x)}
            y_index = {yi: x_index}
            cls.H[key] = y_index

        # If there has not been a 2nd order gradient of the yi variable, calculate
        # it using the 1st order gradient from the J directory, which was with respect
        # to the xi variable. Append it to the xi directory, and then append the xi
        # directory to the yi directory.
        if yi not in cls.H[key]:
            x_index = {xi: cls.tape.gradient(grad, x)}
            cls.H[key][yi] = x_index

        # If there has not been a 2nd order gradient with respect to the xi variable,
        # calculate it using the 1st order gradient from the J directory, which was
        # with respect to the xi variable. Append it to the xi directory.
        if xi not in cls.H[key][yi]:
            cls.H[key][yi][xi] = cls.tape.gradient(grad, x)

        # Return the 2nd gradient of the yi with respect to variables xi and xj.
        return cls.H[key][yi][xi][:, xj: xj + 1]

    @classmethod
    def set_tape(cls, tape):
        """Setter for the GradientTape object"""
        cls.tape = tape

    @classmethod
    def clear_grads_and_tape(cls):
        """Method for clearing J and H gradient dictionaries, and reset the
        GradientTape variable, after each epoch."""
        cls.J = {}
        cls.H = {}
        cls.tape = None
