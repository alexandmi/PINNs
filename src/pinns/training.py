"""Responsible for the creation and training of the PINN model.

Methods:
    net(inputs, layers, activation, outputs): Creates the model
    train(model, x_domain, pde, ic_list, epochs, lr): Trains the model
"""
import pinns
import tensorflow as tf
import time
import silence_tensorflow

# Don't display unnecessary tensorflow warning messages
silence_tensorflow.silence_tensorflow()


def net(inputs, layers, activation, outputs):
    """Creates the neural network with the tensorflow.keras library.

    Arguments:
        inputs (int): Number of input nodes
        layers (list): Nodes per hidden layer
        activation (string/list): Activation function/functions for hidden layers
        outputs (int): Number of output nodes

    Returns:
         model: A neural network model

    Examples:
        >>> model1 = net(1,[20,20],'tanh',1)
        One activation function for all hidden layers
        >>> model2 = net(1,[20,20],['tanh','sigmoid'],1)
        Multiple activation functions
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input((inputs,)))
    if layers:
        # When all hidden layers have the same activation function
        if type(activation) is not list:
            for i in layers:
                model.add(tf.keras.layers.Dense(units=i, activation=activation))
        # When there is at least one hidden layer with a different activation function
        else:
            for i in activation:
                for j in layers:
                    model.add(tf.keras.layers.Dense(units=j, activation=i))
    model.add(tf.keras.layers.Dense(units=outputs))
    return model


# Method for training
def train(model, x_domain, pde, ic_list, epochs, lr):
    """Trains a keras neural network to solve a differential equation.

    Arguments:
        model (keras.engine.sequential.Sequential): Neural network model
        x_domain (Domain): Domain of differential equation
        pde (method): Loss function that returns the standard form of the differential equation
        ic_list (list): Initial or boundary conditions of differential equation
        epochs (int): Number of training epochs
        lr (float): Learning rate of training

    Examples:
        >>> train(model,x,pde,[ic1],2000,0.01)
        Even for one initial condition it has to be inside a list
    """
    print("\nTraining starts...\n")
    train_start = time.process_time()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for i in range(epochs + 1):
        # General GradientTape object that will watch all the error calculation process,
        # and differentiate the node weights with the total loss.
        with tf.GradientTape() as tape_model:
            ic_total_error = 0
            # Calculate total error from the initial conditions
            for ic in ic_list:
                # GradientTape for initial conditions
                with tf.GradientTape(persistent=True) as tape_ic:
                    tape_ic.watch(ic.x_ic)
                    y_ic_original_tf = model(ic.x_ic, training=True)
                    # Take the needed yi variable from vector y
                    y_ic_derivative = y_ic_original_tf[:, ic.yi:ic.yi + 1]
                    # If there are multiple derivatives, differentiation must take place
                    # inside the watch scope of tape_ic.
                    if len(ic.der_i) > 1:
                        for j in ic.der_i:
                            y_ic_derivative = tape_ic.gradient(y_ic_derivative, ic.x_ic)[:, j:j + 1]
                # For one derivative, differentiation takes place outside the tape_ic watch scope
                if len(ic.der_i) == 1:
                    y_ic_derivative = tape_ic.gradient(y_ic_derivative, ic.x_ic)[:, ic.der_i[0]:ic.der_i[0] + 1]

                # If x and y vectors for the initial condition have more than one dimension,
                # turn them into Domain and Range objects respectively, in case the user wants
                # to use get() to take a specific variable inside these vectors.
                # If they have one dimension, keep them as they are, because the user will use
                # them whole and the get() method won't be needed.
                x_ic = pinns.Domain(tensor=ic.x_ic) if ic.x_ic.shape[1] > 1 else ic.x_ic
                y_ic_original = pinns.Range(y_ic_original_tf) if y_ic_original_tf.shape[1] > 1 else y_ic_original_tf

                # Set tape_ic to the Gradients class so the user can use jacobian() and hessian()
                # inside the ic.f method.
                pinns.Grad.set_tape(tape_ic)
                # Calculate the error between the left and the right side of the equation
                ic_error = tf.math.square(y_ic_derivative - ic.f(x_ic, y_ic_original))
                pinns.Grad.clear_grads_and_tape()

                # In case of multiple input variables reduce their errors to one number
                if len(ic_error.shape) == 2:
                    ic_error = tf.math.reduce_mean(ic_error)

                ic_total_error = ic_total_error + ic_error
                del tape_ic

            # Fix the form of total ic error
            if ic_total_error.shape == ():
                ic_total_error = tf.reshape(ic_total_error, (1,))

            # Calculate the error normal form of the differential equation, conventionally called pde.
            # GradientTape for pde
            with tf.GradientTape(persistent=True) as tape_pde:
                # Transform the Domain object to a tf.Variable, so it can be used inside the model
                x_tf = x_domain.get()
                y_tf = model(x_tf, training=True)
                # Set tape_pde to the Gradients class so the user can use jacobian() and hessian()
                # inside the ic.f method.
                pinns.Grad.set_tape(tape_pde)
                # Transform the tf.Variables of x and y to Domain, and Range respectfully, if needed
                x = x_domain if x_tf.shape[1] > 1 else x_tf
                y = pinns.Range(y_tf) if y_tf.shape[1] > 1 else y_tf
                domain_error = pde(x, y)
                pinns.Grad.clear_grads_and_tape()
            del tape_pde

            # Fix the form of pde error
            domain_total_error = 0
            if type(domain_error) is list:
                for losses in domain_error:
                    domain_total_error = domain_total_error + tf.math.reduce_mean(tf.math.square(losses), axis=0)
            else:
                domain_total_error = tf.math.reduce_mean(tf.math.square(domain_error), axis=0)

            total_error = domain_total_error + ic_total_error

            # Print loss results every 1000 epochs
            if i % 1000 == 0:
                print('Epoch: {}\t\tTotal Loss = {:.2e}\tPDE Loss = {:.2e}\t\tBC Loss = {:.2e}'.format(
                    i, total_error.numpy()[0], domain_total_error.numpy()[0], ic_total_error.numpy()[0]))

        # Update network weights
        model_update_gradients = tape_model.gradient(total_error, model.trainable_variables)
        optimizer.apply_gradients(zip(model_update_gradients, model.trainable_variables))

        # Print final results, in case total number of epochs is not a multiple of 1000
        if i == epochs and i % 1000 != 0:
            print(
                'Epoch: {}\tTotal Loss = {:.2e}\tPDE Loss = {:.2e}\tBC Loss = {:.2e}'.format(
                    i, total_error.numpy()[0], domain_total_error.numpy()[0], ic_total_error.numpy()[0]))
        del tape_model

    train_end = time.process_time() - train_start
    print("\nTraining took", train_end, "s")
