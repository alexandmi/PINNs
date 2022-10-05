<p align="center">
  <img src="https://raw.githubusercontent.com/alexandmi/PINNs/main/images/logo.png"/>
</p>
Pinns is a python library which creates neural networks that can solve differential equations.

## Description

Pinns implements the emerging and promising technology of physics-informed neural networks. It provides an interface for the easy creation of neural networks, specifically designed and trained for solving differential equations. It is build on Tensorflow and Keras libraries.

By utilizing the tools it provides, a programmer or a mathematician can build a model easily with few lines of code, and train it in a few minutes. When trained, it can solve the equation for any input instantly. The library has been designed to support many categories of differential equations, as well as a big variety of initial and boundary condition types.

Future versions aim to extend the variety of equations categories that can be solved, while making the training process faster.


## Table of contents

* [What are Physics Informed Neural Networks?](#what-are-physics-informed-neural-networks)
* [Install the pinns library](#install-the-pinns-library)
* [Define the domain](#define-the-domain)  
    - [Method get()](#method-get)
* [Create the loss function method](#create-the-loss-function-method)
    - [First derivatives](#first-derivatives)
    - [Second derivatives](#second-derivatives)
    - [Higher derivatives](#higher-derivatives)
    - [Forms and functions of x and y](#forms-and-functions-of-x-and-y)
    - [Examples of complete pde methods](#examples-of-complete-pde-methods)
* [Define the initial conditions](#define-the-initial-conditions)
    - [Transformation of initial conditions](#transformation-of-initial-conditions)
    - [Creation of initial conditions](#creation-of-initial-conditions)
* [Build and train your model](#build-and-train-your-model)
    - [Build the neural network](#build-the-neural-network)
    - [Train the model](#train-the-model)
* [Complete implementation example](#complete-implementation-example)
    - [Print the results](#print-the-results)
* [License](#license)
* [Author](#author)

## What are Physics Informed Neural Networks?

In short, Physics Informed Neural Networks (PINNs) are neural networks (NN) that can solve differential equations (DE). They belong in the field of Physics-Informed Machine Learning, which aims to simulate physical phenomena with deep learning algorithms that incorporate empirical data and mathematical models.

A PINNs operation is summarized as follows: Data x are fed into the input layer, and an approximation of the solution y comes out of the output layer. Then, using back propagation it calculates all the derivatives present in the DE, using the x and y data. This process is called automatic differentiation. Having the x,y, and derivative data, it formulates the standard from of the DE (right side of the equation equals zero), and sets it as a loss function. Since the NN naturally tries to get the loss function as close to zero as possible, by training the NN it simultaneously finds an approximation of the DE solution.


## Install the pinns library

The library is available at the PyPi repository, so you can easily install it by typing:

    !pip install pinns

## Define the domain

The domain and all things related to point sets are handled by the `geometry` module. The DE domain is created by the `Domain` class. The constructor has 3 parameters:

> **minval** (*int/list*):  *Lower boundary coordinates*  
> **maxval** (*int/list*): *Higher boundary coordinates*  
> **num_domain** (*int*):  *Number of domain points*

The boundary coordinates for an ordinary DE are numbers, whereas for a partial DE are lists of numbers. The number of domain points can vary from a few hundred to a few thousands, depending on the difficulty and complexity of the DE, so feel free to experiment with the number. Keep in mind however that the more points the domain has, the longer will the training take.

**Example 1:** Simple interval $x \in [-1,1]$, with 100 points:
		
	x = pinns.Domain(-1,1,100)

For multidimensional domains, the lower and higher boundaries for each input variable are in lists.

**Example 2:**  Domain $x_1 \in [-1,1], x_2 \in [0,2]$, with 100 points:
  
	x = pinns.Domain([-1,0],[1,2],100)

***Note 1:*** The domain created by the above command is not to be confused with the domain $x_1\in{[-1,0]}, x_2\in{[1,2]}$. The separation of input parameters is based on the boundaries of the variable intervals, and not by the intervals themselves.

***Note 2:*** The library currently supports only domains whose individual variables are intervals. That means that more complex geometrical domains like circles, spheres, polygons and triangles are not yet supported.

### Method get()

The get() method is used on Domain objects, in order to acquire a specific vector from the domain. For example, if we want to access separately one of the input variables from the above example, we type:

	x0 = x.get(0)
	x1 = x.get(1)

***Note 1:*** Watch out for the numbering used for the DE variables. In many mathematical statements the numbering starts from 1, but the first element in the Domain is returned with get(0). In the example above, the original domain variables were $x_1$ and $x_2$, but to avoid confusion when handled in the code, they were renamed to `x0` and  `x1` respectfully.

***Note 2:*** Keep in mind that while the original x variable is a Domain object, variables `x0` and `x1` are tensorflow.Variable tensors, so properties of Domain do not apply to them.


## Create the loss function method

The loss function method is created by the user just like any other python method. For the rest of the doc we'll be conventionally calling it `pde()`. It needs to have 2 parameters, `x` and `y`, and return the standard form of the DE. The DE contains different forms of `x` and `y`, and also their derivatives, which need to be calculated. Derivatives are handled by the `differentiation` module.

### First derivatives
The first derivatives are calculated with the `jacobian()` method of class 	`Gradients`, or `Grad` for short. It takes 4 parameters:
>**y**: *Vector of dependent variables*  
>**x**: *Vector of independent variables*  
>**yi** (*int*): *Index of dependent variable for differentiation*  
>**xi** (*int*): *Index of independent variable for differentiation*

The `y` and `x` arguments are the same as the arguments of the `pde` method. 

Index `yi` specifies the $y$ variable being differentiated. For DEs with one output variable, this index is always 0, which is also the default value. The argument is needed when we have a system of DEs with multiple $y$ outputs, like $y_1, y_2 ...$ etc. 

Index `xi` specifies the $x$ variable with respect to which the derivative is performed. For ordinary DEs with one input, the index is always 0, which is also the default value. The argument is needed when we have a partial DE with multiple $x$ inputs, like $x_1,x_2...$ etc.

**Example 1:** Derivative $y'(x)$ from an ordinary DE

	dyx = pinns.Grad.jacobian(y,x)

**Example 2:** Derivative $dy_0 \over dx_2$ from a DE system

	dy1x2 = pinns.Grad.jacobian(y,x,yi=0,xi=2)

***Note:*** Again, mind that despite the numbering used in the mathematical formula, the first variables in python start from 0.

### Second derivatives
The second derivatives are calculated with the `hessian()` method of class 	`Gradients`. It takes 5 parameters:
>**y**: *Vector of dependent variables*  
>**x**: *Vector of independent variables*  
> **yi** (*int*): *Index of dependent variable for differentiation*  
> **xi** (*int*): *Index of independent variable for the first derivative*  
> **xj** (*int*): *Index of independent variable for the second derivative*  

The `y` and `x` arguments are again the same as the arguments of the `pde` method, and the `yi` index has the same use as in the `jacobian` method.
Indices `xi` and `xj` specify the $x$ variables with respect to which the first and second derivatives are performed. For ordinary DEs with one input, both indices are always 0, which is also the default value.

**Example 1:** Derivative $y''(x)$ from an ordinary DE

	dyxx = pinns.Grad.hessian(y,x)

**Example 2:** Derivative $d^2y_1 \over dx_1 dx_2$ from a DE system

	dy1x1x2 = pinns.Grad.jacobian(y,x,yi=1,xi=1,xj=2)


### Higher derivatives
In order to compute derivatives of higher order than 2, you use a combination of `jacobian` and `hessian` methods. The result of one method must be fed as a `y` argument to the other.

**Example 1:** $\ y^{(3)}(x)$
	
	dyx = pinns.Grad.jacobian(y,x)
	dyxxx = pinns.Grad.hessian(dydx,x)

**Example 2:** $\ d^3y_2 \over dx_1^2dx_2$

	dy2x1x1 = pinns.Grad.hessian(y,x,yi=2,xi=1,xj=1)
	dy2x1x1x2 = pinns.Grad.jacobian(dy2dx1x1,x,yi=0,xi=2)

***Note:*** An already calculated derivative like `dy2x1x1` consists of only one variable y (in this case $y_2$), so when its used as a parameter, the `yi` index must be 0.

### Forms and functions of x and y

Inside the DE formula can exist different variables of x and y, which we can access with the `get()` function. There can also be mathematical functions of them like trigonometric equations. For these equations it's better to use methods provided by the Tensorflow library to avoid any unwanted errors, since the tensors representing the variables are Tensorflow based. For constants like π, Numpy can also be used.

**Example:** $\ q(x,y) = y_1 - e^{-x_0}sin(x_1)$

	import tensorflow as tf
	x0 = x.get(0)
	x1 = x.get(1)
	y1 = y.get(1)
	qx = y1 - tf.math.exp(-x0)*tf.math.sin(x1)


### Examples of complete pde methods

**Example 1:**

Let be DE: $y^{(3)} = 2$

Standard form: $y^{(3)}-2=0$

	def pde(x, y):  
	    dyxx = pinns.Grad.hessian(y, x)  
	    dyxxx = pinns.Grad.jacobian(dy_xx, x)  
	    return dy_xxx - 2


**Example 2:** 

Let be DE system: ${{dy_1 \over dt} = y_2}, {{dy_2 \over dt} = -y_1}$ 

Standard forms of DEs: ${{dy_1 \over dt} - y_2} = 0, {{dy_2 \over dt}  +y_1} = 0$

	def pde(x,y):
		dy1t = pinns.Grad.jacobian(y,x,yi=0)
		dy2t = pinns.Grad.jacobian(y,x,yi=1)
		y1 = y.get(0)
		y2 = t.get(1)
		return [dy1t - y2, dy2t + y1]

***Note:*** When we have a system of DEs, the return statement is a list containing all the DE standard forms.

**Example 3:**

Let be DE: $\ {\partial y \over \partial t} = {\partial ^2y \over \partial x^2} - e^{-t}(sin(πx) - π^2sin(πx))$

Standard form of DE: $\ {\partial y \over \partial t} - {\partial ^2y \over \partial x^2} + e^{-t}(sin(πx) - π^2sin(πx)) = 0$

	def pde(geom,y):
		x = geom.get(0)
		t = geom.get(1)
		dydt = pinns.Grad.jacobian(y, geom, xi=1)  
		dydxx = pinns.Grad.hessian(y, geom, xi=0, xj=0)
		return dydt - dydxx + tf.math.exp(-t) * (tf.math.sin(np.pi*x) - np.pi**2*tf.math.sin(np.pi*x))

***Note 1:***  In this mathematical formula, the second independent variable was named $x$. So if the argument for the input vector was also $x$ there would be confusion. For this, the name changed to `geom`, from *geometry*. If you do this, however, you should not forget to have `geom` as an argument inside `jacobian()` and `hessian()`, and not $x$. 

***Note 2:*** It is important to note that inside the derivative methods, $y$ and $x$ are fed whole, as they were taken as arguments. Any specification for the type of gradient is given from the index parameters.

***Wrong:***
				
	def pde(x,y):
		x1 = x.get(1)
		dydx1 = pinns.Grad.jacobian(y,x1)

***Right:***

	def pde(x,y):
		dydx1 = pinns.Grad.jacobian(y,x,xi=1)

## Define the initial conditions

Initial conditions are equations which include y and/or its derivatives, and use a subset or a specific point of domain x. If there are derivatives, the order of condition must be smaller than the order of the DE. 
Initial conditions are used as part of the loss function, since they are essential for the NN to converge to a specific solution. Without them, the solution of the DE gives us a family of solutions, not a specific one, so the NN cannot work.
They are handled by the `initial_conditions` module.

***Comment:*** *Technically there is a difference between the terms initial condition and boundary condition, the former describing time conditions and the latter space conditions. However, for convenience we are going to call them all initial conditions.* 

### Transformation of initial conditions

To declare an initial condition, first you have to transform it to its normal form, meaning the left side of the equation contains only the highest order derivative, or the y function if there is no derivative.

**Example 1:**

Given form: $y(0) - 1 = 0$

Normal form: $y(0) = 1$

**Example 2:**

Given form: ${{{\partial^2y_1} \over \partial x_0^2}+2x_0} = {\partial y_0 \over \partial x_0}$ , with $x \in [x_0,0]$

Normal form: ${{\partial^2y_1} \over \partial x_0^2} = {\partial y_0 \over \partial x_0} -2x_0$ , with $x \in [x_0,0]$

### Creation of initial conditions

Initial conditions are created with the `IC` class. Its constructor takes 5 parameters:
 
>**x_ic** (*number/list*): *Domain of initial condition*  
>**f** (*number/function*): *Function or number on the right side of the equation*  
>**y_der** (*int*): *Order of y on the left side of the equation*  
>**der_i** (*list*): *Indices of domain variables with respect to which y is derived*  
>**yi** (*int*): *Index of dependent variable that is being differentiated*  

Conditions like *Example 1* are very easy to create, using only the first 2 arguments.

**Example 1:**

	ic1 = pinns.IC(x_ic=0, f=1)

If the right side is a function, we have to declare a method representing this function, according to the guidelines specified at section *Create the loss function method*. For the previous *Example 2*, the right side function is declared as:

	def y_out(x_ic,y):
		dy0x0 = pinns.Grad.jacobian(y, x_ic, yi=0, xi=0)
		x0 = x_ic.get(0)
		return dy0x0 - 2*x0

The last three arguments are meant for the specification of the left side of the equation. For *Example 2*, they would be `y_der=2`,`der_i=[0,0]` and `yi=1`. 

**Example 2:**

	ic2 = pinns.IC(x_ic=[x.get(0),0], f=y_out, y_der=2, der_i=[0,0], yi=1)

When the DE is ordinary and there is a derivative on the left side, the `der_i` can be skipped, since there is only one independent variable with respect of which the differentiation can happen.

***Note 1:*** All the dependent variables and their derivatives in the condition are created by the same domain `x_ic`. This means that for now periodic conditions like $y(0) = y(1)$, are not supported.

***Note 2:*** For now, degrees larger than 1 on the left side of the equation are not supported. This means that the dependent variable of the highest order must not be raised to a power, *e.g.* ${y'(0)}^2 =1$.

## Build and train your model

Model creation and training are handled by the `training` module. It is based on and utilizes properties of the Keras library. The model does not belong to a `pinns` class, so the user can inspect it or train it further, taking advantage of all Keras utilities.


### Build the neural network

The neural network is created with the `net()` method. It returns a sequential NN based on the Keras library. It has 4 parameters:

>**inputs** (*int*): *Number of input nodes*  
>**layers** (*list*): *Nodes per hidden layer*  
>**activation** (*string/list*): *Activation function/functions for hidden layers*  
>**outputs** (*int*): *Number of output nodes*  

The input layer of the NN must have as many nodes as independent variables, and the output layer as many nodes as dependent variables. The number of hidden layers and the number of nodes in each layer is up to the user. Usually the more complicated the DE is, the more nodes and layers it requires. 

The activation function for each layer is also up to the user, although empirically, `tanh` function works best for DEs. The available activation function strings are the same as those provided by Keras.

**Example 1:**

Let be partial DE with 3 independent variables. We want 2 hidden layers with 24 nodes each, and the activation function for both of them to be `tanh`:

	model = pinns.net(inputs=3, layers=[24,24], activation='tanh', outputs=1)

**Example 2:**

Let be a system of 2 ordinary DEs. We want 3 hidden layers with 12,24, and 12 nodes, and activation function `sigmoid`,`tanh`, and `sigmoid` respectfully:

	model = pinns.net(inputs=1,layers=[12,24,12],activation=['sigmoid','tanh','sigmoid'], outputs=2)


### Train the model

The training of a model is executed by the `train()` method. It does not return something, but it changes the model's weights, so it can solve the DE. It has 6 parameters:

>  **model** (*keras.engine.sequential.Sequential*): *Neural network model*   
>  **x_domain** (*pinns.Domain*): *Domain of DE*  
>  **pde** (*method*): *Loss function that returns the standard form of the DE*  
> **ic_list** (*list*): *Initial conditions of the DE*  
> **epochs** (*int*): *Number of training epochs*  
> **lr** (*float*): *Learning rate of training*

**Example:**

	pinns.train(model, x, pde, [ic1,ic2], epochs=3000, lr=0.01)
***Note***: Even if there is one initial condition, it has to be put into a list.

While executed, `train()` prints the total loss, the pde loss and the initial conditions loss every 1000 epochs. At the end it prints the total time needed for the training.


## Complete implementation example

Let be DE:

$y^{(3)} = 2$

With initial conditions:

$y''(1) = 2y'(1)-5$

$y'(-1)=1$

$y(-1)=0$

	import pinns
	
	x = pinns.Domain(-1, 1, 100)  
  
	def pde(x, y):  
	    dy_xx = pinns.Grad.hessian(y, x)  
	    dy_xxx = pinns.Grad.jacobian(dy_xx, x)  
	    return dy_xxx - 2  
  
	def ic_out(x_in, y):  
	    dyx = pinns.Grad.jacobian(y, x_in)  
	    return 2*dyx-5  
  
	ic1 = pinns.IC(x_ic=1, f=ic_out, y_der=2)  
	ic2 = pinns.IC(x_ic=-1, f=1, y_der=1)  
	ic3 = pinns.IC(x_ic=-1, f=0, y_der=0)

	model = pinns.net(inputs=1, layers=3 * [60], activation='tanh', outputs=1)  
	pinns.train(model, x, pde, [ic1, ic2, ic3], epochs=2000, lr=0.001)

Output:

	Training starts...
	Epoch: 0		Total Loss = 2.60e+01	PDE Loss = 4.45e+00		BC Loss = 2.15e+01
	Epoch: 1000		Total Loss = 1.18e-04	PDE Loss = 1.18e-04		BC Loss = 3.20e-08
	Epoch: 2000		Total Loss = 4.24e-04	PDE Loss = 6.71e-05		BC Loss = 3.57e-04
	Training took 140.820704959 s

***Note***: The model chosen after the training is the one which corresponds to the last epoch, not the one with the smallest total loss. This is meant to be fixed in a later version.

### Print the results

You can use the `matplotlib` library to print the graph of the approximate solution, and if you know the real solution, you can overlap and compare them. For the example above, the real solution is:

$y(x) = {1 \over 3}x^3+{5 \over 6}x^2+{5 \over 3}x+{7 \over 6}$

We type:

	import matplotlib.pyplot as plt  
	import numpy as np  
	  
	x_test = np.linspace(-1, 1, 100)  
	y_true = 1/3*x_test**3 + 5/6*x_test**2 + 5/3*x_test + 7/6  
	y_pred = model(x_test)  
	  
	plt.plot(y_true)  
	plt.plot(y_pred)  
	plt.title('Evaluation')  
	plt.legend(['Real', 'Predicted'])  
	plt.show()


![The overlapping graphs of approximate and real solution](https://raw.githubusercontent.com/alexandmi/PINNs/main/images/robin%20plot.png)


As we can see, the training was successful, since the two graphs coincide perfectly.

## License

GNU General Public License v3.0 or later.

Open LICENSE to see the full text.

## Author

Alexandros Mavromatis

Email: *alex38clues@gmail.com*

Github: *https://github.com/alexandmi*
