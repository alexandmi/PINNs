"""Pinn package. Implements PINNs for differential equations.

Modules:
    geometry.py: Responsible for the creation and iteration of mathematical set points.
    differentiation.py: Responsible for the differentiation of functions.
    initial_conditions.py: Responsible for the creation and handling of initial conditions.
    training.py: Responsible for the creation and training of the PINN model.
"""

from pinns.geometry import Domain
from pinns.geometry import Range
from pinns.differentiation import Gradients as Grad
from pinns.initial_conditions import IC
from pinns.training import net
from pinns.training import train


def __init__():
    pass
