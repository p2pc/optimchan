import autograd.numpy as np
from autograd import grad
import math


class Ackley:
    def __init__(self):
        super(Ackley, self).__init__()
        self.xmin, self.xmax = -5, 5
        self.ymin, self.ymax = -5, 5
        self.y_start, self.x_start = -2.8, 2.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0, 0, 0  # Global optimum
        self._compute_derivatives()

    def _compute_derivatives(self):
        # Partial derivative of the objective function over x
        self.df_dx = grad(self.eval, 0)
        # Partial derivative of the objective function over y
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) -
             np.exp(0.5*np.cos(2*np.pi*x) + np.cos(2*np.pi*y)) + np.e + 20)
        return z
