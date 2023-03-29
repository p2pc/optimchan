import autograd.numpy as np
from autograd import grad
import math


class ThreeHumpCamel:
    def __init__(self):
        super(ThreeHumpCamel, self).__init__()
        self.xmin, self.xmax = -5, 5
        self.ymin, self.ymax = -5, 5
        self.y_start, self.x_start = -1.8, 0.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0, 0, 0  # Global optimum
        self._compute_derivatives()

    def _compute_derivatives(self):
        # Partial derivative of the objective function over x
        self.df_dx = grad(self.eval, 0)
        # Partial derivative of the objective function over y
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = 2*x**2 - 1.05 * x**4 + x**6 / 6 + x*y + y**2
        return z
