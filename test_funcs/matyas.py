import autograd.numpy as np
from autograd import grad
import math

class Bukin:
    def __init__(self):
        super(Bukin, self).__init__()
        self.xmin, self.xmax = -10, 10
        self.ymin, self.ymax = -10, 10
        self.y_start, self.x_start = -2.8, 2.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0, 0, 0  # Global optimum
        self._compute_derivatives()

    def _compute_derivatives(self):
        # Partial derivative of the objective function over x
        self.df_dx = grad(self.eval, 0)
        # Partial derivative of the objective function over y
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return z
