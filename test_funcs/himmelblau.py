import autograd.numpy as np
from autograd import grad
import math


class Himmelblau:
    def __init__(self):
        super(Himmelblau, self).__init__()
        self.xmin, self.xmax = -5, 5
        self.ymin, self.ymax = -5, 5
        self.y_start, self.x_start = -2.8, 2.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 3, 2, 0  # Global optimum 1
        self.x_optimum_1, self.y_optimum_1, self.z_optimum_1 = - \
            2.805118, 3.131312, 0  # Global optimum 2
        self.x_optimum_2, self.y_optimum_2, self.z_optimum_2 = - \
            3.779310, -3.283186, 0  # Global optimum 3
        self.x_optimum_3, self.y_optimum_3, self.z_optimum_3 = - \
            3.584428, -1.848126, 0  # Global optimum 4
        self._compute_derivatives()

    def _compute_derivatives(self):
        # Partial derivative of the objective function over x
        self.df_dx = grad(self.eval, 0)
        # Partial derivative of the objective function over y
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        return z
