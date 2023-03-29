import autograd.numpy as np
from autograd import grad
import math


class Bukin:
    def __init__(self):
        super(Bukin, self).__init__()
        self.xmin, self.xmax = -15, -5
        self.ymin, self.ymax = -3, 3
        self.y_start, self.x_start = -2.8, 2.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = -10, 1, 0  # Global optimum
        self._compute_derivatives()

    def _compute_derivatives(self):
        # Partial derivative of the objective function over x
        self.df_dx = grad(self.eval, 0)
        # Partial derivative of the objective function over y
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = np.log(100 * math.sqrt(abs(y - 0.01 * x**2)) +
                   0.01 * abs(x + 10)) / 10
        return z
