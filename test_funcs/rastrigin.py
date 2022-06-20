import autograd.numpy as np
import math
from autograd import grad


class Rastrigin:
    def __init__(self):
        super(Rastrigin, self).__init__()
        self.A = 10
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
        z = (self.A * 2 + (x**2 - self.A * math.cos(2*math.pi*x)) +
             (y**2 - self.A * math.cos(2*math.pi*y)))
        return z
