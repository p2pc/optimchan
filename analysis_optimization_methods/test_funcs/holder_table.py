import autograd.numpy as np
from autograd import grad
import math

class HolderTable:
    def __init__(self):
        super(HolderTable, self).__init__()
        self.xmin, self.xmax = -10, 10
        self.ymin, self.ymax = -10, 10
        self.y_start, self.x_start = -2.8, 2.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 8.05502, 9.66459, -19.2085  # Global optimum 1
        self.x_optimum_1, self.y_optimum_1, self.z_optimum_1 = -8.05502, 9.66459, -19.2085  # Global optimum 2
        self.x_optimum_2, self.y_optimum_2, self.z_optimum_2 = -8.05502, -9.66459, -19.2085  # Global optimum 3
        self.x_optimum_3, self.y_optimum_3, self.z_optimum_3 = -8.05502, -9.66459, -19.2085  # Global optimum 4
        self._compute_derivatives()

    def _compute_derivatives(self):
        # Partial derivative of the objective function over x
        self.df_dx = grad(self.eval, 0)
        # Partial derivative of the objective function over y
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = -abs(math.sin(x)*math.cos(y)*math.exp(abs(1-math.sqrt(x**2 + y**2)/math.pi)))
        return z
