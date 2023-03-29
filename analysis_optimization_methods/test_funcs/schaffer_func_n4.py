import autograd.numpy as np
from autograd import grad
import math


class SchafferN4:
    def __init__(self):
        super(SchafferN4, self).__init__()
        self.xmin, self.xmax = -100, 100
        self.ymin, self.ymax = -100, 100
        self.y_start, self.x_start = -1.8, 0.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0, 1.25313, 0.292579  # Global optimum
        self.x_optimum_1, self.y_optimum_1, self.z_optimum_1 = 0, - \
            1.25313, 0.292579  # Global optimum
        self.x_optimum_2, self.y_optimum_2, self.z_optimum_2 = 1.25313, 0, 0.292579  # Global optimum
        self.x_optimum_3, self.y_optimum_3, self.z_optimum_3 = - \
            1.25313, 0, 0.292579  # Global optimum
        self._compute_derivatives()

    def _compute_derivatives(self):
        # Partial derivative of the objective function over x
        self.df_dx = grad(self.eval, 0)
        # Partial derivative of the objective function over y
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = 0.5 + (math.cos(math.sin(abs(x**2 - y**2))) **
                   2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2
        return z
