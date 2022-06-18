# Goldstein–Price function

import autograd.numpy as np
from autograd import grad


class GoldsteinPrice:
    def __init__(self):
        super(GoldsteinPrice, self).__init__()
        self.xmin, self.xmax = -2, 2
        self.ymin, self.ymax = -2, 2
        self.y_start, self.x_start = -1.8, 0.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 0, -1, 3  # Global optimum
        self._compute_derivatives()

    def _compute_derivatives(self):
        # Partial derivative of the objective function over x
        self.df_dx = grad(self.eval, 0)
        # Partial derivative of the objective function over y
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = np.log((1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14 * y + 6 * x * y + 3*y**2)) * (30 + (2*x - 3*y)**2 * (18 - 32*x + 12 * x ** 2 + 48 * y - 36 *x * y + 27 * y ** 2)))
        return z
