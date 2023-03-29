import autograd.numpy as np
from autograd import grad


class Rosenbrock:
    def __init__(self):
        super(Rosenbrock, self).__init__()
        self.xmin, self.xmax = -1.5, 5
        self.ymin, self.ymax = -3, 1.5
        self.y_start, self.x_start = -2.8, 2.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 1, 1, 0  # Global optimum
        self._compute_derivatives()

    def _compute_derivatives(self):
        # Partial derivative of the objective function over x
        self.df_dx = grad(self.eval, 0)
        # Partial derivative of the objective function over y
        self.df_dy = grad(self.eval, 1)

    def eval(self, x, y):
        z = np.log(100 * (y - x**2)**2 + (1-x)**2) / 10
        return z
