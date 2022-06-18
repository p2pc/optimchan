from .base import Optimizer
import numpy as np

class AdaGrad(Optimizer):
    def __init__(self, cost_f, lr=0.001, x=None, y=None):
        super(AdaGrad, self).__init__(cost_f=cost_f, lr=lr, x=x, y=y)
        self.sumsq_dx = 0
        self.sumsq_dy = 0

    def step(self, lr=None):
        epsilon = 1e-8
        if not lr:
            lr = self.lr

        # derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.sumsq_dx += dx**2
        self.sumsq_dy += dy**2
        
        self.x = self.x - (lr/(np.sqrt(epsilon + self.sumsq_dx)))*dx
        self.y = self.y - (lr/(np.sqrt(epsilon + self.sumsq_dy)))*dy

        return [self.x, self.y]
