from .base import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, cost_f, lr=0.001, x=None, y=None):
        super(GradientDescent, self).__init__(cost_f, lr, x, y)

    def step(self, lr=None):
        if not lr:
            lr = self.lr

        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.x = self.x - lr*dx
        self.y = self.y - lr*dy
        
        return [self.x, self.y]
