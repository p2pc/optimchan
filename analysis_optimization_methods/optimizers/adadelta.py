from .base import Optimizer
import numpy as np

"""
Adadelta is another method for overcoming Adagrad’s monotonically decreasing learning rate

\begin{equation}
    x_i^{(k+1)} = x_i^{(k)} - \frac{RMS(\Delta x_i)}{\epsilon + RMS(g_i)}g_i^{(k)}
\end{equation}
"""

class AdaDelta(Optimizer):
    def __init__(self, cost_f, lr=0.001, decay_rate=0.9, x=None, y=None):
        super(AdaDelta, self).__init__(cost_f=cost_f,
                                       lr=lr, x=x, y=y, decay_rate=decay_rate)
        self.decay_x = 0
        self.decay_y = 0
        self.decay_dx = 1
        self.decay_dy = 1

    def step(self, lr=None, decay_rate=None):
        epsilon = 1e-8
        if not lr:
            lr = self.lr
        if not decay_rate:
            decay_rate = self.decay_rate

        # Derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        # Update decays
        self.decay_x = decay_rate * (self.decay_x) + (1-decay_rate)*dx**2
        self.decay_y = decay_rate * (self.decay_y) + (1-decay_rate)*dy**2

        update_x = dx*((np.sqrt(epsilon + self.decay_dx)) /
                       (np.sqrt(epsilon + self.decay_x)))
        update_y = dy*((np.sqrt(epsilon + self.decay_dy)) /
                       (np.sqrt(epsilon + self.decay_y)))

        self.x = self.x - (update_x)*lr
        self.y = self.y - (update_y)*lr

        # Update decays d
        self.decay_dx = decay_rate * \
            (self.decay_dx) + (1-decay_rate)*update_x**2
        self.decay_dy = decay_rate * \
            (self.decay_dy) + (1-decay_rate)*update_y**2

        return [self.x, self.y]
