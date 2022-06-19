from .base import Optimizer
import numpy as np

"""
The Adagrad update step is:

\begin{equation}
    x_i^{(k+1)} = x_i^{(k)} - \frac{\alpha}{\epsilon + \sqrt{s_i^{(k)}}}g_i^{(k)}
\end{equation}

where $\mathbf{s}^{(k)}$ is a vector whose $i$th entry is the sum of the squares of the partials, with respect to $x_i$ , up to time step $k$,

\begin{equation}
    s_i^{(k)} = \sum_{j=1}^{k}\left(g_i^{(j)}\right)^2
\end{equation}
"""

class AdaGrad(Optimizer):
    def __init__(self, cost_f, lr=0.001, x=None, y=None):
        super(AdaGrad, self).__init__(cost_f=cost_f, lr=lr, x=x, y=y)
        self.sumsq_dx = 0
        self.sumsq_dy = 0

    def step(self, lr=None):
        epsilon = 1e-8
        if not lr:
            lr = self.lr

        # Tính đạo hàm của hàm mất mát
        f = self.cost_f.eval(self.x, self.y) # Gọi đánh giá tại điểm x, y
        dx = self.cost_f.df_dx(self.x, self.y) # 
        dy = self.cost_f.df_dy(self.x, self.y)

        self.sumsq_dx += dx**2
        self.sumsq_dy += dy**2

        self.x = self.x - (lr/(np.sqrt(epsilon + self.sumsq_dx)))*dx
        self.y = self.y - (lr/(np.sqrt(epsilon + self.sumsq_dy)))*dy

        return [self.x, self.y]
