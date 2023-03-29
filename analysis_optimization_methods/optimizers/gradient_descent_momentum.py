from .base import Optimizer

"""
Gradient descent will take a long time to traverse a nearly flat surface. Allowing momentum to accumulate is one way to speed progress
\begin{equation}
	\textbf{v}^{(k+1)} = \beta\textbf{v}^{(k)} - \alpha\textbf{g}^{(k)}
\end{equation}

\begin{equation}
	\textbf{x}^{(k+1)} = \textbf{x}^{(k)} + \textbf{v}^{(k+1)}
\end{equation}
"""

class GradientDescentMomentum(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta=0.9, x=None, y=None):
        super(GradientDescentMomentum, self).__init__(
            cost_f=cost_f, lr=lr, x=x, y=y, beta=beta)
        self.vx = 0
        self.vy = 0

    def step(self, lr=None, beta=None):
        if type(lr) == type(None):
            lr = self.lr
        if type(beta) == type(None):
            beta = self.beta
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.vx = beta * self.vx + lr * dx
        self.vy = beta * self.vy + lr * dy
        self.x += - self.vx
        self.y += - self.vy

        return [self.x, self.y]
