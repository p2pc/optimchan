from .base import Optimizer

"""
Issue of momentum: do not slow down enough at the bottom of a valley and tend to overshoot the valley floor. Nesterov momentum: modifies to use the gradient at the projected future position.

\begin{equation}
	\textbf{v}^{(k+1)} = \beta\textbf{v}^{(k)} - \alpha\nabla f(\textbf{x}^{(k)} + \beta\textbf{v}^{(k)})
\end{equation}

\begin{equation}
	\textbf{x}^{(k+1)} = \textbf{x}^{(k)} + \textbf{v}^{(k+1)}
\end{equation}
"""

class GradientDescentNesterovMomentum(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta=0.9, x=None, y=None):
        super(GradientDescentNesterovMomentum, self).__init__(
            cost_f=cost_f, lr=lr, x=x, y=y, beta=beta)
        self.vx = None

    def step(self, lr=None, beta=None):
        f = self.cost_f.eval(self.x, self.y)

        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        if type(lr) == type(None):
            lr = self.lr
        if type(beta) == type(None):
            beta = self.beta
        if type(self.vx) == type(None) or type(self.vy) == type(None):
            self.vx = lr * dx
            self.vy = lr * dy
        else:
            dx_in_vx = self.cost_f.df_dx(
                self.x-beta*self.vx, self.y-beta*self.vy)
            dy_in_vy = self.cost_f.df_dy(
                self.x-beta*self.vx, self.y-beta*self.vy)

            self.vx = beta * self.vx + lr * dx_in_vx
            self.vy = beta * self.vy + lr * dy_in_vy

        self.x += - self.vx
        self.y += - self.vy

        return [self.x, self.y]
