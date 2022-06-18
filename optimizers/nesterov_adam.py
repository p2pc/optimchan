from .base import Optimizer
import numpy as np


class NesterovAdam(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta_1=0.9, beta_2=0.999, x=None, y=None):
        super(NesterovAdam, self).__init__(cost_f, lr, x, y, beta_1=beta_1, beta_2=beta_2)
        self.m_x, self.m_y, self.v_x, self.v_y, self.t = 0, 0, 0, 0, 0

    def step(self, lr=None):
        self.t += 1
        epsilon = 1e-8
        if not lr:
            lr = self.lr
        # derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        dx_hat = dx/(1-self.beta_1**self.t)
        dy_hat = dy/(1-self.beta_1**self.t)

        self.m_x = self.beta_1*self.m_x + (1-self.beta_1)*dx
        self.m_y = self.beta_1*self.m_y + (1-self.beta_1)*dy
        self.v_x = self.beta_2*self.v_x + (1-self.beta_2)*(dx**2)
        self.v_y = self.beta_2*self.v_y + (1-self.beta_2)*(dy**2)

        m_x_hat = self.m_x/(1-self.beta_1**self.t)
        m_y_hat = self.m_y/(1-self.beta_1**self.t)
        v_x_hat = self.v_x/(1-self.beta_2**self.t)
        v_y_hat = self.v_y/(1-self.beta_2**self.t)

        m_x_dash = (1-self.beta_1)*dx_hat + self.beta_1*m_x_hat
        m_y_dash = (1-self.beta_1)*dy_hat + self.beta_1*m_y_hat

        self.x = self.x - (lr*m_x_dash)/(np.sqrt(v_x_hat)+epsilon)
        self.y = self.y - (lr*m_y_dash)/(np.sqrt(v_y_hat)+epsilon)
        return [self.x, self.y]
