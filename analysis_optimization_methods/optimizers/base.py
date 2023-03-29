class Optimizer:
    def __init__(self, cost_f, lr, x, y, **kwargs):
        """Hàm khởi tạo

        Args:
            cost_f (_type_: class): Hàm chi phí (Hàm mất mát) cần phải tối ưuiption_
            lr (_type_): Tốc độ học
            x (_type_: float): Tọa độ x
            y (_type_: float): Tọa độ y
        """
        super(Optimizer, self).__init__()
        self.lr = lr
        self.cost_f = cost_f
        if x == None or y == None:
            self.x = self.cost_f.x_start
            self.y = self.cost_f.y_start
        else:
            self.x = x
            self.y = y

        self.__dict__.update(kwargs)

    def step(self, lr):
        raise NotImplementedError()
