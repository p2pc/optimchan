import numpy as np


def run_optimizer(opt, cost_f, iterations, *args, **kwargs):
    """run_optimizer function

    Args:
        opt (_type_: class Optimizer): Bộ tối ưu
        cost_f (_type_: class): Hàm chi phí (Hàm mất mát) cần phải tối ưu
        iterations (_type_: int): Số lần lặp

    Returns:
        errors (_type_: list): Độ lỗi
        distance (_type_: float): Khoảng cách đến vị trí tối ưu
        xs (_type_: float): Danh sách tọa độ x
        ys (_type_: float): Danh sách tọa độ y
    """
    errors = [cost_f.eval(cost_f.x_start, cost_f.y_start)]
    xs, ys = [cost_f.x_start], [cost_f.y_start]
    for epochs in range(iterations):
        x, y = opt.step(*args, **kwargs)
        xs.append(x)
        ys.append(y)
        errors.append(cost_f.eval(x, y))
    distance = np.sqrt((np.array(xs)-cost_f.x_optimum) **2 + (np.array(ys)-cost_f.y_optimum)**2)
    return errors, distance, xs, ys
