from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

from optimizers.utils import run_optimizer

from optimizers import GradientDescent, GradientDescentMomentum, GradientDescentNesterovMomentum, AdaGrad, AdaDelta, RMSProp, Adam


def plot_contourf(cost_f, figsize=[10, 10], _show=True, is_save=True):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    x = np.arange(cost_f.xmin, cost_f.xmax, 0.1)
    y = np.arange(cost_f.ymin, cost_f.ymax, 0.1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([cost_f.eval(x, y)
                  for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    Gx, Gy = np.gradient(Z)  # gradients with respect to x and y
    G = (Gx**2.0+Gy**2.0)**.5  # gradient magnitude
    # print(G)
    N = G/G.max()  # normalize 0..1
    ax.contourf(X, Y, Z, cmap=plt.cm.get_cmap(plt.cm.afmhot), levels=np.linspace(zs.min(), zs.max(), 1000))
    plt.text(cost_f.x_optimum, cost_f.y_optimum, "x", color="b", size=20)
    # if is_save:
    #     # plt.savefig('destination_contourf.png', format='png', dpi=500)
    #     plt.savefig('destination_contourf.png', format='png')
    if _show:
        plt.show()
    return fig, ax


def plot_trajectories(trajectories_dict, cost_f, figsize=[10, 10],
                      filepath="test.gif", frames=10):

    fig, ax = plot_contourf(cost_f=cost_f, figsize=[10, 10], _show=False)

    global dots
    dots = []

    def update(frame_number):
        global dots
        ax = fig.gca()
        for sc in dots:
            sc.remove()
        dots = []

        for name, (x, y, c) in trajectories_dict.items():
            ax.plot(x[:frame_number], y[:frame_number],
                    color=c, zorder=1, linewidth=2)
            k = ax.scatter(x[frame_number], y[frame_number],
                           color=c, zorder=1, s=50)
            dots.append(k)

        plt.legend(trajectories_dict.keys())

    animation = FuncAnimation(fig, update, interval=10, frames=frames)
    animation.save(filepath, dpi=80,  writer=PillowWriter(fps=60))


def plot_cost_function_3d(cost_f, grain=0.01, figsize=[10, 6], is_save=True):
    # Controls the X region covered by the mesh grid
    x_grid = np.arange(cost_f.xmin, cost_f.xmax, grain)
    # Controls the Y region covered by the mesh grid
    y_grid = np.arange(cost_f.ymin, cost_f.ymax, grain)

    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.array([cost_f.eval(x_grid, y_grid) for x_grid,
                 y_grid in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    Gx, Gy = np.gradient(Z)  # gradients with respect to x and y

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, antialiased=False, shade=False,
                    rstride=5, cstride=1, linewidth=0, alpha = 1)
    ax.patch.set_facecolor('white')
    ax.view_init(elev=30., azim=70)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 15
    if is_save:
        plt.savefig('destination_cost_function_3d.png', format='png')
    plt.show()


def plot_evolution_charts(cost_f, errors, distance, xs, ys):
    plt.figure(figsize=[18, 6])
    plt.plot(errors, linewidth=3.0)
    #plt.title("Error (Z axis) evolution over time. Minimum error obtained in {0} iterations: {1}".format(len(errors), min(errors)))
    plt.title("Độ lỗi (Trục $Z$) theo thời gian. Độ lỗi cực tiểu thu được trong {0} lần lặp: {1}".format(
        len(errors), min(errors)))
    plt.xlabel("lần lặp")
    plt.ylabel("độ lỗi")
    plt.grid()
    plt.savefig('evolution_charts_01.png', format='png')
    plt.show()

    plt.figure(figsize=[18, 6])
    plt.semilogy(np.abs(np.array(errors) - cost_f.z_optimum), linewidth=3.0)
    #plt.title("Log-error (Z axis) evolution over time. Minimum error obtained in {0} iterations: {1}".format(len(errors), min(errors)))
    plt.title("Log của độ lỗi (Trục $Z$) theo thời gian. Độ lỗi cực tiểu thu được trong {0} lần lặp: {1}".format(
        len(errors), min(errors)))
    plt.xlabel("lần lặp")
    plt.ylabel("log(độ lỗi)")
    plt.grid()
    plt.savefig('evolution_charts_02.png', format='png')
    plt.show()

    plt.figure(figsize=[18, 6])
    plt.plot(distance, linewidth=3.0)
    plt.ylim([0, max(distance)])
    #plt.title("Distance to minimum evolution over time. Minimum distance obtained in {0} iterations: {1}".format(len(errors), min(errors)))
    plt.title("Khoảng cách đến vị trí cực tiểu theo thời gian. Khoảng cách cực tiểu thu được trong {0} lần lặp: {1}".format(
        len(errors), min(errors)))
    plt.xlabel("lần lặp")
    plt.ylabel("khoảng cách")
    plt.grid()
    plt.savefig('evolution_charts_03.png', format='png')
    plt.show()


def plot_cinematics_charts(xs, ys):
    plt.figure(figsize=[18, 6])
    plt.subplot(131)
    plt.plot(xs, linewidth=3.0)
    plt.title("Đánh giá tham số $X$")
    plt.xlabel("lần lặp")
    plt.ylabel("$x$")
    plt.grid()
    plt.subplot(132)
    plt.plot(ys, linewidth=3.0)
    plt.title("Đánh giá tham số $Y$")
    plt.xlabel("lần lặp")
    plt.ylabel("$y$")
    plt.grid()
    plt.subplot(133)
    plt.plot(xs, ys, linewidth=3.0)
    plt.title("Đánh giá $x/ y$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid()
    plt.savefig('cinematics_charts_01.png', format='png')
    plt.show()

    vel_xs = np.diff(xs)
    vel_ys = np.diff(ys)
    plt.figure(figsize=[18, 6])
    plt.subplot(131)
    plt.plot(np.abs(vel_xs), linewidth=3.0)
    plt.title("Tham số vận tốc $X$ (Động lượng - momentum; $v_x$)")
    plt.xlabel("lần lặp")
    plt.ylabel("$v_x$")
    plt.ylim(0, 1.05*max(vel_xs))
    plt.grid()

    plt.subplot(132)
    plt.plot(np.abs(vel_ys), linewidth=3.0)
    plt.title("Tham số vận tốc $Y$ (Động lượng - momentum; $v_y$)")
    plt.xlabel("lần lặp")
    plt.ylabel("$v_y$")
    plt.ylim(0, 1.05*max(vel_ys))
    plt.grid()

    plt.subplot(133)
    plt.plot(np.sqrt(np.array(vel_xs)**2 + np.array(vel_ys)**2), linewidth=3.0)
    plt.title("Vận tốc tuyệt đối - Absolute velocity ($\sqrt{v_x^2 + v_y^2}$)")
    plt.xlabel("lần lặp")
    plt.ylabel("$v$")
    plt.ylim(0, 1.05*max(np.sqrt(np.array(vel_xs)**2 + np.array(vel_ys)**2)))
    plt.grid()
    plt.savefig('cinematics_charts_02.png', format='png')
    plt.show()


def global_compare(cost_fnc, iterations, learning_rate, figsize=[18, 6]):
    print("starting gradient descent.")
    errors_sgd, distance_sgd, _, _ = run_optimizer(opt=GradientDescent(
        cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    print("end gradient descent.")

    print("starting gradient descent with momentum.")
    errors_momentum, distance_momentum, _, _ = run_optimizer(opt=GradientDescentMomentum(
        cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    print("end gradient descent with momentum.")

    print("starting gradient descent with nesterov momentum.")
    errors_nesterov, distance_nesterov, _, _ = run_optimizer(opt=GradientDescentNesterovMomentum(
        cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    print("end gradient descent with nesterov momentum.")

    print("starting adagrad.")
    errors_adagrad, distance_adagrad, _, _ = run_optimizer(opt=AdaGrad(
        cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    print("end adagrad.")

    print("starting adadelta.")
    errors_adadelta, distance_adadelta, _, _ = run_optimizer(opt=AdaDelta(
        cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    print("end adadelta.")

    print("starting rmsprop.")
    errors_rmsprop, distance_rmsprop, _, _ = run_optimizer(opt=RMSProp(
        cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    print("end rmsprop.")

    print("starting adam.")
    errors_adam, distance_adam, _, _ = run_optimizer(opt=Adam(
        cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    print("end adam.")

    plt.figure(figsize=figsize)
    plt.plot(errors_sgd, color="b")
    plt.plot(errors_momentum, color="orange")
    plt.plot(errors_nesterov, color="green")
    plt.plot(errors_adagrad, color="tomato")
    plt.plot(errors_rmsprop, color="purple")
    plt.plot(errors_adadelta, color="turquoise")
    plt.plot(errors_adam, color="k")
    plt.title("So sánh độ lỗi giữa các optimizers")
    plt.ylabel("độ lỗi")
    plt.xlabel("thời gian")
    plt.legend(labels=["GradientDescent", "GradientDescentMomentum",
               "GradientDescentNesterovMomentum", "AdaGrad", "RMSProp", "AdaDelta", "Adam"])
    plt.xlim([0, iterations])
    plt.grid()
    plt.savefig('global_compare_01.png', format='png')
    plt.show()

    plt.figure(figsize=figsize)
    plt.semilogy(errors_sgd, color="b")
    plt.semilogy(errors_momentum, color="orange")
    plt.semilogy(errors_nesterov, color="green")
    plt.semilogy(errors_adagrad, color="tomato")
    plt.semilogy(errors_rmsprop, color="purple")
    plt.semilogy(errors_adadelta, color="turquoise")
    plt.semilogy(errors_adam, color="k")
    plt.title("So sánh độ lỗi giữa các optimizers (log-scale trục y)")
    plt.ylabel("độ lỗi")
    plt.xlabel("thời gian")
    plt.legend(labels=["GradientDescent", "GradientDescentMomentum",
               "GradientDescentNesterovMomentum", "AdaGrad", "RMSProp", "AdaDelta", "Adam"])
    plt.xlim([0, iterations])
    plt.grid()
    plt.savefig('global_compare_02.png', format='png')
    plt.show()

    plt.figure(figsize=figsize)
    plt.plot(distance_sgd, color="b")
    plt.plot(distance_momentum, color="orange")
    plt.plot(distance_nesterov, color="green")
    plt.plot(distance_adagrad, color="tomato")
    plt.plot(distance_rmsprop, color="purple")
    plt.plot(distance_adadelta, color="turquoise")
    plt.plot(distance_adam, color="k")
    plt.title(
        "So sánh khoảng cách đến điểm cực tiểu toàn cục giữa các optimizers với nhau")
    plt.ylabel("độ lỗi")
    plt.xlabel("thời gian")
    plt.legend(labels=["GradientDescent", "GradientDescentMomentum",
               "GradientDescentNesterovMomentum", "AdaGrad", "RMSProp", "AdaDelta", "Adam"])
    plt.xlim([0, iterations])
    plt.grid()
    plt.savefig('global_compare_03.png', format='png')
    plt.show()


def make_gif(cost_fnc, iterations, learning_rate, output="trajectories.gif"):
    print("starting gradient descent.")
    _, _, xs_sgd, ys_sgd = run_optimizer(opt=GradientDescent(
        cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    print("end gradient descent.")

    # print("starting gradient descent with momentum.")
    # _, _, xs_momentum, ys_momentum = run_optimizer(opt=GradientDescentMomentum(
    #     cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    # print("end gradient descent with momentum.")

    # print("starting gradient descent with nesterov momentum.")
    # _, _, xs_nesterov, ys_nesterov = run_optimizer(opt=GradientDescentNesterovMomentum(
    #     cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    # print("end gradient descent with nesterov momentum.")

    # print("starting adagrad.")
    # _, _, xs_adagrad, ys_adagrad = run_optimizer(opt=AdaGrad(
    #     cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    # print("end adagrad.")

    # print("starting adadelta.")
    # _, _, xs_adadelta, ys_adadelta = run_optimizer(opt=AdaDelta(
    #     cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    # print("end adadelta.")

    # print("starting rmsprop.")
    # _, _, xs_rmsprop, ys_rmsprop = run_optimizer(opt=RMSProp(
    #     cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    # print("end rmsprop.")

    # print("starting adam.")
    # _, _, xs_adam, ys_adam = run_optimizer(opt=Adam(
    #     cost_f=cost_fnc, lr=learning_rate), cost_f=cost_fnc, iterations=iterations)
    # print("end adam.")

    trajectories_dict = {"GD": (xs_sgd, ys_sgd, "b"),
                         #  "GD Momentum": (xs_momentum, ys_momentum, "orange"),
                         #  "GD Nesterov": (xs_nesterov, ys_nesterov, "green"),
                         #  "AdaGrad": (xs_adagrad, ys_adagrad, "tomato"),
                         #  "RMSprop": (xs_rmsprop, ys_rmsprop, "purple"),
                         #  "AdaDelta": (xs_adadelta, ys_adadelta, "turquoise"),
                         #  "Adam": (xs_adam, ys_adam, "k"),
                         }

    plot_trajectories(trajectories_dict, cost_fnc, figsize=[
                      10, 10], filepath=output, frames=700)
