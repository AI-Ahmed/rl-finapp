"""
This implementation has been taken from:
https://github.com/TikhonJelvis/RL-book/blob/master/rl/gen_utils/plot_funcs.py

It is an implementation for plotting the simulations that have been implemented
in the book of "Foundations of Reinforcement Learning in Applications of Finance".

Thanks to Tikhon Jelvis for the implementation side!
"""

import logging
import matplotlib.pyplot as mplt
import plotext as plt
import numpy as np

logging.getLogger(__name__)


def plot_list_of_curves(
    list_of_x_vals,
    list_of_y_vals,
    list_of_colors,
    list_of_curve_labels,
    x_label=None,
    y_label=None,
    title=None
):

#   mplt.figure(figsize=(11, 7))
    plt.plot_size(80, 24)
    plt.limit_size()

    plt_style = ['-', '--', '-.']
    for style, (i, x_vals) in zip(plt_style, enumerate(list_of_x_vals)):
        plt.plot(
            x_vals,
            list_of_y_vals[i],
            color=list_of_colors[i],
            label=list_of_curve_labels[i],
            style=style
        )
#    mplt.axes((
#        min(map(min, list_of_x_vals)),
#        max(map(max, list_of_x_vals)),
#        min(map(min, list_of_y_vals)),
#        max(map(max, list_of_y_vals))
#    ))
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)

    plt.grid(True)
#    mplt.legend()
    plt.show()


if __name__ == '__main__':
    x = np.arange(1, 100)
    y = [0.1 * x + 1.0, 0.001 * (x - 50) ** 2, np.log(x)]
    colors = ["r", "b", "g"]
    labels = ["Linear", "Quadratic", "Log"]
    plot_list_of_curves(
        [x, x, x],
        y,
        colors,
        labels,
        "X-Axis",
        "Y-Axis",
        "Test Plot"
    )
