"""
This code is the JAX implementation of the `common_funcs.py` of the book
"""

from typing import Callable, Union

import chex
import numpy as np
import jax.numpy as jnp

FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]
PRNGKey = chex.PRNGKey

eps = 1e-8


def get_logistic_func(alpha: FloatLike) -> Callable[[FloatLike], FloatLike]:
    return lambda x: 1. / (1 + jnp.exp(-alpha * x))


def get_unit_sigmoid_func(alpha: FloatLike) -> Callable[[FloatLike], FloatLike]:
    return lambda x: 1. / (1 + (1 / jnp.where(x == 0, eps, x) - 1) ** alpha)


if __name__ == '__main__':
    from plot_functions import plot_list_of_curves
    alpha = [2.0, 1.0, 0.5]
    colors = ["r-", "b--", "g-."]
    labels = [(r"$\alpha$ = %.1f" % a) for a in alpha]
    logistics = [get_logistic_func(a) for a in alpha]
    x_vals = jnp.arange(-3.0, 3.01, 0.05)
    y_vals = [f(x_vals) for f in logistics]
    plot_list_of_curves(
        [np.asarray(x_vals)] * len(logistics),
        np.asarray(y_vals),
        colors,
        labels,
        title="Logistic Functions"
    )

    alpha = [2.0, 1.0, 0.5]
    colors = ["r-", "b--", "g-."]
    labels = [(r"$\alpha$ = %.1f" % a) for a in alpha]
    unit_sigmoids = [get_unit_sigmoid_func(a) for a in alpha]
    x_vals = jnp.arange(0.0, 1.01, 0.01)
    y_vals = [f(x_vals) for f in unit_sigmoids]
    plot_list_of_curves(
        [np.asarray(x_vals)] * len(logistics),
        np.asarray(y_vals),
        colors,
        labels,
        title="Unit-Sigmoid Functions"
    )
