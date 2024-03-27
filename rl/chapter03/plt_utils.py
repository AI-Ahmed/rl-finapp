"""
This implementation has been taken from:
https://github.com/TikhonJelvis/RL-book/blob/master/rl/gen_utils/plot_funcs.py

It is an implementation for plotting the simulations that have been implemented
in the book of "Foundations of Reinforcement Learning in Applications of Finance".

Thanks to Tikhon Jelvis for the implementation side!
"""

import logging
from loguru import logger

import chex
import numpy as np

from typing import Tuple, Sequence, Union
from collections import Counter
from operator import itemgetter

logging.getLogger(__name__)

FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]
PRNGKey = chex.PRNGKey


def plot_single_trace_all_processes(
    process1_trace: np.ndarray,
    process2_trace: np.ndarray,
    process3_trace: np.ndarray
) -> None:

    from gen_utils.plot_functions import plot_list_of_curves

    traces_len: IntLike = len(process1_trace)

    plot_list_of_curves(
        [range(traces_len)] * 3,
        [process1_trace, process2_trace, process3_trace],
        ["r-", "b--", "g-."],
        [
            r"Process 1 ($\alpha_1=0.25$)",
            r"Process 2 ($\alpha_2=0.75$)",
            r"Process 3 ($\alpha_3=1.0$)"
        ],
        "Time Steps",
        "Stock Price",
        "Single-Trace Simulation for Each Process"
    )


def get_terminal_histogram(price_traces: np.ndarray) -> Tuple[Sequence[IntLike], Sequence[IntLike]]:
    pairs: Sequence[Tuple[IntLike, IntLike]] = sorted(
            list(Counter(price_traces[:, -1]).items()), key=itemgetter(0))
    return [x for x, _ in pairs], [y for _, y in pairs]


def plot_distribution_at_time_all_processes(
    process1_traces: np.ndarray,
    process2_traces: np.ndarray,
    process3_traces: np.ndarray
) -> None:

    from gen_utils.plot_functions import plot_list_of_curves

    num_traces: IntLike = len(process1_traces)
    time_steps: IntLike = len(process1_traces[0]) - 1

    x1, y1 = get_terminal_histogram(process1_traces)
    x2, y2 = get_terminal_histogram(process2_traces)
    x3, y3 = get_terminal_histogram(process3_traces)

    logger.debug(f"x_1: {len(x1)}, x_2: {len(x2)}, x_3: {len(x3)}")
    logger.debug(f"y_1: {len(y1)}, y_2: {len(y2)}, y_3: {len(y3)}")

    plot_list_of_curves(
        [x1, x2, x3],
        [y1, y2, y3],
        ["r-", "b--", "g-."],
        [
            r"Process 1 ($\alpha_1=0.25$)",
            r"Process 2 ($\alpha_2=0.75$)",
            r"Process 3 ($\alpha_3=1.0$)"
        ],
        "Terminal Stock Price",
        "Counts",
        f"Terminal Price Counts (T={time_steps:d}, Traces={num_traces:d})"
    )
