import itertools
from typing import Generator

import jax
import numpy as np
import jax.numpy as jnp

from mk_property import Process1, Process2


def simulation(process, start_state) -> Generator[int, None, None]:
    # Based on the simulation, the first price produced will be the `start_price` (e.g., 100).
    state = start_state
    while True:
        yield state
        state = process.next_state(state)


def process1_price_traces(
        start_price: int,
        level_param: int,
        alpha1: float,
        time_steps: int,
        num_traces: int) -> jnp.ndarray:
    process = Process1(level_param=level_param, alpha1=alpha1)
    start_state = Process1.State(price=start_price)
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
            simulation(process, start_state),
            time_steps + 1)), float)
        for _ in range(num_traces)])


def process2_price_traces(
        start_price: int,
        alpha2: float,
        time_steps: int,
        num_traces: int) -> jnp.ndarray:
    process = Process2(alpha2=alpha2)
    start_state = Process2.State(price=start_price, is_prev_mv_up=None)
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
            simulation(process, start_state),
            time_steps + 1
        )), float) for _ in range(num_traces)])
