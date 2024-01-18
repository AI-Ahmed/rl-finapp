import itertools
from typing import Generator

import jax
import numpy as np
import jax.numpy as jnp

from mk_property import Process1


def simulation(process, start_state) -> Generator[int, None, None]:
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


# Using Jax
def jx_simulation(process: Process1, start_state: Process1.State, time_steps: int) -> jnp.ndarray:
    def update_state(carry, _):
        return process.next_state(carry), None

    final_state, _ = jax.lax.scan(update_state, start_state, xs=None, length=time_steps)

    return jnp.concatenate([start_state[None, :], final_state])


@jax.jit
def jx_process1_price_traces(
        start_price: int,
        level_param: int,
        alpha1: float,
        time_steps: int,
        num_traces: int) -> jnp.ndarray:
    def jx_loop(i):
        return jx_simulation(process, start_state, time_steps)

    process = Process1(level_param=level_param, alpha1=alpha1)
    start_state = Process1.State(price=start_price)

    traces = jax.vmap(lambda _: jx_simulation(process, start_state, time_steps))(jnp.arange(0, num_traces,
        dtype=jnp.int32))

    return traces
