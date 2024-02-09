import itertools
import logging
from typing import Generator

import numpy as np

from chapter03.processes import Process1, Process2, Process3
from chapter03.mk_property import NonTerminal
from chapter03.stock_mk import StockPriceMP3, StateMP3
from gen_utils.distribution import Constant

logging.getLogger(__name__)


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
        num_traces: int) -> np.ndarray:
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
        num_traces: int) -> np.ndarray:
    process = Process2(alpha2=alpha2)
    start_state = Process2.State(price=start_price, is_prev_mv_up=None)
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
            simulation(process, start_state),
            time_steps + 1
        )), float) for _ in range(num_traces)])


def process3_price_traces(
        start_price: int,
        alpha3: float,
        time_steps: int,
        num_traces: int) -> np.ndarray:
    process = Process3(alpha3=alpha3)
    start_state = Process3.State(num_up_move=0, num_down_move=0)
    return np.vstack([
        np.fromiter((start_price + s.num_up_move - s.num_down_move
                    for s in itertools.islice(
                     simulation(process, start_state), time_steps + 1)), float)
        for _ in range(num_traces)])


def mkprocess3_price_traces(
    start_price: int,
    alpha3: float,
    time_steps: int,
    num_traces: int) -> np.ndarray:

    mp = StockPriceMP3(alpha3=alpha3)
    start_state_distribution = Constant(
            NonTerminal(StateMP3(num_up_moves=0, num_down_moves=0))
    )
    return np.vstack([
        np.fromiter((start_price + s.state.num_up_moves - s.state.num_down_moves
                    for s in itertools.islice(mp.simulation(start_state_distribution),
                        time_steps + 1)), float)
        for _ in range(num_traces)])
