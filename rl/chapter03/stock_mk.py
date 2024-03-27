# from dataclasses import dataclass
import chex
from chex import dataclass
from typing import Union

import numpy as np

from gen_utils.distribution import Categorical
from gen_utils.common_funcs import get_unit_sigmoid_func
from mk_process import MarkovProcess, State, NonTerminal

Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]
PRNGKey = chex.PRNGKey


@dataclass(frozen=True)
class StateMP3:
    num_up_moves: IntLike
    num_down_moves: IntLike


@dataclass
class StockPriceMP3(MarkovProcess[StateMP3]):

    alpha3: FloatLike = 1.0  # strength of reverse-pull (non-negative value)

    def up_prob(self, state: StateMP3) -> FloatLike:
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(self.alpha3)(
                state.num_down_moves / total
        ) if total else 0.5

    def transition(self, state: NonTerminal[StateMP3]) -> Categorical[State[StateMP3]]:
        up_p = self.up_prob(state.state)
        return Categorical({
            NonTerminal(state=StateMP3(
                num_up_moves=state.state.num_down_moves + 1, num_down_moves=state.state.num_down_moves
                )): up_p,
            NonTerminal(state=StateMP3(
                num_up_moves=state.state.num_up_moves, num_down_moves=state.state.num_down_moves + 1
                )): 1 - up_p
        })
