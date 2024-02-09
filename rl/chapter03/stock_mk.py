from dataclasses import dataclass
from gen_utils.distribution import Categorical
from gen_utils.common_funcs import get_unit_sigmoid_func
from chapter03.mk_property import MarkovProcess, State, NonTerminal


@dataclass(frozen=True)
class StateMP3:
    num_up_moves: int
    num_down_moves: int


@dataclass
class StockPriceMP3(MarkovProcess[StateMP3]):

    alpha3: float = 1.0  # strength of reverse-pull (non-negative value)

    def up_prob(self, state: StateMP3) -> float:
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(self.alpha3)(
                state.num_down_moves / total
        ) if total else 0.5

    def transition(self, state: NonTerminal[StateMP3]) -> Categorical[State[StateMP3]]:
        up_p = self.up_prob(state.state)
        return Categorical({
            NonTerminal(StateMP3(
                state.state.num_down_moves + 1, state.state.num_down_moves
                )): up_p,
            NonTerminal(StateMP3(
                state.state.num_up_moves, state.state.num_down_moves + 1
                )): 1 - up_p
        })
