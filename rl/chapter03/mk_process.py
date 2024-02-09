from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Callable, TypeVar, Iterable, Sequence, Mapping, Set
from gen_utils.distribution import Distribution, FiniteDistribution, Categorical


S = TypeVar('S')
X = TypeVar('X')

class State(ABC, Generic[S]):
    state: S

    def on_non_terminal(
            self,
            f: Callable[['NonTerminal[S]'], X],
            default: X) -> X:
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S


class MarkovProcess(ABC, Generic[S]):
    """
        Markov Process with states of type S
    """
    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        pass

    def simulation(
            self,
            start_state_distribution: Distribution[NonTerminal[S]]
            ) -> Iterable[State[S]]:
        state: State[S] = start_state_distribution.sample()
        yield state

        while isinstance(state, NonTerminal):
            state = self.transition(state).sample()
            yield state


Transition = Mapping[NonTerminal[S], FiniteDistribution[State[S]]]


class FiniteMarkovProcess(MarkovProcess[S]):
    non_terminal_state: Sequence[NonTerminal[S]]
    transition_map: Transition[S]

    def __init__(self, transition_map: Mapping[S, FiniteDistribution[S]]):
        non_terminals: Set[S] = set(transition_map.keys())
        self.transition_map = {
                NonTerminal(s): Categorical(
                    {(NonTerminal(s1) if s1 in non_terminals) else Terminal(s1): p for s1, p in v}
                    ) for s, v in transition_map.items()
        }
        self.non_terminal_state = list(self.transition_map.keys())

    def __repr__(self) -> str:
        display = ""
        for s, d in self.transition_map.items():
            display += f"From State {s.state}: \n"
            for s1, p in d:
                opt = (
                        "Terminal State" if isinstance(s1, Terminal) else "State"
                )
                display += f"  To {opt} {s1.state} with Probability {p: .3f}\n"
        return display

    def transition(self, state: NonTerminal[S]) -> FiniteDistribution[State[S]]:
        return self.transition_map[state]
