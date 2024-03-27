from __future__ import annotations
from abc import ABC, abstractmethod
# from dataclasses import dataclass
from collections import defaultdict
from typing import Generic, Callable, TypeVar, Iterable, Sequence, Mapping, Set, Tuple, Dict, Union
from gen_utils.distribution import Distribution, FiniteDistribution, Categorical, SampledDistribution


import chex
from chex import dataclass
import numpy as np
from numba import jit, float32

Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]

S = TypeVar('S')
X = TypeVar('X')


class State(ABC, Generic[S]):
    state: S

    def on_non_terminal(
            self,
            f: Callable[[NonTerminal[S]], X],
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
            state = self.transition(state).sample(42)
            yield state


Transition = Mapping[NonTerminal[S], FiniteDistribution[State[S]]]


class FiniteMarkovProcess(MarkovProcess[S]):
    non_terminal_state: Sequence[NonTerminal[S]]
    transition_map: Transition[S]

    def __init__(self, transition_map: Mapping[S, FiniteDistribution[S]]):
        non_terminals: Set[S] = set(transition_map.keys())
        self.transition_map = {
                NonTerminal(s): Categorical(
                    {NonTerminal(s1) if s1 in non_terminals else Terminal(s1): p for s1, p in v}
                    ) for s, v in transition_map.items()
        }
        self.non_terminal_states = list(self.transition_map.keys())

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

    @jit
    def get_transtion_matrix(self) -> np.ndarray:
        """
        Computes the transtion probability matrix P
        """
        sz = np.int32(len(self.non_terminal_states))
        mat = np.zeros((sz, sz), dtype=np.float32)

        vect_trans_prob = np.vectorize(lambda s1, s2: self.transition_map(s1).probability(s2), otypes=float32)

        i, j = np.meshgrid(range(sz), range(sz))

        mat = vect_trans_prob(self.non_terminal_states[i], self.non_terminal_state[j])
        return mat

    def get_stationary_dist(self) -> FiniteDistribution[S]:
        eig_vals, eig_vecs = np.linalg.eigh(self.get_transtion_matrix().T)
        index_of_first_unit_eig_val = np.where(np.abs(eig_vals - 1 < 1e-8))[0][0]
        eig_vec_of_unit_eig_val = np.real(eig_vecs[:, index_of_first_unit_eig_val])

        return Categorical({
            self.non_terminal_states[i].state: ev
            for i, ev in enumerate(eig_vec_of_unit_eig_val / sum(eig_vec_of_unit_eig_val))
        })


@dataclass(frozen=True)
class TransitionStep(Generic[S]):
    state: NonTerminal[S]
    next_state: State[S]
    reward: FloatLike


class MarkovRewardProcess(MarkovProcess[S]):
    @abstractmethod
    def transition_reward(self, state: NonTerminal[S]) -> Distribution[Tuple[State[S], FloatLike]]:
        pass

    def simulate_reward(
            self,
            start_state_distribution: Distribution[NonTerminal[S]]
            ) -> Iterable[TransitionStep[S]]:
        state: State[S] = start_state_distribution.sample()
        reward: FloatLike = 0.

        while isinstance(state, NonTerminal):
            next_distribution = self.transition_reward(state)

            next_state, reward = next_distribution.sample()
            yield TransitionStep(state, next_state, reward)

            state = next_state

    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        distribution = self.transition_reward(state)

        def next_state(distribution=distribution):
            next_s, _ = distribution.sample()
            return next_s
        return SampledDistribution(next_state)


StateReward = FiniteDistribution[Tuple[State[S], FloatLike]]
RewardTransition = Mapping[NonTerminal[S], StateReward[S]]


class FiniteMarkovRewardProcess(FiniteMarkovProcess[S], MarkovRewardProcess[S]):

    transition_reward_map: RewardTransition[S]
    reward_function_vec: np.ndarray

    def __init__(self, transition_reward_map: Mapping[S, StateReward]):
        transition_map: Dict[S, FiniteDistribution[S]] = {}

        for state, trans in transition_reward_map.items():
            probabilities: Dict[S, FloatLike] = defaultdict(FloatLike)
            for (next_state, _), probability in trans:
                probabilities[next_state] += probability
            transition_map[state] = Categorical(probabilities)

        super().__ini__(transition_map)

        nt: Set[S] = set(transition_reward_map.keys())
        self.transition_reward_map = {
            NonTerminal(s): Categorical(
                {(NonTerminal(s1) if s1 in nt else Terminal(s1), r): p
                 for (s1, r), p in v}
            ) for s, v in transition_reward_map.items()
        }
        self.reward_function_vec = np.array([
            sum(probability * reward for (_, reward), probability in self.transition_reward_map[state])
            for state in self.non_terminal_states])

    def transition_reward(self, state: NonTerminal[S]) -> StateReward[S]:
        return self.transition_reward_map[state]

    def get_value_function_vec(self, gamma: FloatLike) -> np.ndarray:
        return np.linalg.solve(
            np.eye(len(self.non_terminal_states)) - gamma * self.get_transtion_matrix(),
            self.reward_function_vec
        )
