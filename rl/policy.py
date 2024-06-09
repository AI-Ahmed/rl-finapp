from typing import Union, Generic, TypeVar, Callable, Mapping
from abc import ABC, abstractmethod
# from dataclasses import dataclass

import chex
from chex import dataclass
from distrax._src.utils import jittable

import numpy as np

from mk_process import NonTerminal, Distribution
from gen_utils.distribution import Constant, Choose, FiniteDistribution

Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]

A = TypeVar('A')
S = TypeVar('S')


class Policy(ABC, Generic[S, A]):
    @abstractmethod
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        pass


@dataclass(frozen=True)
class DeterministicPolicy(Policy[S, A]):
    action_for: Callable[[S], A]
    reorder_point: IntLike

    def act(self, state: NonTerminal[S]) -> Constant[A]:
        return Constant(value=self.action_for(state.state))


@dataclass(frozen=True)
class UniformPolicy(Policy[S, A]):
    valid_actions: Callable[[S], A]

    def act(self, state: NonTerminal[S]) -> Choose[A]:
        return Choose(value=self.valid_actions(state.state))


@dataclass(frozen=True)
class FinitePolicy(Policy[S, A]):
    policy_map: Mapping[S, FiniteDistribution[A]]

    def __repr__(self) -> str:
        display = ""
        for s, d in self.policy_map.items():
            display += f"For State {s}:\n"
            for a, p in d:
                display += f"\tDo Action {a} with Probability {p:.3f}\n"
        return display

    def act(self, state: NonTerminal[S]) -> FiniteDistribution[A]:
        return self.policy_map[state.state]


class FiniteDeterministicPolicy(jittable.Jittable, FinitePolicy[S, A]):
    action_for: Mapping[S, A]

    def __init__(self, action_for: Mapping[S, A]):
        super().__init__(policy_map={s: Constant(value=a)
                                     if not isinstance(a, Array)
                                     else Constant(value=a.item())
                                     for s, a in action_for.items()})
        object.__setattr__(self, "action_for", action_for)

    def __repr__(self) -> str:
        display = ""
        for s, a in self.action_for.items():
            display += f"For State {s}: Do Action {a}\n"
        return display
