from typing import Union, Generic, TypeVar, Callable
from abc import ABC, abstractmethod
# from dataclasses import dataclass

import chex
from chex import dataclass

import numpy as np

from mk_process import NonTerminal, Distribution
from gen_utils.distribution import Constant, Choose

Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]

A = TypeVar('A', bound=IntLike)
S = TypeVar('S', bound=Union[IntLike, Array])


class Policy(ABC, Generic[S, A]):
    @abstractmethod
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        pass


@dataclass(frozen=True)
class DeterministicPolicy(Policy[S, A]):
    action_for: Callable[[S], A]

    def act(self, state: NonTerminal[S]) -> Constant[A]:
        return Constant(self.action_for(state.state))


@dataclass(frozen=True)
class UniformPolicy(Policy[S, A]):
    valid_actions: Callable[[S], A]

    def act(self, state: NonTerminal[S]) -> Choose[A]:
        return Choose(self.valid_actions(state.state))
