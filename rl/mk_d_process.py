from abc import ABC, abstractmethod
from typing import Tuple, Union, Iterable, Generic, TypeVar

import chex
from chex import dataclass

# import jax.numpy as jnp
import numpy as np

from gen_utils.distribution import Distribution
from mk_process import NonTerminal, MarkovRewardProcess, State
from policy import Policy

Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]

A = TypeVar('A', bound=IntLike)
S = TypeVar('S', bound=Union[IntLike, Array])


@dataclass(frozen=True)
class TransitionStep(Generic[S, A]):
    state: NonTerminal[S]
    action: A
    next_state: State[S]
    reward: FloatLike


class MarkovDecisionProcess(ABC, Generic[S, A]):
    @abstractmethod
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        pass

    def step(
        self,
        state: NonTerminal[S],
        action: A
    ) -> Distribution[Tuple[State[S]], FloatLike]:
        pass

    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        mdp = self

        class RewardProcess(MarkovRewardProcess[S]):
            def transition_reward(
                self,
                state: NonTerminal[S],
            ) -> Distribution[Tuple[State, FloatLike]]:

                actions: Distribution[A] = policy.act(state)
                return actions.apply(lambda a: mdp.step(state, a))
        return RewardProcess()

    def simulate_actions(
            self,
            start_states: Distribution[NonTerminal[S]],
            policy: Policy[S, A]
    ) -> Iterable[TransitionStep[S, A]]:
        state: State[S] = start_states.sample()

        while isinstance(state, NonTerminal):
            action_distribution = policy.act(state)

            action = action_distribution.sample()
            next_distribution = self.step(state, action)

            next_state, reward = next_distribution.sample()

            yield TransitionStep(state, action, next_state, reward)
            state = next_state
