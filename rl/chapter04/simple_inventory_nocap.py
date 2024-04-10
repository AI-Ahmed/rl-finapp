import random
import itertools
from typing import Union, TypeVar, Tuple, Iterator

import chex
from chex import dataclass

import jax
import numpy as np

from scipy.stats import poisson

from gen_utils.distribution import Constant, SampledDistribution
# from dataclasses import dataclass
from mk_process import Terminal, NonTerminal, State
from mk_d_process import MarkovDecisionProcess, MarkovRewardProcess
from policy import Policy, DeterministicPolicy

Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]

A = TypeVar('A', bound=IntLike)
S = TypeVar('S', bound=Union[Terminal, NonTerminal, IntLike, Array])


@dataclass(frozen=True)
class InventoryState:
    on_hand: IntLike  # alpha
    on_order: IntLike  # beta

    def inventory_position(self) -> IntLike:
        return self.on_hand + self.on_order


class SimpleInventoryDeterministicPolicy(
    DeterministicPolicy[InventoryState, IntLike]
):
    def __init__(self, reorder_point: IntLike):
        def action_for(s: InventoryState) -> IntLike:
            return max(reorder_point - s.inventory_position(), 0)

        super().__init__(action_for=action_for, reorder_point=reorder_point)


class SimpleInventoryStochasticPolicy(Policy[InventoryState, IntLike]):
    def __init__(self, reorder_point_poisson_mean: FloatLike):
        self.reorder_point_poisson_mean: FloatLike = reorder_point_poisson_mean

    def act(self, state: NonTerminal[InventoryState]) -> SampledDistribution[IntLike]:
        def action_func(state=state) -> IntLike:
            key = jax.random.PRNGKey(random.randint(42, 1234))
            reorder_point_sample: IntLike = jax.random.poisson(key,
                                                               self.reorder_point_poisson_mean)
            return max(reorder_point_sample - state.state.inventory_position(), 0)
        return SampledDistribution(action_func)


@dataclass(frozen=True)
class SimpleInventoryMDPNoCap(MarkovDecisionProcess[InventoryState, IntLike]):
    poisson_lambda: FloatLike
    holding_cost: FloatLike
    stockout_cost: FloatLike

    def step(
        self,
        state: NonTerminal[InventoryState],
        order: IntLike,
    ) -> SampledDistribution[Tuple[State[InventoryState], FloatLike]]:

        def sample_next_state_reward(
            state=state,
            order=order,
        ) -> Tuple[State[InventoryState], FloatLike]:
            rng = jax.random.PRNGKey(random.randint(42, 1234))
            demand_sample: IntLike = jax.random.poisson(rng, self.poisson_lambda)
            ip: IntLike = state.state.inventory_position()
            next_state: InventoryState = InventoryState(
                    on_hand=max(ip - demand_sample, 0),
                    on_order=order
            )
            reward: FloatLike = - self.holding_cost * state.state.on_hand\
                    - self.stockout_cost * max(demand_sample - ip, 0)
            return NonTerminal(state=next_state), reward

        return SampledDistribution(sample_next_state_reward)

    def actions(self, state: NonTerminal[InventoryState]) -> Iterator[int]:
        return itertools.count(start=0, step=1)

    def fraction_of_days_oos(
        self,
        policy: Policy[InventoryState, IntLike],
        time_steps: IntLike,
        num_traces: IntLike
    ) -> float:
        impl_mrp: MarkovRewardProcess[InventoryState] =\
            self.apply_policy(policy)
        count: int = 0
        high_fractile: IntLike = np.int32(poisson(self.poisson_lambda).ppf(0.98))
        start: InventoryState = random.choice(
            [InventoryState(on_hand=i, on_order=0) for i in range(high_fractile + 1)])

        for _ in range(num_traces):
            steps = itertools.islice(
                impl_mrp.simulate_reward(Constant(value=NonTerminal(state=start))),
                time_steps
            )
            for step in steps:
                if step.reward < -self.holding_cost * step.state.state.on_hand:
                    count += 1

        return float(count) / (time_steps * num_traces)
