import chex
from chex import dataclass
from typing import Tuple, Dict, Mapping, Union
from mk_d_process import FiniteMarkovDecisionProcess
from gen_utils.distribution import Categorical
from scipy.stats import poisson

import numpy as np

Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]


@dataclass(frozen=True)
class InventoryState:
    on_hand: IntLike
    on_order: IntLike

    def inventory_position(self) -> IntLike:
        return self.on_hand + self.on_order


InvOrderMapping = Mapping[
    InventoryState,
    Mapping[IntLike, Categorical[Tuple[InventoryState, FloatLike]]]
]


class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, IntLike]):

    def __init__(
        self,
        capacity: IntLike,
        poisson_lambda: FloatLike,
        holding_cost: FloatLike,
        stockout_cost: FloatLike
    ):
        self.capacity: IntLike = capacity
        self.poisson_lambda: FloatLike = poisson_lambda
        self.holding_cost: FloatLike = holding_cost
        self.stockout_cost: FloatLike = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[IntLike,
                                     Categorical[Tuple[InventoryState,
                                                       FloatLike]]]] = {}

        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state: InventoryState = InventoryState(on_hand=alpha, on_order=beta)
                ip: IntLike = state.inventory_position()
                base_reward: FloatLike = - self.holding_cost * alpha
                d1: Dict[IntLike, Categorical[Tuple[InventoryState, FloatLike]]] = {}

                for order in range(self.capacity - ip + 1):
                    sr_probs_dict: Dict[Tuple[InventoryState, FloatLike], FloatLike] =\
                        {(InventoryState(on_hand=ip - i, on_order=order), base_reward):
                         self.poisson_distr.pmf(i) for i in range(ip)}

                    probability: FloatLike = 1 - self.poisson_distr.cdf(ip - 1)
                    reward: FloatLike = base_reward - self.stockout_cost * \
                        (self.poisson_lambda - ip * (1 - self.poisson_distr.pmf(ip) / probability))
                    sr_probs_dict[(InventoryState(on_hand=0, on_order=order), reward)] = \
                        probability
                    d1[order] = Categorical(sr_probs_dict)

                d[state] = d1
        return d
