import chex
import numpy as np
from typing import Union

from chapter04.simple_inventory_cap import (FiniteMarkovDecisionProcess,
                                            InventoryState,
                                            SimpleInventoryMDPCap)

from policy import FiniteDeterministicPolicy
from mk_process import FiniteMarkovRewardProcess, FiniteMarkovProcess
from gen_utils.distribution import Categorical

Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, IntLike] =\
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)

    fdp: FiniteDeterministicPolicy[InventoryState, IntLike] = \
        FiniteDeterministicPolicy(
            {InventoryState(on_hand=alpha, on_order=beta): user_capacity - (alpha + beta)
             for alpha in range(user_capacity + 1)
             for beta in range(user_capacity + 1 - alpha)}
    )

    print("Deterministic Policy Map")
    print("------------------------")
    print(fdp)

    implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
        si_mdp.apply_finite_policy(fdp)
    print("Implied MP Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(
        {s.state: Categorical({s1.state: p for s1, p in v.table().items()})
         for s, v in implied_mrp.transition_map.items()}
    ))

    print("Implied MRP Transition Reward Map")
    print("---------------------")
    print(implied_mrp)

    print("Implied MP Stationary Distribution")
    print("-----------------------")
    implied_mrp.display_stationary_distribution()
    print()

    print("Implied MRP Reward Function")
    print("---------------")
    implied_mrp.display_reward_function()
    print()

    print("Implied MRP Value Function")
    print("--------------")
    implied_mrp.display_value_function(gamma=user_gamma)
    print()

#    from pprint import pprint
#    from rl.dynamic_programming import evaluate_mrp_result
#    from rl.dynamic_programming import policy_iteration_result
#    from rl.dynamic_programming import value_iteration_result
#
#    print("Implied MRP Policy Evaluation Value Function")
#    print("--------------")
#    pprint(evaluate_mrp_result(implied_mrp, gamma=user_gamma))
#    print()
#
#    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
#    print("--------------")
#    opt_vf_pi, opt_policy_pi = policy_iteration_result(
#        si_mdp,
#        gamma=user_gamma
#    )
#    pprint(opt_vf_pi)
#    print(opt_policy_pi)
#    print()
#
#    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
#    print("--------------")
#    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
#    pprint(opt_vf_vi)
#    print(opt_policy_vi)
#    print()
