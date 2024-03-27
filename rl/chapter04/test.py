from chapter04.simple_inventory import (SimpleInventoryDeterministicPolicy,
                                        SimpleInventoryMDPNoCap,
                                        SimpleInventoryStochasticPolicy)

if __name__ == '__main__':
    user_poisson_lambda = 2.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_reorder_point = 8
    user_reorder_point_poisson_mean = 8.0

    user_time_steps = 1000
    user_num_traces = 1000

    si_mdp_nocap = SimpleInventoryMDPNoCap(poisson_lambda=user_poisson_lambda,
                                           holding_cost=user_holding_cost,
                                           stockout_cost=user_stockout_cost)

    si_dp = SimpleInventoryDeterministicPolicy(
        reorder_point=user_reorder_point
    )

    oos_frac_dp = si_mdp_nocap.fraction_of_days_oos(policy=si_dp,
                                                    time_steps=user_time_steps,
                                                    num_traces=user_num_traces)
    print(
        f"Deterministic Policy yields {oos_frac_dp * 100:.2f}%"
        + " of Out-Of-Stock days"
    )

    si_sp = SimpleInventoryStochasticPolicy(
        reorder_point_poisson_mean=user_reorder_point_poisson_mean)

    oos_frac_sp = si_mdp_nocap.fraction_of_days_oos(policy=si_sp,
                                                    time_steps=user_time_steps,
                                                    num_traces=user_num_traces)
    print(
        f"Stochastic Policy yields {oos_frac_sp * 100:.2f}%"
        + " of Out-Of-Stock days"
    )
