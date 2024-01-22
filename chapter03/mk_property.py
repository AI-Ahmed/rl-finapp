from dataclasses import dataclass
from typing import Mapping, Union, Optional
from loguru import logger

import jax
import jax.numpy as jnp


k = jax.random.PRNGKey(42)
handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass(frozen=True)
class Process1:
    @dataclass
    class State:
        price: Union[jnp.ndarray, Optional[int]]

        @property
        def dtype(self):
            return jnp.dtype(object)

    level_param: int  # level to which price mean-reverts
    alpha1: float  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: State) -> float:
        return 1. / (1 + jnp.exp(-self.alpha1 * (self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        up_move: jnp.ndarray = jax.random.binomial(k, 1, self.up_prob(state))
        return Process1.State(price=state.price + up_move * 2 - 1)

    @property
    def dtype(self):
        return jnp.dtype(object)


@dataclass
class Process2:
    @dataclass
    class State:
        price: Union[jnp.ndarray, Optional[int]]
        is_prev_mv_up: Optional[bool]

    alpha2: float = 0.75  # strength of reverse-pull (value in [0, 1])

    def up_prob(self, state: State) -> float:
        """
        The implementation the formula:
        $$\mathbb{P}[X_{t+1} = X_t + 1] = \begin{cases} 0.5(1 - \alpha(X_t - X_{t-1}) & t > 0 \\
        0.5 & t = 0\end{cases}$$

        We will add trick of using Mapping variable to map the difference between
        price_t and price_t-1
        """
        return 0.5 * (1 + self.alpha2 * (handy_map[state.is_prev_mv_up]))

    def next_state(self, state: State) -> State:
        up_move: jnp.ndarray = jax.random.binomial(k, 1, self.up_prob(state))
#        logger.debug(f"Prob: {up_move}, Price_t+1: {state.price + 2 * up_move - 1}")
        return Process2.State(
                price=state.price + 2 * up_move - 1,
                is_prev_mv_up=bool(up_move))
