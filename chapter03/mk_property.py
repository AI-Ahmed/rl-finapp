from dataclasses import dataclass
from typing import Mapping, Union, Optional

import jax
import jax.numpy as jnp

k = jax.random.PRNGKey(42)
handy_map: Mapping[Optional[bool], int]


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
    class State
        price: Union[jnp.ndarray, Optional[int]]
        is_prev_mv_up: Optional[bool]

        @property
        def dtype(self):
            return jnp.dtype(object)

    alpha2: float = 0.75  # strength of reverse-pull (value in [0, 1])

    def up_prob(self, state: State) -> float:
        """
        To implement the formula of
            $$\mathbb{P}[X_{t+1} = X_t + 1] = {}$$
        
        The first condition of t = 0, we produce infinite prob of 0 due to the presistance of the reserved
        random generation number.

        In the second part of the conditional formula, following the initial `Null` price (X_0, Null), will
        also produce an output of 0.5, which bring us back to the first condition. And, if we followed the 
        assumption of assigning the inital price `price_0 = 0` this would produce a baised results.


        """
        # We won't implement the conditional aspect of t = 1 due to the random generation
        # that consists of producing only zeros if p = 0.5
        # Now, if we also applied the formula of t > 0
        return 0.5 * (1 - self.alpha2 * (self.price))

    def next_state(self, state: State) -> State:
        up_move: jnp.ndarray = jax.random.binomial(k, 1, self.up_prob(state))
