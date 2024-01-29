import logging
from dataclasses import dataclass
from typing import Mapping, Union, Optional, Iterator
from loguru import logger

import jax
import numpy as np
import jax.numpy as jnp

logging.getLogger(__name__)

handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}
eps: float = 1e-8


@dataclass
class Process1:
    @dataclass
    class State:
        price: Union[jnp.ndarray, Optional[int]]

        @property
        def dtype(self):
            return jnp.dtype(object)

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)
    seed: int = 42

    def init_keys(self, seed: int):
        rng = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(rng)
        return key

    def up_prob(self, state: State) -> float:
        return 1. / (1 + jnp.exp(-self.alpha1 * (self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        key = self.init_keys(self.seed)
        up_move: jnp.ndarray = jax.random.binomial(key, 1, self.up_prob(state))
        self.seed = key[0]
        return Process1.State(price=state.price + up_move * 2 - 1)


@dataclass
class Process2:
    @dataclass
    class State:
        price: Union[jnp.ndarray, Optional[int]]
        is_prev_mv_up: Optional[bool]

    alpha2: float = 0.75  # strength of reverse-pull (value in [0, 1])
    seed: int = 42

    def init_keys(self, seed: int):
        rng = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(rng)
        return key

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
        k = self.init_keys(self.seed)
        up_move: jnp.ndarray = jax.random.binomial(k, 1, self.up_prob(state))
        self.seed = k[0]
        return Process2.State(
                price=state.price + 2 * up_move - 1,
                is_prev_mv_up=bool(up_move))


@dataclass
class Process3:
    @dataclass
    class State:
        num_up_move: int
        num_down_move: int

    alpha3: float = 1.  # strength of reverse-pull (non-negative value)
    seed: int = 42

    def init_keys(self, seed: int):
        rng = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(rng)
        return key

    def up_prob(self, state: State) -> float:
        total = state.num_up_move + state.num_down_move
        nd_mv = state.num_down_move
        x = total / jnp.where(nd_mv == 0, eps, nd_mv)
        return 1. / 1 + (1/x - 1) ** self.alpha3 if total else 0.5

    def next_state(self, state: State) -> State:
        k = self.init_keys(self.seed)
        up_move: int = jax.random.binomial(k, 1, self.up_prob(state))
        self.seed = k[0]
        return Process3.State(
                num_up_move=state.num_up_move + up_move,
                num_down_move=state.num_down_move + 1 - up_move)
