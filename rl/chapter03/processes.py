import logging
import random
# from dataclasses import dataclass
from typing import Mapping, Union, Optional
# from loguru import logger

import jax
import chex
from chex import dataclass

import numpy as np
import jax.numpy as jnp

logging.getLogger(__name__)

Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]
PRNGKey = chex.PRNGKey

handy_map: Mapping[Optional[bool], IntLike] = {True: -1, False: 1, None: 0}
eps: FloatLike = 1e-8


@dataclass
class Process1:
    @dataclass
    class State:
        price: Union[FloatLike, IntLike]

    level_param: IntLike  # level to which price mean-reverts
    alpha1: FloatLike = 0.25  # strength of mean-reversion (non-negative value)

    def generate_keys(self, seed: IntLike, n: IntLike) -> PRNGKey:
        """
        This function generates a set of random keys using JAX's deterministic
        random number generation system. This system ensures reproducible results
        across different runs by stabilizing numerical outputs.

        Parameters:
        - seed (int): The seed number used for randomization.

        Returns:
        - PRNGKey: An array of generated random keys.
        """
        rng = jax.random.PRNGKey(seed)
        key, *_ = jax.random.split(rng, n)
        return key

    def up_prob(self, state: State) -> FloatLike:
        return 1. / (1 + jnp.exp(-self.alpha1 * (self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        key = self.generate_keys(random.randint(1, 1010), 3)
        up_move: IntLike = jax.random.binomial(key, 1, self.up_prob(state))
        return Process1.State(price=state.price + up_move * 2 - 1)


@dataclass
class Process2:
    @dataclass
    class State:
        price: Union[FloatLike, IntLike]
        is_prev_mv_up: Optional[bool]

    alpha2: FloatLike = 0.75  # strength of reverse-pull (value in [0, 1])

    def generate_keys(self, seed: IntLike, n: IntLike) -> PRNGKey:
        """
        This function generates a set of random keys using JAX's deterministic
        random number generation system. This system ensures reproducible results
        across different runs by stabilizing numerical outputs.

        Parameters:
        - seed (int): The seed number used for randomization.

        Returns:
        - PRNGKey: An array of generated random keys.
        """
        rng = jax.random.PRNGKey(seed)
        key, *_ = jax.random.split(rng, n)
        return key

    def up_prob(self, state: State) -> FloatLike:
        """
        The implementation the formula:
        $$\mathbb{P}[X_{t+1} = X_t + 1] = \begin{cases} 0.5(1 - \alpha(X_t - X_{t-1}) & t > 0 \\
        0.5 & t = 0\end{cases}$$

        We will add trick of using Mapping variable to map the difference between
        price_t and price_t-1
        """
        return 0.5 * (1 + self.alpha2 * (handy_map[state.is_prev_mv_up]))

    def next_state(self, state: State) -> State:
        k = self.generate_keys(random.randint(1, 1010), 3)
        up_move: IntLike = jax.random.binomial(k, 1, self.up_prob(state))
        return Process2.State(
                price=state.price + 2 * up_move - 1,
                is_prev_mv_up=bool(up_move))


@dataclass
class Process3:
    @dataclass
    class State:
        num_up_move: IntLike
        num_down_move: IntLike

    alpha3: FloatLike = 1.  # strength of reverse-pull (non-negative value)

    def generate_keys(self, seed: IntLike, n: IntLike) -> PRNGKey:
        """
        This function generates a set of random keys using JAX's deterministic
        random number generation system. This system ensures reproducible results
        across different runs by stabilizing numerical outputs.

        Parameters:
        - seed (int): The seed number used for randomization.

        Returns:
        - PRNGKey: An array of generated random keys.
        """
        rng = jax.random.PRNGKey(seed)
        key, *_ = jax.random.split(rng, n)
        return key

    def up_prob(self, state: State) -> FloatLike:
        total = state.num_up_move + state.num_down_move
        nd_mv = state.num_down_move
        x = total / jnp.where(nd_mv == 0, eps, nd_mv)
        return 1. / 1 + (1/x - 1) ** self.alpha3 if total else 0.5

    def next_state(self, state: State) -> State:
        k = self.generate_keys(random.randint(1, 1010), 3)
        up_move: IntLike = jax.random.binomial(k, 1, self.up_prob(state))
        return Process3.State(
                num_up_move=state.num_up_move + up_move,
                num_down_move=state.num_down_move + 1 - up_move)
