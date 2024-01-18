import jax
import jax.numpy as jnp

from dataclasses import dataclass


@dataclass
class Process1:
    @dataclass
    class State:
        price: int

    level_param: int  # level to which price mean-reverts
    alpha1: float  # strength of mean-reversion (non-negative value)
    k: jnp.ndarray = jax.random.PRNGKey(42)

    def up_prob(self, state: State) -> float:
        return 1. / (1 + jnp.exp(-self.alpha1 * (self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        up_move: int = jax.random.binomial(self.k, 1, self.up_prob(state), dtype=jnp.float32)
        return Process1.State(price=state.price + up_move * 2 - 1)
