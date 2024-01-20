import jax
import jax.numpy as jnp

from dataclasses import dataclass

k = jax.random.PRNGKey(42)


@dataclass(frozen=True)
class Process1:
    @dataclass
    class State:
        price: int

        @property
        def dtype(self):
            return jnp.dtype(object)

    level_param: int  # level to which price mean-reverts
    alpha1: float  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: State) -> float:
        return 1. / (1 + jnp.exp(-self.alpha1 * (self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        up_move: int = jax.random.binomial(k, 1, self.up_prob(state), dtype=jnp.float32)
        return Process1.State(price=state.price + up_move * 2 - 1)

    @property
    def dtype(self):
        return jnp.dtype(object)
