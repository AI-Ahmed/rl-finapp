from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Sequence

import jax
import jax.numpy as jnp
import numpy as np

# A type variable named "A"
A = TypeVar("A")


class Distribution(ABC, Generic[A]):
    @abstractmethod
    def sample(self) -> A:
        pass


@dataclass(frozen=True)  # for converting the class to immutable dataclass
class Die(Distribution[int]):
    sides: int
    k: jnp.ndarray = jax.random.PRNGKey(42)

    def sample(self):
        return jax.random.randint(self.k, 1, self.sides)


@dataclass
class Gaussian(Distribution[float]):
    mu: float
    sigma: float
    k: jnp.ndarray = jax.random.PRNGKey(42)

    def sample(self) -> float:
        return self.mu + self.sigma * jax.random.normal(self.k)

    def n_sample(self, n: int) -> Sequence[float]:
        return [self.sample() for _ in range(n)]

    def np_n_sample(self, n: int) -> Sequence[float]:
        return np.random.normal(self.mu, self.sigma, n)

    def jax_n_sample(self, n: int) -> Sequence[float]:
        return self.mu + self.sigma * jax.random.normal(self.k, shape=(n, ), dtype=jnp.float32)


if __name__ == "__main__":
    import dataclasses

    d6 = Die(6)
    # d6.sides = 7  # This raise error
    d20 = dataclasses.replace(d6, sides=20)  # It is inefficient to do that, instead, it is better
    # to initialize a new instance

    d = {d6: 'abc'}
