from abc import ABC, abstractmethod

import chex
from chex import dataclass

import distrax

# from dataclasses import dataclass
from typing import Generic, TypeVar, Sequence, Union

import jax
import jax.numpy as Array
import numpy as np

FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]
PRNGKey = chex.PRNGKey
# A type variable named "A"
A = TypeVar('A', bound=IntLike)


class Distribution(ABC, Generic[A]):
    @abstractmethod
    def sample(self) -> A:
        pass


@dataclass(frozen=True)  # for converting the class to immutable dataclass
class Die(Distribution[IntLike]):
    sides: IntLike

    def sample(self):
        k: PRNGKey = jax.random.PRNGKey(42)
        return jax.random.randint(k, 1, self.sides)


@dataclass
class Gaussian(Distribution[FloatLike]):
    mu: FloatLike
    sigma: FloatLike

    def sample(self) -> FloatLike:
        rng: PRNGKey = jax.random.PRNGKey(42)
        return self.mu + self.sigma * jax.random.normal(rng)

    def n_sample(self, n: IntLike) -> Sequence[FloatLike]:
        rng: PRNGKey = jax.random.PRNGKey(42)
        dist = distrax.Normal(0., 1.)
        return dist.sample(rng, sample_shape=n)

    def np_n_sample(self, n: IntLike) -> Sequence[FloatLike]:
        return np.random.normal(self.mu, self.sigma, n)

    def jax_n_sample(self, n: IntLike) -> Sequence[FloatLike]:
        return self.mu + self.sigma * jax.random.normal(self.k, shape=(n, ), dtype=Array.float32)


if __name__ == "__main__":
    import dataclasses

    d6 = Die(6)
    # d6.sides = 7  # This raise error
    d20 = dataclasses.replace(d6, sides=20)  # It is inefficient to do that, instead, it is better
    # to initialize a new instance

    d = {d6: 'abc'}
