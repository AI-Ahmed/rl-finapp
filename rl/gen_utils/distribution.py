from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
# from dataclasses import dataclass

# import chex
from chex import dataclass
from distrax._src.utils import jittable
import jax
import jax.numpy as jnp
import numpy as np

from typing import (Union, Callable, Dict, Generic, Iterator, Iterable,
                    Mapping, Optional, Sequence, Tuple, TypeVar)

IntLike = Union[int, np.int16, np.int32, np.int64]
FloatLike = Union[int, np.float16, np.float32, np.float64]

A = TypeVar('A', bound=IntLike)
B = TypeVar('B')


class Distribution(ABC, Generic[A]):
    '''A probability distribution that we can sample.

    '''
    @abstractmethod
    def sample(self) -> A:
        '''Return a random sample from this distribution.

        '''
        pass

    def sample_n(self, n: IntLike) -> Sequence[A]:
        '''Return n samples from this distribution.'''
        n_iters = jnp.arange(n)
        _, samples = jax.lax.scan(lambda state, i: self.sample(), (), n_iters)
        return samples
        # return [self.sample() for _ in range(n)]

    @abstractmethod
    def expectation(
        self,
        f: Callable[[A], FloatLike]
    ) -> FloatLike:
        '''Return the expecation of f(X) where X is the
        random variable for the distribution and f is an
        arbitrary function from X to float

        '''
        pass

    def map(
        self,
        f: Callable[[A], B]
    ) -> Distribution[B]:
        '''Apply a function to the outcomes of this distribution.'''
        return SampledDistribution(lambda: f(self.sample()))

    def apply(
        self,
        f: Callable[[A], Distribution[B]]
    ) -> Distribution[B]:
        '''Apply a function that returns a distribution to the outcomes of
        this distribution. This lets us express *dependent random
        variables*.

        '''
        def sample():
            a = self.sample()
            b_dist = f(a)
            return b_dist.sample()

        return SampledDistribution(sample)


class SampledDistribution(jittable.Jittable, Distribution[A]):
    '''A distribution defined by a function to sample it.

    '''
    sampler: Callable[[], A]
    expectation_samples: IntLike

    def __init__(
        self,
        sampler: Callable[[], A],
        expectation_samples: IntLike = 10000
    ):
        self.sampler = sampler
        self.expectation_samples = expectation_samples

    def sample(self) -> A:
        return self.sampler()

    def expectation(
        self,
        f: Callable[[A], FloatLike]
    ) -> FloatLike:
        '''Return a sampled approximation of the expectation of f(X) for some f.

        '''
        e_iters = jnp.arange(self.expectation_samples)
        _, samples = jax.lax.scan(lambda state, i: f(self.sample()), (), e_iters)
        return jnp.sum(samples) / self.expectation_samples
#        return sum(f(self.sample()) for _ in
#                  range(self.expectation_samples)) / self.expectation_samples


class FiniteDistribution(Distribution[A], ABC):
    '''A probability distribution with a finite number of outcomes, which
    means we can render it as a PDF or CDF table.

    '''
    @abstractmethod
    def table(self) -> Mapping[A, FloatLike]:
        '''Returns a tabular representation of the probability density
        function (PDF) for this distribution.

        '''
        pass

    def probability(self, outcome: A) -> FloatLike:
        '''Returns the probability of the given outcome according to this
        distribution.

        '''
        return self.table()[outcome]

    def map(self, f: Callable[[A], B]) -> FiniteDistribution[B]:
        '''Return a new distribution that is the result of applying a function
        to each element of this distribution.

        '''
        result: Dict[B, FloatLike] = defaultdict(FloatLike)

        for x, p in self:
            result[f(x)] += p

        return Categorical(result)

    def sample(self, seed) -> A:
        outcomes = list(self.table().keys())
        weights = list(self.table().values())
        return random.choices(outcomes, weights=weights)[0]

    # TODO: Can we get rid of f or make it optional? Right now, I
    # don't think that's possible with mypy.
    def expectation(self, f: Callable[[A], FloatLike]) -> FloatLike:
        '''Calculate the expected value of the distribution, using the given
        function to turn the outcomes into numbers.

        '''
        return sum(p * f(x) for x, p in self)

    def __iter__(self) -> Iterator[Tuple[A, FloatLike]]:
        return iter(self.table().items())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FiniteDistribution):
            return self.table() == other.table()
        else:
            return False

    def __repr__(self) -> str:
        return repr(self.table())


class Choose(jittable.Jittable, FiniteDistribution[A]):
    '''Select an element of the given list uniformly at random.
    '''

    options: Sequence[A]
    _table: Optional[Mapping[A, FloatLike]] = None

    def __init__(self, options: Iterable[A]):
        self.options = list(options)

    def sample(self, seed) -> A:
        key = jax.random.PRNGKey(seed)
        return jax.random.choice(key, jnp.array(self.options, dtype=IntLike))
        # return random.choice(self.options)

    def table(self) -> Mapping[A, FloatLike]:
        if self._table is None:
            counter = Counter(self.options)
            length = len(self.options)
            self._table = {x: counter[x] / length for x in counter}

        return self._table

    def probability(self, outcome: A) -> FloatLike:
        return self.table().get(outcome, 0.0)


class Categorical(FiniteDistribution[A]):
    '''Select from a finite set of outcomes with the specified
    probabilities.

    '''

    probabilities: Mapping[A, FloatLike]

    def __init__(self, distribution: Mapping[A, FloatLike]):
        total = sum(distribution.values())
        # Normalize probabilities to sum to 1
        self.probabilities = {outcome: probability / total
                              for outcome, probability in distribution.items()}

    def table(self) -> Mapping[A, FloatLike]:
        return self.probabilities

    def probability(self, outcome: A) -> FloatLike:
        return self.probabilities.get(outcome, 0.)


@dataclass(frozen=True)
class Constant(FiniteDistribution[A]):
    '''A distribution that has a single outcome with probability 1.

    '''
    value: A

    def sample(self) -> A:
        return self.value

    def table(self) -> Mapping[A, FloatLike]:
        return {self.value: 1}

    def probability(self, outcome: A) -> FloatLike:
        return 1. if outcome == self.value else 0.
