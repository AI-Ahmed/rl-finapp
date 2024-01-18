from abc import ABC, abstractmethod
import jax

k = jax.random.PRNGKey(42)


class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass


class Die(Distribution):
    def __init__(self, sides):
        self.sides = sides

    def sample(self):
        return jax.random.randint(k, 1, self.sides)

    def __repr__(self):
        return f"Die(sides={self.sides})"

    def __eq__(self, other):
        if isinstance(other, Die):
            return self.sides == other.sides
