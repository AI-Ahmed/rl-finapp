import chex
import numpy as np
from typing import Callable, TypeVar, Optional, Union, Iterator


Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]

S = TypeVar('S')
X = TypeVar('X')


def converge(values: Array, done: Callable[[X, X], bool]):
    '''Read from an iterator until two consecutive values satisfy the
       given done function or the input iterator ends.

       Raises an error if the input iterator is empty.

       Will loop forever if the input iterator doesn't end *or* converge.
    '''
    a = values[0]
    for i in range(values.shape[0] - 1):
        b = values[i+1]
        yield b
        if done(a, b):
            return
        a = b


def last(values: Iterator[Array]) -> Optional[X]:
    '''Return the last value of the given iterator.

    Returns None if the iterator is empty.

    If the iterator does not end, this function will loop forever.
    '''
    try:
        *_, last_element = values
        return last_element
    except ValueError:
        return None


def converged(values: Array,
              done: Callable[[X, X], bool]) -> X:
    '''Return the final value of the given iterator when its values
    converge according to the done function.

    Raises an error if the iterator is empty.

    Will loop forever if the input iterator doesn't end *or* converge.
    '''
    result = last(converge(values, done))

    if result is None:
        raise ValueError("converged called on an empty iterator")

    return result
