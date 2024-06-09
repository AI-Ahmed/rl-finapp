import chex

import jax
import jax.numpy as jnp

import numpy as np

from typing import Callable, TypeVar, Optional, Union, Iterator


Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]

S = TypeVar('S')
X = TypeVar('X')


def iterate(step: Callable[[X], X], start: X) -> Iterator[X]:
    '''Find the fixed point of a function f by applying it to its own
    result, yielding each intermediate value.

    That is, for a function f, iterate(f, x) will give us a generator
    producing:

    x, f(x), f(f(x)), f(f(f(x)))...

    '''
    state = start

    while True:
        yield state
        state = step(state)


def iterate_jax(step: Callable[[X], X], start: X):
    """
    Find the fixed point of a function f by applying iit to its own
    result, yielding each itermediate value.

    that is, for a function f, iterate(f, x) will give us a generator
    producing:

    x, f(x), f(f(x)), f(f(f(x))), ...
    """
    def body_fn(carry):
        i, conv, state, step = carry
        return i + 1, conv, step(state), state

    def cond_fn(carry):
        i, conv, _, _ = carry
        return jnp.logical_not(conv)

    state = start

    init_val = (1, False, step, state)
    _, _, _, state = jax.lax.while_loop(cond_fn, body_fn, init_val)
    return state


def converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    '''Read from an iterator until two consecutive values satisfy the
    given done function or the input iterator ends.

    Raises an error if the input iterator is empty.

    Will loop forever if the input iterator doesn't end *or* converge.

    '''
    a = next(values, None)
    if a is None:
        return

    yield a

    for b in values:
        yield b
        if done(a, b):
            return

        a = b


def converge_jax(values: Array, done: Callable[[X, X], bool]) -> Array:
    def body_fn(states):
        i, converged, prev_val, values = states
        curr_val = values[i]
        converged = done(prev_val, curr_val)
        return i + 1, converged, curr_val, values

    def cond_fn(states):
        i, converged, _, _ = states
        return jnp.logical_and(i < values.shape[0], jnp.logical_not(converged))

    init_val = (1, False, values[0], values)  # Start with the second element
    _, _, _, values = jax.lax.while_loop(cond_fn, body_fn, init_val)
    return values


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
    if isinstance(values, Array):
        result = last(converge_jax(values, done))
    else:
        result = last(converge(values, done))

    if result is None:
        raise ValueError("converged called on an empty iterator")

    return result
