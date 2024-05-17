import chex
import numpy as np
from typing import Union, Mapping, TypeVar

import jax
import jax.numpy as jnp

from mk_process import FiniteMarkovRewardProcess, NonTerminal
from .utils import converged


Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]

S = TypeVar('S')
# A representation of a value fn for a finite MDP with state of type `S`
V = Mapping[NonTerminal[S], FloatLike]
X = TypeVar('X')

DEFAULT_TOLERANCE = 1e-5


def evaluate_mrp(
        mrp: FiniteMarkovRewardProcess[S],
        gamma: FloatLike
) -> Array:

    def update(states, n):
        gamma, v, rwd_fn_vec, mrp_trans_mat = states
        v_pi = rwd_fn_vec + gamma * mrp_trans_mat.dot(v)
        return (gamma, v_pi, rwd_fn_vec, mrp_trans_mat), v_pi

    v_0: jnp.ndarray = jnp.zeros(len(mrp.non_terminal_states))
    n_iter = jnp.arange(len(mrp.non_terminal_states))

    *_, v_pi = jax.lax.scan(update, (gamma, v_0, mrp.reward_function_vec,
                                     mrp.get_transtion_matrix()), n_iter)
    return v_pi


def almost_equal_arrays(
        v1: Array,
        v2: Array,
        tolerance: FloatLike = DEFAULT_TOLERANCE
) -> bool:
    return max(abs(v1 - v2)) < tolerance


def evaluate_mrp_result(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: FloatLike
) -> V[S]:
    v_ast: jnp.ndarray = converged(evaluate_mrp(mrp=mrp, gamma=gamma),
                                   done=almost_equal_arrays)  # TODO: converged fn
    return {s: v_ast[i].item() for i, s in enumerate(mrp.non_terminal_states)}
