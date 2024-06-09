import chex
import numpy as np
from loguru import logger

from typing import Union, Mapping, TypeVar, Tuple, Dict

import jax
import jax.numpy as jnp

from mk_process import FiniteMarkovRewardProcess, NonTerminal, State
from mk_d_process import FiniteMarkovDecisionProcess
from policy import FiniteDeterministicPolicy, FinitePolicy
from .utils import converged, iterate
from gen_utils.distribution import Choose


Array = Union[chex.Array, chex.ArrayNumpy]
FloatLike = Union[float, np.float16, np.float32, np.float64]
IntLike = Union[int, np.int16, np.int32, np.float64]

A = TypeVar('A')
S = TypeVar('S')
# A representation of a value fn for a finite MDP with state of type `S`
V = Mapping[NonTerminal[S], FloatLike]
X = TypeVar('X')

DEFAULT_TOLERANCE = 1e-5


def evaluate_mrp(
        mrp: FiniteMarkovRewardProcess[S],
        gamma: FloatLike
) -> Array:

    def update(states, _: None):
        gamma, v, rwd_fn_vec, mrp_trans_mat = states
        v_pi = rwd_fn_vec + gamma * mrp_trans_mat.dot(v)
        return (gamma, v_pi, rwd_fn_vec, mrp_trans_mat), v_pi

    v_0: Array = jnp.zeros(len(mrp.non_terminal_states))
    *_, v_pi = jax.lax.scan(update, (gamma, v_0, mrp.reward_function_vec,
                                     mrp.get_transition_matrix()), None,
                            len(mrp.non_terminal_states))
    return v_pi


def almost_equal_arrays(
        v1: Array,
        v2: Array,
        tolerance: FloatLike = DEFAULT_TOLERANCE
) -> bool:
    return jnp.max(jnp.abs(v1 - v2)) < tolerance


def evaluate_mrp_result(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: FloatLike
) -> V[S]:
    v_ast: Array = converged(evaluate_mrp(mrp=mrp, gamma=gamma),
                             done=almost_equal_arrays)
    return {s: v_ast[i] for i, s in enumerate(mrp.non_terminal_states)}


def extended_vf(v: V[S], s: State[S]) -> FloatLike:
    def non_terminal_vf(st: NonTerminal[S], v=v) -> FloatLike:
        return v[st]
    return s.on_non_terminal(non_terminal_vf, 0.0)


def greedy_policy_from_vf(
        mdp: FiniteMarkovDecisionProcess[S, A],
        vf: V[S],
        gamma: FloatLike
) -> Tuple[FinitePolicy[S, A], FiniteDeterministicPolicy[S, A]]:

    actions = mdp.actions
    mapping = mdp.mapping

    def calculate_q_values(state):
        if state not in mapping:
            logger.warning(f"State {state} not found in mapping")
            return jnp.array([])  # Handle missing state gracefully
        state_actions = actions(state)
        action_values = jnp.array([
            mapping[state][a].expectation(
                lambda s_r: s_r[1] + gamma * extended_vf(vf, s_r[0])
            )
            for a in state_actions
        ])
        return action_values

    non_terminal_states = mdp.non_terminal_states
    # print("States: ", non_terminal_states, "length: ", len(non_terminal_states))

    greedy_policy_dict = {}

    # JAX-compatible calculation of Q-values
    # We assume non_terminal_states is a list of NonTerminal objects
    state_action_values = []
    for state in non_terminal_states:
        q_values = calculate_q_values(state)
        state_action_values.append(q_values)

    for state, action_values in zip(non_terminal_states, state_action_values):
        if action_values.size == 0:
            logger.error(f"No action values for state {state}")
            continue  # Skip states with no action values
        best_action_idx = jnp.argmax(action_values)
        best_action = jnp.array(actions(state))[best_action_idx]
        greedy_policy_dict[state.state] = best_action

    return FiniteDeterministicPolicy(action_for=greedy_policy_dict)


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: FloatLike,
    matrix_method_for_mrp_eval: bool = False
) -> Tuple[Array, Array]:

    def update(vf_policy: Tuple[V[S], FinitePolicy[S, A]]):
        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)

        def mat_eval():
            return {mrp.non_terminal_states[i]: v
                    for i, v in enumerate(mrp.get_value_function_vec(gamma))}

        def func_eval():
            return evaluate_mrp_result(mrp, gamma)

        imp_vf: V[S] = jax.lax.cond(matrix_method_for_mrp_eval, mat_eval, func_eval)
        imp_pi = greedy_policy_from_vf(
            mdp,
            imp_vf,
            gamma)
        return imp_vf, imp_pi

    pi_0: FinitePolicy[S, A] = FinitePolicy(policy_map={s.state: Choose(options=mdp.actions(s))
                                                        for s in mdp.non_terminal_states})
    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}  # for equivalent parameter size
    # scan here doesn't work due to the change in the structure of the policy
    # from `FinitePolicy` to `Deterministic Policy.
#    *_, (pi_vf, imp_pi) = jax.lax.scan(update, (gamma, pi_0), None,
#                                       len(mdp.non_terminal_states))
    return iterate(update, (v_0, pi_0))


def almost_eqal_vf_pis(
        x1: Tuple[V[S], FinitePolicy[S, A]],
        x2: Tuple[V[S], FinitePolicy[S, A]]
) -> bool:
    return max(abs(x1[0][s] - x2[0][s]) for s in x1[0]) < DEFAULT_TOLERANCE


def policy_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: FloatLike
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    return converged(policy_iteration(mdp, gamma), done=almost_eqal_vf_pis)


def value_iteration(
      mdp: FiniteMarkovDecisionProcess[S, A],
      gamma: float
  ) -> V[S]:

    actions = mdp.actions
    mapping = mdp.mapping

    def extended_vf(v: V[S], s: S) -> float:
        return v.get(s, jnp.array(0.0, dtype=jnp.float32))

    def yield_states_vf(value_dict: Mapping[NonTerminal, Array]):
        """
        Yield a dictionary of states with one scalar corresponding to each state from the input value dictionary.

        :param value_dict: Dictionary with NonTerminal states as keys and arrays of value function values as values.
        :return: Generator that yields dictionaries with NonTerminal states and single scalar values.
        """
        # Get the length of the arrays (assuming all arrays have the same length)
        array_length = len(next(iter(value_dict.values())))

        for i in range(array_length):
            yield {state: values[i] for state, values in value_dict.items()}

    def update(
            state: Tuple[Dict[S, Array], float],
            _: None
    ) -> Tuple[Tuple[Dict[S, Array], float], Dict[S, Array]]:
        v, gamma = state

        def get_updated_v(s: S) -> jnp.array:
            q_values = jnp.array([
                jnp.sum(jnp.array([  # we replaced the expectation fn from `FiniteDistribution`.
                    p_sr * (r + gamma * extended_vf(v, s_next))
                    for (s_next, r), p_sr in dist.table().items()
                ]))
                for a in actions(s)
                for dist in [mapping[s][a]]
            ])
            return jnp.max(q_values)

        updated_vf = {s: get_updated_v(s) for s in v}
        return (updated_vf, gamma), updated_vf

    v_0: V[S] = {s: jnp.array(0.0, dtype=jnp.float32) for s in mdp.non_terminal_states}
    _, st_vf = jax.lax.scan(update, (v_0, gamma), None, len(mdp.non_terminal_states))
    # import pdb; pdb.set_trace()
    return yield_states_vf(st_vf)


def almost_eqal_vfs(v1: V[S], v2: V[S], tol: FloatLike = DEFAULT_TOLERANCE) -> bool:
    return max(abs(v1[s] - v2[s]) for s in v1)


def value_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: FloatLike
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:

    vf_ast: V[S] = converged(value_iteration(mdp, gamma), almost_eqal_vfs)
    pi_ast: FiniteDeterministicPolicy[S, A] = greedy_policy_from_vf(mdp, vf_ast, gamma)
    return vf_ast, pi_ast
