import jax.numpy as jnp

from jax import jit, lax

from jax_md import minimize

f32 = jnp.float32
f64 = jnp.float64


# This version is fully differentiable and internally jitted
def run_minimization_scan(energy_fn, R_init, shift, num_steps=5000, **kwargs):
    init, apply = minimize.fire_descent(jit(energy_fn), shift, **kwargs)
    apply = jit(apply)

    @jit
    def scan_fn(state, i):
        return apply(state), 0.0

    state = init(R_init)
    state, _ = lax.scan(scan_fn, state, jnp.arange(num_steps))

    return state.position, jnp.amax(jnp.abs(state.force))


# This version is internally jitted but forward mode differentiable only
# The benefit is that it terminates when properly minimized
# Always use this version unless you need backward AD
def run_minimization_while(
    energy_fn, R_init, shift, max_grad_thresh=1e-12, max_num_steps=1000000, **kwargs
):
    init, apply = minimize.fire_descent(jit(energy_fn), shift, **kwargs)
    apply = jit(apply)

    @jit
    def get_maxgrad(state):
        return jnp.amax(jnp.abs(state.force))

    @jit
    def cond_fn(val):
        state, i = val
        return jnp.logical_and(get_maxgrad(state) > max_grad_thresh, i < max_num_steps)

    @jit
    def body_fn(val):
        state, i = val
        return apply(state), i + 1

    state = init(R_init)
    state, num_iterations = lax.while_loop(cond_fn, body_fn, (state, 0))

    return state.position, get_maxgrad(state), num_iterations


# This version is internally jitted, differentiable and uses neighbor lists.
# The benefit is that it terminates when properly minimized.
def run_minimization_while_neighbor_list(
    energy_fn,
    neighbor_fn,
    R_init,
    shift,
    max_grad_thresh=1e-12,
    max_num_steps=1000000,
    step_inc=1000,
    verbose=False,
    **kwargs
):
    nbrs = neighbor_fn.allocate(R_init)

    init, apply = minimize.fire_descent(jit(energy_fn), shift, **kwargs)
    apply = jit(apply)

    @jit
    def get_maxgrad(state):
        return jnp.amax(jnp.abs(state.force))

    @jit
    def body_fn(state_nbrs, t):
        state, nbrs = state_nbrs
        nbrs = neighbor_fn.update(state.position, nbrs)
        state = apply(state, neighbor=nbrs)
        return (state, nbrs), 0

    state = init(R_init, neighbor=nbrs)

    step = 0
    while step < max_num_steps:
        if verbose:
            print("minimization step {}".format(step))
        rtn_state, _ = lax.scan(body_fn, (state, nbrs), step + jnp.arange(step_inc))
        new_state, nbrs = rtn_state
        # If the neighbor list overflowed, rebuild it and repeat part of
        # the simulation.
        if nbrs.did_buffer_overflow:
            print("Buffer overflow.")
            nbrs = neighbor_fn.allocate(state.position)
        else:
            state = new_state
            step += step_inc
            if get_maxgrad(state) <= max_grad_thresh:
                break

    if verbose:
        print("successfully finished {} steps.".format(step * step_inc))

    return state.position, get_maxgrad(state), nbrs, step
