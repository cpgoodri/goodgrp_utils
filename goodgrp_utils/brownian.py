import jax.numpy as jnp

from jax import random
from jax import jit, lax

from jax_md import simulate

f32 = jnp.float32
f64 = jnp.float64


# Run a brownian dynamics simulation and save periodic snapshots
def run_brownian(
    energy_fn,
    R_init,
    shift,
    key,
    num_total_steps,
    record_every,
    dt,
    measure_fn=lambda R: R,
    **kwargs
):
    # define the simulation
    init, apply = simulate.brownian(energy_fn, shift, dt, **kwargs)
    apply = jit(apply)

    @jit
    def apply_single_step(state, t):
        return apply(state, t=t), 0

    @jit
    def apply_many_steps(state, t_list):
        state, _ = lax.scan(apply_single_step, state, t_list)
        return state, measure_fn(state.position)

    # initialize the system
    key, split = random.split(key)
    initial_state = init(split, R_init)

    # run the simulation
    final_state, data = lax.scan(
        apply_many_steps,
        initial_state,
        jnp.arange(num_total_steps).reshape(
            num_total_steps // record_every, record_every
        ),
    )

    # return the trajectory
    return final_state, data


# DIFFERENTIABLE and uses neighbor_lists
def run_brownian_neighbor_list(
    energy_fn,
    neighbor_fn,
    R_init,
    shift,
    key,
    num_steps,
    step_inc=1000,
    dt=0.0001,
    kT=1.0,
    gamma=1.0,
    verbose=False,
    measure_fn=lambda R, nbrs: R,
    **static_kwargs
):
    nbrs = neighbor_fn.allocate(R_init)

    init, apply = simulate.brownian(energy_fn, shift, dt=dt, kT=kT, gamma=gamma)

    def body_fn(state_nbrs, t):
        state, nbrs = state_nbrs
        nbrs = neighbor_fn.update(state.position, nbrs)
        state = apply(state, neighbor=nbrs, **static_kwargs)
        return (state, nbrs), 0

    key, split = random.split(key)
    state = init(split, R_init)

    step = 0
    data = []
    while step < num_steps:
        if verbose:
            print("simulation step {}".format(step))
        rtn_state_nbrs, _ = lax.scan(
            body_fn, (state, nbrs), step + jnp.arange(step_inc)
        )
        new_state, nbrs = rtn_state_nbrs
        # If the neighbor list overflowed, rebuild it and repeat part of
        # the simulation.
        if nbrs.did_buffer_overflow:
            print("Buffer overflow.")
            nbrs = neighbor_fn.allocate(state.position)
        else:
            state = new_state
            step += step_inc
            data += [measure_fn(state.position, nbrs=nbrs, **static_kwargs)]

    if verbose:
        print("successfully finished {} steps.".format(step))

    return state, jnp.array(data), nbrs
