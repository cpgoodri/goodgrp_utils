"""Scan_in_dim function copied from flax (https://github.com/google/flax).
It let's you easily perform a scan over an arbitrary axis of a N dimensional array.
"""

from collections.abc import Iterable
import numpy as onp

import jax
from jax import lax, jit
import jax.numpy as jnp


def _scan_nd(body_fn, init, xs, n=1):
    """Utility for performing an n-dimensional `lax.scan`.
  The n-d scan is simply recursive call of 1-d scan.
  Args:
    body_fn: the body of the loop of type (c, x) -> (c, y).
    init: initial value for the carry.
    xs: a pytree of tensors to scan over.
    n: number of dimensions to scan over (default: 1)
  Returns:
    A tuple of the final carry and the values returned by the body.
  """
    if n == 1:
        return lax.scan(body_fn, init, xs)
    else:

        def scan_body(c, x):
            return _scan_nd(body_fn, c, x, n=n - 1)

        return lax.scan(scan_body, init, xs)


def _invert_perm(perm):
    perm_inv = [0] * len(perm)
    for i, j in enumerate(perm):
        perm_inv[j] = i
    return tuple(perm_inv)


def scan_in_dim(body_fn, init, xs, axis=(0,), keepdims=False):
    """utility for doing a scan along arbitrary dimensions.
  see `lax.scan` for details on how the scan operation works.
  Args:
    body_fn: the body of the loop of type (c, x) -> (c, y).
    init: initial value for the carry.
    xs: a pytree of tensors to scan over.
    axis: the axis to scan over.
    keepdims: keep the dimensions that are scanned over.
  Returns:
    A tuple of the final carry and the values returned by the body.
  """
    if not isinstance(axis, Iterable):
        axis = (axis,)

    def transpose_in(x):
        perm = axis + tuple(onp.delete(onp.arange(x.ndim), axis))
        return x.transpose(perm)

    def transpose_out(x):
        perm = axis + tuple(onp.delete(onp.arange(x.ndim), axis))
        return x.transpose(_invert_perm(perm))

    def body_wrapper(c, xs):
        if keepdims:
            xs = jax.tree_map(lambda x: x.reshape((1,) * len(axis) + x.shape), xs)
            xs = jax.tree_map(transpose_out, xs)
        c, ys = body_fn(c, xs)
        if keepdims:
            ys = jax.tree_map(transpose_in, ys)
            ys = jax.tree_map(lambda x: x.reshape(x.shape[len(axis) :]), ys)
        return c, ys

    xs = jax.tree_map(transpose_in, xs)
    c, ys = _scan_nd(body_wrapper, init, xs, n=len(axis))
    ys = jax.tree_map(transpose_out, ys)
    return c, ys


@jit
def _vector2symmat(v, zeros):
    n = zeros.shape[0]
    assert v.shape == (
        n * (n + 1) / 2,
    ), f"The input must have shape jnp.int16(((1 + 8 * v.shape[0]) ** 0.5 - 1) / 2) = {(n * (n + 1) / 2,)}, got {v.shape} instead."
    ind = jnp.triu_indices(n)
    return zeros.at[ind].set(v).at[(ind[1], ind[0])].set(v)


@jit
def vector2symmat(v):
    """ Convert a vector into a symmetric matrix.
  Args:
    v: vector of length (n*(n+1)/2,)
  Return:
    symmetric matrix m of shape (n,n) that satisfies
    m[jnp.triu_indices_from(m)] == v

  Example:
    v = jnp.array([0,1,2,3,4,5])
    returns: [[ 0, 1, 2],
                1, 3, 4],
                2, 4, 5]]
  """
    n = int(((1 + 8 * v.shape[0]) ** 0.5 - 1) / 2)
    return _vector2symmat(v, jnp.zeros((n, n), dtype=v.dtype))


@jit
def _vector2symmat_diag0(v, zeros):
    n = zeros.shape[0]
    assert v.shape == (
        n * (n - 1) / 2,
    ), f"The input must have shape jnp.int16(((1 + 8 * v.shape[0]) ** 0.5 + 1) / 2) = {(n * (n + 1) / 2,)}, got {v.shape} instead."
    ind = jnp.triu_indices(n, 1)
    return zeros.at[ind].set(v).at[(ind[1], ind[0])].set(v)


@jit
def vector2symmat_diag0(v):
    """ Convert a vector into a symmetric matrix with zeros on the diagonal.
  Args:
    v: vector of length (n*(n-1)/2,)
  Return:
    symmetric matrix m of shape (n,n) that satisfies
    m[jnp.triu_indices_from(m, 1)] == v

  Example:
    v = jnp.array([0,1,2])
    returns: [[ 0, 0, 1],
                0, 0, 2],
                1, 2, 0]]
  """
    n = int(((1 + 8 * v.shape[0]) ** 0.5 + 1) / 2)
    return _vector2symmat_diag0(v, jnp.zeros((n, n), dtype=v.dtype))


def get_species_from_distribution(N, species_distribution, key=None):
  """ Convert an array of probability distributions of M species into an 
      array with length N and matching distribution of the species.

  Args:
    N: total number of particles
    species_dist: array of shape (M,) indicating the desired distribution of 
      species, or an int M indicating an even distribution of M species
    key [Optional]: RNG key to draw random numbers to fill out the remainder
    of the species array when N is not a even multiple of M
  Return:
    1-D Array species that has length N and specified distribution from 
    species_dist

  Example:
    N = 10, species_dist = jnp.array([0.2,0.2,0.2,0.2,0.2])
    returns: [0,0,1,1,2,2,3,3,4,4]

  """
  if isinstance(species_distribution, int):
    _species_distribution = jnp.ones((species_distribution,), dtype=jnp.float32) / species_distribution
  else:
    _species_distribution = species_distribution

  species = jnp.zeros(shape=(N,), dtype=jnp.int32)
  
  _species_distribution = _species_distribution / jnp.sum(_species_distribution)
  species_dist_N = jnp.array(_species_distribution * N).astype(int)

  particle_index = 0
  species_index = 0

  for n_species_i in species_dist_N:
    #species = species.at[particle_index:particle_index + n_species_i].set(jnp.full((n_species_i,), species_index, dtype=jnp.int32))
    species = species.at[particle_index:particle_index + n_species_i].set(species_index * jnp.ones((n_species_i,), dtype=jnp.int32))
    species_index += 1
    particle_index += n_species_i

  p = _species_distribution * N - species_dist_N
  additions = jnp.zeros(N, dtype=jnp.int32)
  if key is None:
    additions = additions.at[N-p.shape[0]:].set(jnp.argsort(p))
  else:
    additions = additions.at[N-p.shape[0]:].set(random.choice(key, _species_distribution.shape[0], (p.shape[0],), True, p=p))
  return jnp.sort(jnp.where(jnp.arange(N)>=particle_index, additions, species))




