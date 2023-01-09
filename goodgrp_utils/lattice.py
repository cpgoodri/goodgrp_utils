import jax.numpy as jnp
from jax.api import jit, vmap


"""
TODO:
  - add lattices
    - kagome
    - twisted kagome
  - pass kwargs to get_lattice_details so lattices can be parameterized (e.g. twisted kagome)

"""


@jit
def tile_lattice_general(index_list, unit_cell_particles, lattice_vectors):
    """ Very general function that takes a set of particles in a unit cell, and
        tiles them into new cells defined by integer combinations of lattice
        vectors.

        For every particle p in the reference unit cell and for every set of
        indices {k_0, k_1, ..., k_{d-1}} in index_list, there will be a returned
        partucle with position p + \sum_i k_i * a_i, where a_0, a_1, ..., a_{d-1}
        are the lattice vectors.

  Args:
  index_list: array of shape (N_copies, d) that gives the lattice indices,
    where N_copies is the number of copies of the unit cell to be made
  unit_cell_particles: array of shape (N_ref_cell, d) giving the N_ref_cell
    particles in the reference unit cell
  lattice_vectors: array of shape (d,d) giving the lattice vectors.
    Lattice vectors do not have to be orthogonal or normalized.

  Return: array of shape (N_ref_cell * N_copies, d) of particle positions
  """

    # check dimension consistency
    N_copies, dim = index_list.shape
    N_ref_cell = unit_cell_particles.shape[0]
    N = N_ref_cell * N_copies

    def get_cell(indices):
        return unit_cell_particles + jnp.matmul(
            lattice_vectors.T, indices
        )  # p + i*a0 + j*a1 + k*a2

    return vmap(get_cell)(index_list).reshape((N, dim))


def get_lattice_details(lattice_type, offset=0.0, lattice_vectors=None):
    """ Function to generate particles within a reference unit cell, and optionally
        lattice vectors, for common lattices.
  Args:
  lattice_type: string indicating the lattice type. Much match one of the
    predefined options.
  offset: scalar or array of length (dim,) indicating a uniform translation,
    or offset, to apply to every particle. This can be useful if you want to
    move particles away from a boundary, for example.
  lattice_vectors: array of shape (dim,dim) indicating the lattice vectors.
    When not provided, default lattice vectors are used (this should be the
    most frequent use case by far).

  lattices include:
    3d: fcc, bcc, cubic
    2d: square, triangular (tri), honeycomb (honey)

  Return: tuple containing the lattice vectors and unit cell particles. When
    lattice_vectors is passed as an input, the returned lattice vectors are
    unchanged.
  """

    if lattice_type == "fcc":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0, 0.0])
            a1 = jnp.array([0.0, 1.0, 0.0])
            a2 = jnp.array([0.0, 0.0, 1.0])
            lattice_vectors = jnp.array([a0, a1, a2])
        else:
            a0, a1, a2 = lattice_vectors
        unit_cell_particles = (
            jnp.array(
                [[0.0, 0.0, 0.0], 0.5 * (a1 + a2), 0.5 * (a0 + a2), 0.5 * (a0 + a1)]
            )
            + offset
        )

    elif lattice_type == "bcc":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0, 0.0])
            a1 = jnp.array([0.0, 1.0, 0.0])
            a2 = jnp.array([0.0, 0.0, 1.0])
            lattice_vectors = jnp.array([a0, a1, a2])
        else:
            a0, a1, a2 = lattice_vectors
        unit_cell_particles = (
            jnp.array([[0.0, 0.0, 0.0], 0.5 * (a0 + a1 + a2)]) + offset
        )

    elif lattice_type == "cubic":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0, 0.0])
            a1 = jnp.array([0.0, 1.0, 0.0])
            a2 = jnp.array([0.0, 0.0, 1.0])
            lattice_vectors = jnp.array([a0, a1, a2])
        else:
            a0, a1, a2 = lattice_vectors
        unit_cell_particles = jnp.array([offset])

    if lattice_type == "tri":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0])
            a1 = jnp.array([0.5, jnp.sqrt(3.0) / 2.0])
            lattice_vectors = jnp.array([a0, a1])
        else:
            a0, a1 = lattice_vectors
        unit_cell_particles = jnp.array([[0.0, 0.0]]) + offset

    if lattice_type == "honey":
        if lattice_vectors is None:
            a0 = jnp.array([jnp.sqrt(3.0), 0.0])
            a1 = jnp.array([jnp.sqrt(3.0) / 2.0, 3.0 / 2.0])
            lattice_vectors = jnp.array([a0, a1])
        else:
            a0, a1 = lattice_vectors
        unit_cell_particles = jnp.array([[0.0, 0.0], (a0 + a1) / 3.0]) + offset

    elif lattice_type == "square":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0])
            a1 = jnp.array([0.0, 1.0])
            lattice_vectors = jnp.array([a0, a1])
        else:
            a0, a1 = lattice_vectors
        unit_cell_particles = jnp.array([[0.0, 0.0]]) + offset

    return lattice_vectors, unit_cell_particles


def make_lattice(lattice_type, ranges, return_box=False, **kwargs):
    """General API function for generating lattices.

  Args:
  lattice_type: string indicating the lattice type. Much match one of the
    predefined options.
  ranges: tuple or list of length d to indicate how the unit cell should be tiled.
    Each element of the tuple/list should either be a scalar or array. If an
    element is a scalar r, it is replaced with jnp.arange(r). The resulting d
    arrays are used to define the tiled unit cells using jnp.meshgrid.

    Example 1: ranges=[10,10]
      This results in a 10 by 10 tiling of the unit cell
    Example 2: ranges=[jnp.arange(-5,5),10]
      This also results in a 10 by 10 tiling, but the tiling in the x axis does
      not start at the origin.

  return_box: Boolean. If Ture, return a (d,d) array giving the box that fits
    the tiling. NOTE: this does not work properly when the elements of ranges
    do not start at 0.

  Return: array of particle positions, and optionally a (d,d) box matrix

  """

    def convert(r):
        if jnp.isscalar(r):
            r = jnp.arange(r)
        else:
            r = jnp.array(r)
        return r

    ranges = tuple([convert(r) for r in ranges])
    index_list = jnp.array(jnp.meshgrid(*ranges)).T.reshape(-1, len(ranges))
    lattice_vectors, unit_cell_particles = get_lattice_details(lattice_type, **kwargs)
    lattice = tile_lattice_general(index_list, unit_cell_particles, lattice_vectors)
    if return_box is False:
        return lattice
    else:
        box = vmap(lambda a, r: a * (jnp.amax(r) + 1), in_axes=(0, 0))(
            lattice_vectors, jnp.array(ranges)
        ).T
        return lattice, box


"""
TODO:
#  - rename things
#  - have option to return the box
#  - add more lattices
#  - better error handling
"""


@jit
def tile_lattice_3d(
    index_list,  # (i, j, k)'s indexing the unit cells to fill
    unit_cell_particles,
    a0=jnp.array([1.0, 0.0, 0.0]),
    a1=jnp.array([0.0, 1.0, 0.0]),
    a2=jnp.array([0.0, 0.0, 1.0]),
):
    """ Very general function that takes a set of particles in a unit cell, and
        tiles them into new cells defined by integer combinations of lattice
        vectors. For every particle p in the reference unit cell and for every
        triple (i,j,k) found in index_list, there will be a returned particle
        with position p + i*a0 + j*a1 + k*a2, where a0, a1, and a2 are the
        lattice vectors.

  Args:
    index_list: array of shape (N_copies, 3) that gives the (i,j,k) triples,
      where N_copies is the number of copies of the unit cell
    unit_cell_particles: array of shape (N_ref_cell, 3) giving the N_ref_cell
      particles in the reference unit cell
    a0: array of shape (3,) giving the first lattice vector
    a1: array of shape (3,) giving the second lattice vector
    a2: array of shape (3,) giving the third lattice vector

  Return: array of shape (N_ref_cell * N_copies, 3) of particle positions

  """
    N = unit_cell_particles.shape[0] * index_list.shape[0]

    def get_cell(i, j, k):
        return unit_cell_particles + i * a0 + j * a1 + k * a2

    return jnp.reshape(
        vmap(get_cell, in_axes=(0, 0, 0))(
            index_list[:, 0], index_list[:, 1], index_list[:, 2]
        ),
        (N, 3),
    )


def make_lattice_from_index_list_3d(
    lattice_type, index_list, offset=jnp.array([0.0, 0.0, 0.0]), lattice_vectors=None
):
    """ Use predefined lattice vectors and unit cells for common lattice types

  Args:
    lattice_type: string indicating the lattice type to use
    index_list: see doc for tile_lattice_3d()
    offset: scalar or array of shape (3,) to be added to every particle in the
      system. This can be useful to avoid having particle sit exactly on the
      edge of a simulation box
    lattice_vectors: array of shape (3, 3) used to override the default lattice
      vectors

    Returns: same as tile_lattice_3d
  """
    if lattice_type == "fcc":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0, 0.0])
            a1 = jnp.array([0.0, 1.0, 0.0])
            a2 = jnp.array([0.0, 0.0, 1.0])
        else:
            a0, a1, a2 = lattice_vectors
        unit_cell_particles = (
            jnp.array(
                [[0.0, 0.0, 0.0], 0.5 * (a1 + a2), 0.5 * (a0 + a2), 0.5 * (a0 + a1)]
            )
            + offset
        )
    elif lattice_type == "bcc":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0, 0.0])
            a1 = jnp.array([0.0, 1.0, 0.0])
            a2 = jnp.array([0.0, 0.0, 1.0])
        else:
            a0, a1, a2 = lattice_vectors
        unit_cell_particles = (
            jnp.array([[0.0, 0.0, 0.0], 0.5 * (a0 + a1 + a2)]) + offset
        )
    elif lattice_type == "square":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0, 0.0])
            a1 = jnp.array([0.0, 1.0, 0.0])
            a2 = jnp.array([0.0, 0.0, 1.0])
        else:
            a0, a1, a2 = lattice_vectors
        unit_cell_particles = jnp.array([offset])
    else:
        assert False
    return tile_lattice_3d(index_list, unit_cell_particles, a0, a1, a2)


def make_lattice_3d(lattice_type, irange, jrange, krange, **kwargs):
    """Convenience function to make a grid of the (i,j,k) triples
  """
    index_list = jnp.array(jnp.meshgrid(irange, jrange, krange)).T.reshape(-1, 3)
    return make_lattice_from_index_list_3d(lattice_type, index_list, **kwargs)


def tile_lattice_2d(
    index_list,  # (i, j)'s indexing the unit cells to fill
    unit_cell_particles,
    a0=jnp.array([1.0, 0.0]),
    a1=jnp.array([0.0, 1.0]),
):
    N = unit_cell_particles.shape[0] * index_list.shape[0]

    def get_cell(i, j):
        return unit_cell_particles + i * a0 + j * a1

    return jnp.reshape(
        vmap(get_cell, in_axes=(0, 0))(index_list[:, 0], index_list[:, 1]), (N, 2)
    )


def make_lattice_from_index_list_2d(
    lattice_type, index_list, offset=jnp.array([0.0, 0.0]), lattice_vectors=None
):

    if lattice_type == "tri":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0])
            a1 = jnp.array([0.5, jnp.sqrt(3.0) / 2.0])
        else:
            a0, a1 = lattice_vectors
        unit_cell_particles = jnp.array([[0.0, 0.0]]) + offset

    elif lattice_type == "square":
        if lattice_vectors is None:
            a0 = jnp.array([1.0, 0.0])
            a1 = jnp.array([0.0, 1.0])
        else:
            a0, a1 = lattice_vectors
        unit_cell_particles = jnp.array([[0.0, 0.0]]) + offset
    else:
        assert False
    return tile_lattice_2d(index_list, unit_cell_particles, a0, a1)


def make_lattice_2d(lattice_type, irange, jrange, **kwargs):
    index_list = jnp.array(jnp.meshgrid(irange, jrange)).T.reshape(-1, 2)
    return make_lattice_from_index_list_2d(lattice_type, index_list, **kwargs)


def make_lattice(lattice_type, irange, jrange, krange=None, **kwargs):
    """ General API function """
    if jnp.isscalar(irange):
        irange = jnp.arange(irange)
    if jnp.isscalar(jrange):
        jrange = jnp.arange(jrange)
    if jnp.isscalar(krange):
        krange = jnp.arange(krange)

    if krange is None:
        return make_lattice_2d(lattice_type, irange, jrange, **kwargs)
    else:
        return make_lattice_3d(lattice_type, irange, jrange, krange, **kwargs)
