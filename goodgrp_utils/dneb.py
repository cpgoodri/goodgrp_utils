import numpy as onp
import jax.numpy as jnp
from jax import grad, jit, vmap, lax

from jax_md import minimize, space


"""
TODO
  - neighbor lists
  - implicit AD for minimizations
  - properly pass kwargs
  - error check minimizations
  - implement _affine_matrix_from_points using jax
"""



def interpolate_positions(R1, R2, num_images, displacement, shift):
  """ Create images that span between R1 and R2

  Args:
    R1: array of shape (N,d) 
    R2: array of shape (N,d) 
    num_images: int indicating number of total images to be returned
    displacement: jax_md displacement function
    shift: jax_md shift function

  Return: array of shape (num_images, N, d) 
  """

  #dr = R2 - R1
  system_displacement = vmap(displacement, in_axes=(0,0))
  dr = system_displacement(R2, R1)
  
  #linear interpolation
  #return jnp.array([R1 + i * dr for i in jnp.linspace(0, 1, num_images)])  
  return jnp.array([ shift(R1, i * dr) for i in jnp.linspace(0, 1, num_images)])


def setup_DNEB_force(true_energy_fn, spring_energy_fn, displacement):
  """ Defines Doubly Nudged Elastic Band calculations

  Args:
    true_energy_fn:  A function that takes an (N,d) array of positions and 
      returns an energy. 
    spring_energy_fn: A function that takes an (m,N,d) array of positions and 
      returns an energy (corresponding to the springs that couple the m systems)
    displacement: jax_md displacement function

  """

  multi_system_displacement = vmap(vmap(
    displacement, in_axes=(0,0)), in_axes=(0,0))

  true_energy_full_fn = vmap(true_energy_fn)
  true_gradient_full_fn = vmap(grad(true_energy_fn))
  spring_gradient_fn = grad(spring_energy_fn)

  true_energy_full_fn = jit(true_energy_full_fn)
  true_gradient_full_fn = jit(true_gradient_full_fn)
  spring_gradient_fn = jit(spring_gradient_fn)

  def calculate_tau_hat(R):
    energies = true_energy_full_fn(R)
    dE = energies[1:] - energies[:-1]  #E(i+1) - E(i) for all i
    #energy differences of neighbors i-1, i, and i+1
    dEr = dE[1:]  #E(i+1) - E(i)
    dEl = dE[:-1]  #E(i) - E(i-1)
    #maximum/minimum values
    dEmax = jnp.maximum(jnp.abs(dEr), jnp.abs(dEl))
    dEmin = jnp.minimum(jnp.abs(dEr), jnp.abs(dEl))

    #case separation: write down and normalize all 4 different tangent vectors (all have same length)
    #tau0 & tau1 are cases in which eq. (4) from DNEB paper can be applied
    #tau0 = R[2:] - R[1:-1]  #R(i+1)-R(i)
    tau0 = multi_system_displacement(R[1:-1], R[2:])
    #tau1 = R[1:-1] - R[:-2]  #R(i)-R(i-1)
    tau1 = multi_system_displacement(R[:-2], R[1:-1])
    #tau2 & tau3 are cases in which eq. (4) cannot be applied, i.e., image i is at a minimum/maximum
    #compute weighted average of the vectors to the two neighboring images, eqs. (10)+(11) from G. Henkelman and H. Jónsson, J. Chem. Phys. 113, 9978 (2000)
    tau2 = vmap(jnp.dot)(tau0, dEmax) + vmap(jnp.dot)(tau1, dEmin)
    tau3 = vmap(jnp.dot)(tau0, dEmin) + vmap(jnp.dot)(tau1, dEmax)
    #normalization
    tau0 = tau0 / jnp.linalg.norm(tau0, axis=(1, 2), keepdims=True)
    tau1 = tau1 / jnp.linalg.norm(tau1, axis=(1, 2), keepdims=True)
    tau2 = tau2 / jnp.linalg.norm(tau2, axis=(1, 2), keepdims=True)
    tau3 = tau3 / jnp.linalg.norm(tau3, axis=(1, 2), keepdims=True)

    #4 cases
    v0 = jnp.array(
        jnp.logical_and(dEl > 0, dEr > 0), dtype=jnp.int32)  #E(i+1)>E(i)>E(i-1)
    v1 = jnp.array(
        jnp.logical_and(dEl < 0, dEr < 0), dtype=jnp.int32)  #E(i+1)<E(i)<E(i-1)
    v2 = jnp.array(
        jnp.logical_and((v0 + v1) != 1, (dEr + dEl) > 0),
        dtype=jnp.int32)  #min/max and E(i+1)>E(i-1)
    v3 = jnp.array(
        jnp.logical_and((v0 + v1) != 1, (dEr + dEl) < 0),
        dtype=jnp.int32)  #min/max and E(i+1)<E(i-1)
    v = 0 * v0 + 1 * v1 + 2 * v2 + 3 * v3  #index array, tells for each element, which of the taus to choose

    choices = jnp.array([tau0, tau1, tau2, tau3])  #set of arrays to choose from
    tau = jnp.array(
        vmap(lambda choices, v: choices[v], in_axes=(1, 0))(choices, v))
    return tau

  #non-normalized projection of a onto b.
  # for this to be a proper projection, b should already be normalized
  def projection(a, b):
    return jnp.tensordot(a, b) * b

  vmap_projection = vmap(projection)

  def total_force(Rs):
    g = true_gradient_full_fn(Rs)[1:-1]
    gtilde = spring_gradient_fn(Rs)[1:-1]
    tauhat = calculate_tau_hat(Rs)

    #split the true potential up into parallel and perpendicular parts
    g_parallel = vmap_projection(g, tauhat)
    g_perp = g - g_parallel

    #split the spring potential up into parallel and perpendicular parts
    gtilde_parallel = vmap_projection(gtilde, tauhat)
    gtilde_perp = gtilde - gtilde_parallel

    #calculate gtilde_star via eq. (13). this is the "second nudge"
    g_perp_hat = g_perp / jnp.linalg.norm(g_perp, axis=(1, 2), keepdims=True)
    gtilde_star = gtilde_perp - vmap_projection(gtilde_perp, g_perp_hat)

    #get the total gradient via eq. (12) with some parts (g_parallel and gtilde_perp) projected out
    gtotal = g_perp + gtilde_parallel + gtilde_star

    #put zeros at the beginning and end, and multiply by -1 to make the gradient a force
    temp = jnp.zeros((1,) + tauhat[0].shape, dtype=Rs.dtype)
    return -jnp.concatenate((temp, gtotal, temp))

  return total_force


def setup_endpoint_minimization(energy_fn,
                                shift_fn,
                                max_grad_thresh=1e-12,
                                max_num_steps=100000,
                                dt_start=0.001,
                                dt_max=0.004):
  """ Define a function to simultaneously minimize all endpoints

  Args:
    energy_fn: a function that takes an array of shape (N,d) and returns an 
      energy
    shift_fn: a standard jax-md shift function
    max_grad_thresh: float indicating the primary stopping condition for the
      energy minimization
    max_num_steps: int indicating the secondary stopping condition for the 
      energy minimization
    dt_start: parameter for the fire algorithm
    dt_max: parameter for the fire algorithm

  Return: a function that takes an array of shape (n,N,d) and minimizes all n 
    systems independently

  Note: this may or may not be more efficient than minimizing the systems 
    individually. In operates on them in parallel, which has some benefits and
    some drawbacks.
  """

  energy_full_fn = vmap(energy_fn)

  @jit
  def energy_full_sum_fn(R):
    return jnp.sum(energy_full_fn(R))

  fire_init, fire_apply = minimize.fire_descent(
      energy_full_sum_fn, vmap(shift_fn), dt_start=dt_start, dt_max=dt_max)
  fire_apply = jit(fire_apply)

  @jit
  def get_maxgrad(state):
    return jnp.amax(jnp.abs(state.force))

  @jit
  def cond_fn(val):
    state, i = val
    return jnp.logical_and(
        get_maxgrad(state) > max_grad_thresh, i < max_num_steps)

  @jit
  def body_fn(val):
    state, i = val
    return fire_apply(state), i + 1

  @jit
  def minimize_endpoints(Rs):
    """ Independently minimize a collection of systems. In practice this will 
          be DNEB endpoints.
    Args:
      Rs: array of initial positions of shape (n, N, d), where n is number of
            systems, N is the system size, and d is the dimension of space.
    Return: 
      - array of final positions with the same shape.
      - maxgrad
      - num_iterations
    """
    state = fire_init(jnp.array(Rs))
    state, num_iterations = lax.while_loop(cond_fn, body_fn, (state, 0))
    return state.position, get_maxgrad(state), num_iterations

  return minimize_endpoints


def setup_DNEB_minimization(true_energy_fn,
                            spring_energy_fn,
                            displacement_fn,
                            shift_fn,
                            dt_start=0.0001,
                            dt_max=0.0004,
                            num_images=100,
                            max_grad_thresh=1e-12,
                            max_num_steps=100000,
                            return_dneb_force_fn=False):
  """ Define a function to minimize a DNEB
  """

  true_energy_full_fn = vmap(true_energy_fn)

  total_force = setup_DNEB_force(true_energy_fn, spring_energy_fn, displacement_fn)
  total_force = jit(total_force)

  fire_init, fire_apply = minimize.fire_descent(
      total_force, vmap(shift_fn), dt_start=dt_start, dt_max=dt_max)
  fire_apply = jit(fire_apply)

  @jit
  def get_maxgrad(state):
    return jnp.amax(jnp.abs(state.force))

  @jit
  def cond_fn(val):
    state, i = val
    return jnp.logical_and(
        get_maxgrad(state) > max_grad_thresh, i < max_num_steps)

  @jit
  def body_fn(val):
    state, i = val
    return fire_apply(state), i + 1

  #@jit
  def minimize_DNEB(R_input, verbose=False):
    """ minimize a DNEB

    Args: 
      R_input: list or array of shape (n, N, d)
        If n==2, these are taken to be DNEB endpoints and image states are 
          generated using linear interpolation
        If n>2, these are taken to be the full DNEB path

      Return: (Rfinal, Efinal)
        Rfinal: an array of shape (num_images, N, d) or (n, N, d) 
          giving the positions of all images as found by the DNEB
        Efinal: array of shape (num_images,) or (n,) giving the true 
          energy of each image
    """
    R_input = jnp.array(R_input)
    if R_input.shape[0] > 2:
      Rinit = R_input
    elif R_input.shape[0] == 2:
      R1 = R_input[0]
      R2 = R_input[1]
      Rinit = interpolate_positions(R1, R2, num_images, displacement_fn, shift_fn)
    else:
      assert False

    state = fire_init(Rinit)
    state, num_iterations = lax.while_loop(cond_fn, body_fn, (state, 0))

    if verbose:
      print('finished minimizing dneb. max_grad = {}, num_iter = {}'.format(jnp.amax(jnp.abs(state.force)), num_iterations))

    Rfinal = state.position
    Efinal = true_energy_full_fn(Rfinal)

    return Rfinal, Efinal  #This tuple is sometimes called "results"

  if return_dneb_force_fn:
    return minimize_DNEB, total_force
  else:
    return minimize_DNEB


def find_DNEB_paths(R_endpoints,
                    transitions,
                    true_energy_fn,
                    spring_energy_fn,
                    displacement_fn,
                    shift_fn,
                    max_num_steps=10000,
                    max_grad_thresh=1e-12,
                    dt_start=0.001,
                    dt_max=0.004,
                    num_images=100,
                    max_num_steps_DNEB=10000,
                    max_grad_thresh_DNEB=1e-12,
                    dt_start_DNEB=0.001,
                    dt_max_DNEB=0.004,
                    minimize_endpoints=True,
                    verbose=False
                    ):
  """ Minimize the set of endpoints and run DNEB calculation between endpoint pairs

  This is the primary API function. 

  Args:
    R_endpoints: array of shape (e, N, d) corresponding to e endpoints
    transitions: array of shape (t, 2) where t is the number of transitions to consider.
      If e==2, set transitions to [[0,1]] since there is only 1 possible transition.
    true_energy_fn:  A function that takes an (N,d) array of positions and 
      returns an energy. 
    spring_energy_fn: A function that takes an (m,N,d) array of positions and 
      returns an energy (corresponding to the springs that couple the m systems)
    displacement_fn: a jaxmd displacement function
    shift_fn: a standard jax-md shift function


  Returns:
    - array of shape (e, N, d) corresponding to the minimized endpoints
    - list of length t containing DNEB results
  """

  if minimize_endpoints:
    minimize_endpoints = setup_endpoint_minimization(
        true_energy_fn,
        shift_fn,
        max_num_steps=max_num_steps,
        max_grad_thresh=max_grad_thresh,
        dt_start=dt_start,
        dt_max=dt_max)
    R_endpoints_new, _, _ = minimize_endpoints(R_endpoints)
  else:
    R_endpoints_new = R_endpoints

  minimize_DNEB = setup_DNEB_minimization(
      true_energy_fn,
      spring_energy_fn,
      displacement_fn,
      shift_fn,
      num_images=num_images,
      max_num_steps=max_num_steps_DNEB,
      max_grad_thresh=max_grad_thresh_DNEB,
      dt_start=dt_start_DNEB,
      dt_max=dt_max_DNEB)
  
  DNEB_results_all = [
      minimize_DNEB(Rs, verbose) for Rs in R_endpoints_new[jnp.array(transitions)]
  ]
  return R_endpoints_new, DNEB_results_all


############################################################################
######################### Analysis Functions ###############################
############################################################################


def extract_energies(results):
  """ Extract the energies of the two endpoints and the (maximum) barrier

  Args:
    results: the list [Rfinal,Efinal] as returned from a DNEB calculation

  Returns: (E1, E2, ET), i.e. a tuple containing the energies at the two
    endpoints and the maximum energy. Note, it is possible for ET to be
    the energy at one of the two endpoints (i.e. if there is no barrier)
  """
  _, Efinal = results
  return Efinal[0], Efinal[-1], jnp.amax(Efinal)


def extract_positions(results):
  """ Extract the positions of the two endpoints and the (maximum) barrier

  Args:
    results: the list [Rfinal,Efinal] as returned from a DNEB calculation

  Returns: (R1, R2, RT), i.e. a tuple containing the positions at the two
    endpoints and the image with the maximum energy. Note, it is possible 
    for RT to be one of the two endpoints (i.e. if there is no barrier)
  """
  Rfinal, Efinal = results
  idx = jnp.argmax(Efinal)
  return Rfinal[0], Rfinal[-1], Rfinal[idx]


############################################################################
########################## Utility Functions ###############################
############################################################################


def setup_coupling_spring_energy(k_spr, displacement):
  half_k_spr = 0.5 * k_spr

  mapped_displacement = vmap(vmap(
    displacement, in_axes = (0,0)), in_axes = (0,0))

  def spring_energy_fn(R):
    """ Calculate the total energy that couples neighboring images

    Args:
      R: array of shape (num_images, N, d)

    Returns: the total coupling energy
    """
    #return half_k_spr * jnp.sum((R[1:] - R[:-1])**2)
    return half_k_spr * jnp.sum( mapped_displacement(R[:-1], R[1:]) ** 2)

  return spring_energy_fn


#This is one of the methods from the transformations.py package
def _affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
  """Return affine transform matrix to register two point sets.
    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.
    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.
    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.
    The returned matrix performs rotation, translation and uniform scaling
    (if specified).
    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(onp.random.random(3)-0.5)
    >>> R = random_rotation_matrix(onp.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (onp.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = onp.dot(M, v0)
    >>> v0[:3] += onp.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> onp.allclose(v1, onp.dot(M, v0))
    True
    More examples in superimposition_matrix()
    """
  v0 = onp.array(v0, dtype=onp.float64, copy=True)
  v1 = onp.array(v1, dtype=onp.float64, copy=True)

  ndims = v0.shape[0]
  if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
    raise ValueError('input arrays are of wrong shape or type')

  # move centroids to origin
  t0 = -onp.mean(v0, axis=1)
  M0 = onp.identity(ndims + 1)
  M0[:ndims, ndims] = t0
  v0 += t0.reshape(ndims, 1)
  t1 = -onp.mean(v1, axis=1)
  M1 = onp.identity(ndims + 1)
  M1[:ndims, ndims] = t1
  v1 += t1.reshape(ndims, 1)

  if shear:
    # Affine transformation
    A = onp.concatenate((v0, v1), axis=0)
    u, s, vh = onp.linalg.svd(A.T)
    vh = vh[:ndims].T
    B = vh[:ndims]
    C = vh[ndims:2 * ndims]
    t = onp.dot(C, onp.linalg.pinv(B))
    t = onp.concatenate((t, onp.zeros((ndims, 1))), axis=1)
    M = onp.vstack((t, ((0.0,) * ndims) + (1.0,)))
  elif usesvd or ndims != 3:
    # Rigid transformation via SVD of covariance matrix
    u, s, vh = onp.linalg.svd(onp.dot(v1, v0.T))
    # rotation matrix from SVD orthonormal bases
    R = onp.dot(u, vh)
    if onp.linalg.det(R) < 0.0:
      # R does not constitute right handed system
      R -= onp.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
      s[-1] *= -1.0
    # homogeneous transformation matrix
    M = onp.identity(ndims + 1)
    M[:ndims, :ndims] = R
  else:
    # Rigid transformation matrix via quaternion
    # compute symmetric matrix N
    xx, yy, zz = onp.sum(v0 * v1, axis=1)
    xy, yz, zx = onp.sum(v0 * onp.roll(v1, -1, axis=0), axis=1)
    xz, yx, zy = onp.sum(v0 * onp.roll(v1, -2, axis=0), axis=1)
    N = [[xx + yy + zz, 0.0, 0.0, 0.0], [yz - zy, xx - yy - zz, 0.0, 0.0],
         [zx - xz, xy + yx, yy - xx - zz, 0.0],
         [xy - yx, zx + xz, yz + zy, zz - xx - yy]]
    # quaternion: eigenvector corresponding to most positive eigenvalue
    w, V = onp.linalg.eigh(N)
    q = V[:, onp.argmax(w)]
    q /= vector_norm(q)  # unit quaternion
    # homogeneous transformation matrix
    M = quaternion_matrix(q)

  if scale and not shear:
    # Affine transformation; scale is ratio of RMS deviations from centroid
    v0 *= v0
    v1 *= v1
    M[:ndims, :ndims] *= math.sqrt(onp.sum(v1) / onp.sum(v0))

  # move centroids back
  M = onp.dot(onp.linalg.inv(M1), onp.dot(M, M0))
  M /= M[ndims, ndims]
  return M


def align_points(R1, R2):
  _R1 = onp.array(R1)
  _R2 = onp.array(R2)
  M = _affine_matrix_from_points(
      _R2.T, _R1.T, shear=False, scale=False, usesvd=True)
  M = jnp.array(M)

  R2temp = jnp.pad(
      jnp.atleast_2d(R2), ((0, 0), (0, 1)),
      mode='constant',
      constant_values=1.0)
  return jnp.reshape((jnp.matmul(M, R2temp.T)[:3]).T, _R2.shape)
