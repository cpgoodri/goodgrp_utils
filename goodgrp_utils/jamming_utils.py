import jax.numpy as jnp
from jax import vmap
from jax_md import space





def _remove_rattlers_oneshot(R, bonds, node_arrays, bond_arrays):
  N, dimension = R.shape
  #Z_alpha = get_Z_alpha(bonds, R.shape[0])
  Z_alpha = jnp.bincount(bonds.reshape(bonds.size), length=R.shape[0])
  rattler_yesno = jnp.where(Z_alpha > dimension, False, True)
  
  if jnp.any(rattler_yesno):
    nodes_to_keep = jnp.where(rattler_yesno == False)
    node_map = jnp.full((N,), -1).at[nodes_to_keep].set(jnp.arange(nodes_to_keep[0].shape[0]))

    rattlers, = jnp.where(rattler_yesno == True)
    bonds_with_rattlers = vmap(jnp.any)(jnp.isin(bonds, rattlers)) #boolean vector of length N_bonds, True if bond contains a rattler node
    bonds_to_keep = jnp.where(bonds_with_rattlers==False)

    R_new = R[nodes_to_keep]
    if node_arrays is None:
      node_arrays_new = None
    else:
      node_arrays_new = [a[nodes_to_keep] for a in node_arrays]

    bonds_new = node_map[bonds[bonds_to_keep]]
    if bond_arrays is None:
      bond_arrays_new = None
    else:
      bond_arrays_new = [a[bonds_to_keep] for a in bond_arrays]

    return R_new, bonds_new, node_arrays_new, bond_arrays_new, True
  return R, bonds, node_arrays, bond_arrays, False

def remove_rattlers(R, bonds, node_arrays = None, bond_arrays = None):
  """ Remove rattlers from a network

  Recursively removes all nodes (and connected bonds) which do not have at least
  dimension+1 bonds. Both R and bonds are updated. 

  Args:
    R:            Array of length (N, dimension) of node positions
    bonds:        Array of length (Nbonds, 2) of bond indices
    node_arrays:  List of node-based arrays. If node i is identified as a
                  rattler and removed, element i of each array is also removed.
    bond_arrays:  List of bond-based arrays. If bond i is connected to a rattler
                  and removed, element i of each array is also removed.
  
  Return: new versions of R, bonds, node_arrays, bond_arrays

  Note: the contents of bonds is updated to reflect the new indices of nodes
    in R. However, the contents of node_arrays and bond_arrays are not updated 
    other than removing the appropriate elements. If you need to map old indices
    to new indices, this can be obtained by passing jnp.arange(N) to the 
    node_arrays list. e.g. 
      R_new, bonds, [index_map], _ = remove_rattlers(R, bonds, [jnp.arange(N)])
      R[(index_map,)] == R_new # All True
  """ 
  _R = R
  _bonds = bonds
  _node_arrays = node_arrays
  _bond_arrays = bond_arrays
  keep_trying = True

  ii = 0
  while keep_trying:
    ii += 1
    print('removing rattlers, iteration {}'.format(ii))
    _R, _bonds, _node_arrays, _bond_arrays, keep_trying = _remove_rattlers_oneshot(_R, _bonds, _node_arrays, _bond_arrays)
    if R.shape[0] < 1:
      keep_trying = False
  
  return _R, _bonds, _node_arrays, _bond_arrays

def get_dNciso(Nc, N, dimension):
  return Nc - dimension * (N - 1)

def calculate_bond_data(displacement_or_metric, R, dr_cutoff, species=None):
  if species is not None:
    #TODO
    assert(False)
    
  metric = space.map_product(space.canonicalize_displacement_or_metric(displacement))
  dr = metric(R,R)

  dr_include = jnp.triu(jnp.where(dr<dr_cutoff, 1, 0)) - jnp.eye(R.shape[0],dtype=jnp.int32)
  index_list=jnp.dstack(jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij'))

  i_s = jnp.where(dr_include==1, index_list[:,:,0], -1).flatten()
  j_s = jnp.where(dr_include==1, index_list[:,:,1], -1).flatten()
  ij_s = jnp.transpose(jnp.array([i_s,j_s]))

  bonds = ij_s[(ij_s!=jnp.array([-1,-1]))[:,1]]
  lengths = dr.flatten()[(ij_s!=jnp.array([-1,-1]))[:,1]]

  return bonds, lengths

def analyze_connectivity(bonds):
  unique, counts = jnp.unique(bonds, return_counts=True)
  Npp = unique.shape[0]
  Nc = bonds.shape[0]
  Z = 2 * Nc / Npp
  return Npp, Nc, Z, unique, counts




