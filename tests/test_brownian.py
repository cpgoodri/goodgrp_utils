from absl.testing import absltest
from absl.testing import parameterized

from jax.config import config as jax_config
from jax import test_util as jtu
from jax import random
import jax.numpy as jnp

from jax_md import energy, space, quantity
from jax_md.util import *

from common_utils import brownian

from jax.config import config as jax_config
jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


PARTICLE_COUNT = 256
SPATIAL_DIMENSION = [2, 3]
STOCHASTIC_SAMPLES = 1

if FLAGS.jax_enable_x64:
  DTYPE = [f32, f64]
else:
  DTYPE = [f32]


def setup_system(N, dimension, key, dtype):
  diameters = 1.0
  box_size = quantity.box_size_at_number_density(N, 0.4, dimension)
  displacement, shift = space.periodic(box_size)
  energy_fn = energy.soft_sphere_pair(displacement, sigma=diameters)
  R_init = random.uniform(key, (N,dimension), minval=0.0, maxval=box_size, dtype=dtype) 
  return displacement, shift, energy_fn, R_init, box_size

def setup_system_nl(N, dimension, key, dtype):
  diameters = 1.0
  box_size = quantity.box_size_at_number_density(N, 0.4, dimension)
  displacement, shift = space.periodic(box_size)
  #energy_fn = energy.soft_sphere_pair(displacement, sigma=diameters)
  neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
    displacement, box_size, sigma=diameters, dr_threshold = 0.2,
    capacity_multiplier = 1.5)
  R_init = random.uniform(key, (N,dimension), minval=0.0, maxval=box_size, dtype=dtype) 
  return displacement, shift, energy_fn, neighbor_fn, R_init, box_size


class BrownianTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(
            dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION 
        for dtype in DTYPE))
  def test_brownian(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    for _ in range(STOCHASTIC_SAMPLES):
      key, pos_key, brownian_key = random.split(key, 3)
      
      _, shift, energy_fn, R_init, _ = setup_system(
          PARTICLE_COUNT, spatial_dimension, pos_key, dtype)
      
      nsteps, record_every = 1000, 10
      final_state, trajectory = brownian.run_brownian(
          energy_fn, R_init, shift, brownian_key, 
          num_total_steps=nsteps, record_every=record_every, 
          dt=0.0001, kT=0.01, gamma=0.1)
      
      assert trajectory.shape == (nsteps//record_every, PARTICLE_COUNT, spatial_dimension)
      self.assertAllClose(final_state.position, trajectory[-1])

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(
            dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION 
        for dtype in DTYPE))
  def test_measurement(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    for _ in range(STOCHASTIC_SAMPLES):
      key, pos_key, brownian_key = random.split(key, 3)

      displacement, shift, energy_fn, R_init, box_size = setup_system(
          PARTICLE_COUNT, spatial_dimension, pos_key, dtype)

      rs = jnp.linspace(0,box_size/2.0, 101)[1:]
      g_fn = quantity.pair_correlation(displacement, rs, 0.1)
      def measurement(R):
        return jnp.mean(g_fn(R),axis=0)

      nsteps, record_every = 1000, 10
      final_state, gofr_all = brownian.run_brownian(
          energy_fn, R_init, shift, brownian_key, 
          num_total_steps=nsteps, record_every=record_every, 
          dt=0.0001, kT=0.01, gamma=0.1,
          measure_fn = measurement)

      assert gofr_all.shape == (nsteps//record_every, 100)
      assert final_state.position.shape == R_init.shape



  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(
            dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION 
        for dtype in DTYPE))
  def test_brownian_nl(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    for _ in range(STOCHASTIC_SAMPLES):
      key, pos_key, brownian_key = random.split(key, 3)
      
      _, shift, energy_fn, neighbor_fn, R_init, _ = setup_system_nl(
          PARTICLE_COUNT, spatial_dimension, pos_key, dtype)
      
      nsteps, record_every = 1000, 10
      final_state, trajectory, nbrs = brownian.run_brownian_neighbor_list(
          energy_fn, neighbor_fn, R_init, shift, brownian_key, 
          num_steps=nsteps, step_inc=record_every, 
          dt=0.0001, kT=0.001, gamma=0.1)
      
      assert trajectory.shape == (nsteps//record_every, PARTICLE_COUNT, spatial_dimension)
      self.assertAllClose(final_state.position, trajectory[-1])

  @parameterized.named_parameters(jtu.cases_from_list(
      {
          'testcase_name': '_dim={}_dtype={}'.format(
            dim, dtype.__name__),
          'spatial_dimension': dim,
          'dtype': dtype
      } for dim in SPATIAL_DIMENSION 
        for dtype in DTYPE))
  def test_measurement_nl(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    for _ in range(STOCHASTIC_SAMPLES):
      key, pos_key, brownian_key = random.split(key, 3)
      
      displacement, shift, energy_fn, neighbor_fn, R_init, box_size = setup_system_nl(
          PARTICLE_COUNT, spatial_dimension, pos_key, dtype)
     
      rs = jnp.linspace(0,4.0, 101)[1:]
      gnbr_fn, g_fn = quantity.pair_correlation_neighbor_list(displacement, box_size, rs, 0.1)
      gnbrs = gnbr_fn.allocate(R_init)
      def measurement(R,nbrs,gnbrs):
        gnbrs = gnbr_fn.update(R, gnbrs)
        return jnp.mean(g_fn(R,gnbrs),axis=0)

      nsteps, record_every = 1000, 10
      final_state, gofr_all, nbrs = brownian.run_brownian_neighbor_list(
          energy_fn, neighbor_fn, R_init, shift, brownian_key, 
          num_steps=nsteps, step_inc=record_every, 
          dt=0.0001, kT=0.001, gamma=0.1,
          measure_fn=measurement, gnbrs=gnbrs)
      
      assert gofr_all.shape == (nsteps//record_every, 100)
      assert final_state.position.shape == R_init.shape





if __name__ == '__main__':
  absltest.main()

