from absl.testing import absltest

from jax import test_util as jtu
import jax.numpy as jnp

from common_utils import utils

from jax.config import config as jax_config
jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS



class utilsTest(jtu.JaxTestCase):

  def test_vector2symmat(self):
    v = jnp.array([0,1,2,3,4,5],dtype=jnp.float32)
    m = utils.vector2symmat(v)
    m_expected = jnp.array([[0,1,2],
                            [1,3,4],
                            [2,4,5]],dtype=v.dtype)
    self.assertAllClose(m, m_expected)

  def test_vector2symmat_diag0(self):
    v = jnp.array([0,1,2,3,4,5],dtype=jnp.float32)
    m = utils.vector2symmat_diag0(v)
    m_expected = jnp.array([[0,0,1,2],
                            [0,0,3,4],
                            [1,3,0,5],
                            [2,4,5,0]],dtype=v.dtype)
    self.assertAllClose(m, m_expected)




if __name__ == '__main__':
  absltest.main()
