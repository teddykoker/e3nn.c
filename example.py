import jax.numpy as jnp
import e3nn_jax as e3nn


node_position = jnp.asarray([1, 2, 3])
node_position_sh = e3nn.spherical_harmonics("1x0e + 1x1o + 1x2e", node_position, normalize=True, normalization="component")
print("sp ", node_position_sh.array)

neighbor_feature = e3nn.IrrepsArray("1x1e", jnp.asarray([7,8,9]))
tp = e3nn.tensor_product(node_position_sh, neighbor_feature)
print("product", tp.array)
linear = e3nn.flax.Linear("1x0o + 1x1o + 2x1e + 1x2e + 1x2o + 1x3e",
                          "1x0o + 1x1o + 2x1e + 1x2e + 1x2o + 1x3e")
weights = {'params': {'w[0,0] 1x0o,1x0o': jnp.asarray([[1]]),
                      'w[1,1] 1x1o,1x1o': jnp.asarray([[2]]),
                      'w[2,2] 2x1e,2x1e': jnp.asarray([[3 , 4], [ 5,  6]]),
                      'w[3,3] 1x2e,1x2e': jnp.asarray([[7]]),
                      'w[4,4] 1x2o,1x2o': jnp.asarray([[8]]),
                      'w[5,5] 1x3e,1x3e': jnp.asarray([[9]])}}
message = linear.apply(weights, tp)
print("output", message.array)