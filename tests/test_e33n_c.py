import ctypes
import math

import e3nn_jax._src
import e3nn_jax._src.su2
import jax
import numpy as np
import e3nn_jax
import pytest

e3nn_c = ctypes.CDLL("./e3nn.so")

for tp in [e3nn_c.tensor_product_v1, e3nn_c.tensor_product_v2, e3nn_c.tensor_product_v3]:
    tp.argtypes = (
        ctypes.c_char_p,
        np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
        ctypes.c_char_p,
        np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
        ctypes.c_char_p,
        np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    )
    tp.restype = None

clebsch_gordan_c = ctypes.CDLL("./clebsch_gordan.so")

clebsch_gordan_c.compute_clebsch_gordan.argtypes = (
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)
clebsch_gordan_c.compute_clebsch_gordan.restype = ctypes.c_float


@pytest.mark.parametrize("fn", [
    e3nn_c.tensor_product_v1, 
    e3nn_c.tensor_product_v2, 
    e3nn_c.tensor_product_v3,
])
@pytest.mark.parametrize("input1,input2", [
    (e3nn_jax.normal("2x0e + 1x1o", jax.random.PRNGKey(0)), e3nn_jax.normal("1x0o + 1x2o", jax.random.PRNGKey(0))),
    (e3nn_jax.normal("1x3o", jax.random.PRNGKey(0)), e3nn_jax.normal("1x3o", jax.random.PRNGKey(0))),
])
def test_tensor_product(fn, input1, input2):
    output = e3nn_jax.tensor_product(input1, input2)
    output_c = np.zeros_like(output.array, dtype=ctypes.c_float)
    fn(
        repr(input1.irreps).encode("utf-8"),
        np.array(input1.array, dtype=ctypes.c_float),
        repr(input2.irreps).encode("utf-8"),
        np.array(input2.array, dtype=ctypes.c_float),
        repr(output.irreps).encode("utf-8"),
        output_c
    )
    assert np.allclose(output_c, output.array, rtol=1e-5, atol=1e-6)


def test_compute_clebsch_gordan():
    l_max = 8
    for l1 in range(l_max // 2 + 1):
        for l2 in range(l_max // 2 + 1):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                jax_cg = e3nn_jax._src.so3.clebsch_gordan(l1, l2, l3)
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        for m3 in range(-l3, l3 + 1):
                            print("l1", l1, "l2", l2, "l3", l3, "m1", m1, "m2", m2, "m3", m3)
                            assert math.isclose(
                                jax_cg[m1 + l1, m2 + l2, m3 + l3],
                                clebsch_gordan_c.compute_clebsch_gordan(l1, l2, l3, m1, m2, m3),
                                rel_tol=1e-5,
                                abs_tol=1e-7,
                            )
