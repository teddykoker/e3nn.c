import ctypes
import math

import e3nn_jax._src
import e3nn_jax._src.su2
import jax
import numpy as np
from numpy.testing import assert_allclose
import e3nn_jax
import pytest

e3nn_c = ctypes.CDLL("./e3nn.so")

class Irrep(ctypes.Structure):
    _fields_ = [("c", ctypes.c_int),
                ("l", ctypes.c_int),
                ("p", ctypes.c_int)]

class Irreps(ctypes.Structure):
    _fields_ = [("irreps", ctypes.POINTER(Irrep)),
                ("size", ctypes.c_int)]

e3nn_c.irreps_create.argtypes = (ctypes.c_char_p,)
e3nn_c.irreps_create.restype = ctypes.POINTER(Irreps)
e3nn_c.irreps_free.argtypes = (ctypes.POINTER(Irreps),)
e3nn_c.irreps_free.restype = None

for tp in [e3nn_c.tensor_product_v1, e3nn_c.tensor_product_v2, e3nn_c.tensor_product_v3]:
    tp.argtypes = (
        ctypes.POINTER(Irreps),
        np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
        ctypes.POINTER(Irreps),
        np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
        ctypes.POINTER(Irreps),
        np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    )
    tp.restype = None

e3nn_c.spherical_harmonics.argtypes = (
    ctypes.POINTER(Irreps),
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
)
e3nn_c.spherical_harmonics.restype = None

e3nn_c.linear.argtypes = (
    ctypes.POINTER(Irreps),
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    ctypes.POINTER(Irreps),
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
)
e3nn_c.linear.restype = None

e3nn_c.concatenate.argtypes = (
    ctypes.POINTER(Irreps),
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    ctypes.POINTER(Irreps),
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
)
e3nn_c.concatenate.restype = None

e3nn_c.gate.argtypes = (
    ctypes.POINTER(Irreps),
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    ctypes._CFuncPtr,
    ctypes._CFuncPtr,
    ctypes._CFuncPtr,
    ctypes._CFuncPtr,
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
)
e3nn_c.gate.restype = None

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
    # e3nn_c.tensor_product_v1, # commenting out for now bc slow
    # e3nn_c.tensor_product_v2, # commenting out for now bc slow
    e3nn_c.tensor_product_v3,
])
@pytest.mark.parametrize("input1,input2", [
    (e3nn_jax.normal("2x0e + 1x1o", jax.random.PRNGKey(0)), e3nn_jax.normal("1x0o + 1x2o", jax.random.PRNGKey(0))),
    (e3nn_jax.normal("1x3o", jax.random.PRNGKey(0)), e3nn_jax.normal("1x3o", jax.random.PRNGKey(0))),
])
def test_tensor_product(fn, input1, input2):
    output = e3nn_jax.tensor_product(input1, input2)
    output_c = np.zeros_like(output.array, dtype=ctypes.c_float)
    irreps1 = e3nn_c.irreps_create(repr(input1.irreps).encode("utf-8"))
    irreps2 = e3nn_c.irreps_create(repr(input2.irreps).encode("utf-8"))
    irrepso = e3nn_c.irreps_create(repr(output.irreps).encode("utf-8"))
    fn(
        irreps1,
        np.array(input1.array, dtype=ctypes.c_float),
        irreps2,
        np.array(input2.array, dtype=ctypes.c_float),
        irrepso,
        output_c
    )
    assert_allclose(output_c, output.array, rtol=1e-5, atol=1e-6)

    e3nn_c.irreps_free(irreps1)
    e3nn_c.irreps_free(irreps2)
    e3nn_c.irreps_free(irrepso)


@pytest.mark.parametrize("irreps", [
    "1x0e",
    "1x0e + 1x1o + 1x2e",
    "2x0e + 3x1e + 4x2e",
    "2x0e + 3x1o + 4x2e",
    "1x0e + 1x1o + 1x2e + 1x3o",
    "3x3e",
    "1x0e + 1x1e + 1x2e + 1x3e + 1x4e + 1x5e + 1x6e + 1x7e + 1x8e",
])
@pytest.mark.parametrize("input", [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1, 2, 3],
])
def test_spherical_harmonics(irreps, input):
    output = e3nn_jax.spherical_harmonics(irreps, np.array(input), normalize=True, normalization="component")
    output_c = np.zeros_like(output.array)
    irreps = e3nn_c.irreps_create(irreps.encode("utf-8"))
    e3nn_c.spherical_harmonics(irreps, *input, output_c)
    assert_allclose(output_c, output.array, rtol=1e-5, atol=1e-6)

    e3nn_c.irreps_free(irreps)


@pytest.mark.parametrize("irreps_in,irreps_out", [
    ("2x0e + 2x1o", "3x0e + 3x1o"),
    ("1x0e + 2x1o", "3x0e + 3x1o"),
    ("1x0e + 2x1o", "3x0e + 3x1o + 1x2e"),
    ("1x0e + 2x1o + 1x2e", "3x0e + 3x1o"),
])
def test_linear(irreps_in, irreps_out):
    input = e3nn_jax.normal(irreps_in, jax.random.PRNGKey(0))
    linear = e3nn_jax.flax.Linear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
    )
    weight = linear.init(jax.random.PRNGKey(0), input)
    output = linear.apply(weight, input)
    
    weight_c = np.concatenate([w.ravel() for w in weight["params"].values()])
    output_c = np.zeros_like(output.array)
    irreps_in = e3nn_c.irreps_create(irreps_in.encode("utf-8"))
    irreps_out = e3nn_c.irreps_create(irreps_out.encode("utf-8"))
    e3nn_c.linear(irreps_in, np.array(input.array), weight_c, irreps_out, output_c)
    assert_allclose(output_c, output.array, rtol=1e-5, atol=1e-6)

    e3nn_c.irreps_free(irreps_in)
    e3nn_c.irreps_free(irreps_out)


@pytest.mark.parametrize("irreps_1,irreps_2", [
    ("2x0e + 2x1o", "3x0e + 3x1o"),
    ("1x0e + 2x1o", "3x0e + 3x1o"),
    ("1x0e + 2x1o", "3x0e + 3x1o + 1x2e"),
    ("1x0e + 2x1o + 1x2e", "3x0e + 3x1o"),
])
def test_concatenate(irreps_1, irreps_2):
    array_1 = e3nn_jax.normal(irreps_1, jax.random.PRNGKey(0))
    array_2 = e3nn_jax.normal(irreps_2, jax.random.PRNGKey(1))
    # e33n.c should concatenate and regroup
    output = e3nn_jax.concatenate([array_1, array_2]).regroup()
    output_c = np.zeros_like(output.array)
    irreps_1 = e3nn_c.irreps_create(irreps_1.encode("utf-8"))
    irreps_2 = e3nn_c.irreps_create(irreps_2.encode("utf-8"))
    e3nn_c.concatenate(irreps_1, np.array(array_1.array), irreps_2, np.array(array_2.array), output_c)
    assert_allclose(output_c, output.array, rtol=1e-5, atol=1e-6)

    e3nn_c.irreps_free(irreps_1)
    e3nn_c.irreps_free(irreps_2)


def test_gate():
    irreps = "15x0e + 2x1e + 1x2e"
    array = e3nn_jax.normal(irreps, jax.random.PRNGKey(0))
    output = e3nn_jax.gate(
        array,
        jax.nn.silu,
        jax.nn.tanh,
        jax.nn.silu,
        jax.nn.silu,
        normalize_act=True,
    )
    output_c = np.zeros_like(output.array)
    irreps_c = e3nn_c.irreps_create(irreps.encode("utf-8"))
    e3nn_c.gate(
        irreps_c,
        np.array(array.array),
        e3nn_c.silu_normalized,
        e3nn_c.tanh_normalized,
        e3nn_c.silu_normalized,
        e3nn_c.silu_normalized,
        output_c,
    )
    assert_allclose(output_c, output.array, rtol=1e-5, atol=1e-6)
    e3nn_c.irreps_free(irreps_c)

    irreps = "12x0e + 3x0o + 2x1e + 1x2e"
    array = e3nn_jax.normal(irreps, jax.random.PRNGKey(0))
    output = e3nn_jax.gate(
        array,
        jax.nn.gelu,
        e3nn_jax.soft_odd,
        jax.nn.sigmoid,
        jax.nn.tanh,
        normalize_act=True,
    )
    output_c = np.zeros_like(output.array)
    irreps_c = e3nn_c.irreps_create(irreps.encode("utf-8"))
    e3nn_c.gate(
        irreps_c,
        np.array(array.array),
        e3nn_c.gelu_normalized,
        e3nn_c.soft_odd_normalized,
        e3nn_c.sigmoid_normalized,
        e3nn_c.tanh_normalized,
        output_c,
    )
    assert_allclose(output_c, output.array, rtol=1e-5, atol=1e-6)
    e3nn_c.irreps_free(irreps_c)

    irreps = "12x0e + 3x0o"
    array = e3nn_jax.normal(irreps, jax.random.PRNGKey(0))
    output = e3nn_jax.gate(
        array,
        jax.nn.gelu,
        e3nn_jax.soft_odd,
        jax.nn.sigmoid,
        jax.nn.tanh,
        normalize_act=True,
    )
    output_c = np.zeros_like(output.array)
    irreps_c = e3nn_c.irreps_create(irreps.encode("utf-8"))
    e3nn_c.gate(
        irreps_c,
        np.array(array.array),
        e3nn_c.gelu_normalized,
        e3nn_c.soft_odd_normalized,
        e3nn_c.sigmoid_normalized,
        e3nn_c.tanh_normalized,
        output_c,
    )
    assert_allclose(output_c, output.array, rtol=1e-5, atol=1e-6)
    e3nn_c.irreps_free(irreps_c)


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
