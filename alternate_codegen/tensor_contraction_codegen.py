import numpy as np

import jinja2
import e3nn
import e3nn_jax 

def remove_similar_values(arr : np.ndarray, tolerance=1e-6): 
    unique_values = np.array([], dtype=arr.dtype)
    for value in arr:
        if np.all(np.abs(value - unique_values) > tolerance):
            if np.allclose(value, 1):
                unique_values = np.append(unique_values, 1)
            else:
                unique_values = np.append(unique_values, value)
    return unique_values

def get_coefficients(t : np.ndarray):
    """list of dictionaries of coefficients"""
    return (
        [ {"index": index, "value" : value.__repr__()} 
         for index, value 
         in list(enumerate(list(remove_similar_values(np.unique(np.abs(t[t!=0]))))))]
            )

def get_inputs(func_info : dict, t : np.ndarray):
    """Returns all the input values that are used"""
    in1_inputs = np.argwhere(np.max(t != 0, axis=(1,2)))
    in2_inputs = np.argwhere(np.max(t != 0, axis=(0,2)))
    return (
        [{"type":func_info["in_1_type"], "index":i.item(), "name":1} for i in in1_inputs] + 
        [{"type":func_info["in_2_type"], "index":i.item(), "name":2} for i in in2_inputs]
    )

def get_outputs(t : np.ndarray):
    """returns all the output indexes"""
    return [{"index":i} for i in range(0,t.shape[2])] 

def get_outer_products(t : np.ndarray):
    """Returns all the needed outer product values"""
    return [ {"in_1_index":in_1_index,"in_2_index":in_2_index} for in_1_index,in_2_index in np.argwhere(np.max(t != 0, axis=2))]

def get_contractions(t : np.ndarray):
    """Returns all nonzero functions in the format (index_0, index_1, index_2, index_coeff, sign)"""
    non_zero_indices = np.argwhere(t)
    unique_coeffs = remove_similar_values(np.unique(np.abs(t[t!=0])))
    coeff_values = t[np.nonzero(t)]
    coeff_indices = np.argmin(np.abs(unique_coeffs[:, None] - abs(coeff_values)), axis=0)
    signs = ['+' if sign == 1 else '-' for sign in np.sign(coeff_values)]
    return (
    {
        "in_1_index": in_1_index,
        "in_2_index": in_2_index,
        "out_index": out_index,
        "coeff_index": coeff_index,
        "sign":sign
     } 
    for in_1_index, in_2_index, out_index, coeff_index, sign
    in zip(non_zero_indices[:,0], non_zero_indices[:,1], non_zero_indices[:,2], coeff_indices, signs)
    )

def create_func_info_for_cg(l_in1 : int, l_in2 : int, l_out : int):
    return {
        "func_name" : f"tensor_product_cg_{l_in1}_{l_in2}_{l_in2}",
        "in_1_type" : "float",
        "in_2_type" : "float",
        "out_type" : "float",
        "coeff_type" :"float",
    }

def create_contraction_kernel_info(func_info : dict, t : np.ndarray):
    assert isinstance(t,np.ndarray)
    assert t.ndim == 3
    func_info['coefficients'] = get_coefficients(t)
    func_info['inputs'] = get_inputs(func_info, t)
    func_info['outputs'] = get_outputs(t)
    func_info['outer_products'] = get_outer_products(t)
    func_info['contractions'] = get_contractions(t)

    return func_info

def create_cg_contraction_kernel_info(l_in1 : int, l_in2 : int, l_out : int):
    func_info = create_func_info_for_cg(l_in1, l_in2, l_out)
    cg_tensor = e3nn.o3.wigner_3j(l_in1, l_in2, l_out).numpy() * np.sqrt(2 * l_out + 1)
    return create_contraction_kernel_info(func_info, cg_tensor)

def create_cg_contraction_c_kernel(l_in1 : int, l_in2 : int, l_out : int):
    func_info = create_cg_contraction_kernel_info(l_in1, l_in2, l_out)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("jinja/templates/"), 
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        )
    template = env.get_template(name='tensor_product.c.jinja2')

    filename = f"kernels/c/tensor_product_{func_info['func_name']}.c"
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write(template.render(func_info))

def create_cg_contraction_cpp_kernel(l_in1 : int, l_in2 : int, l_out : int):
    func_info = create_cg_contraction_kernel_info(l_in1, l_in2, l_out)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("jinja/templates/"), 
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        )
    template = env.get_template(name='tensor_product.cpp.jinja2')

    filename = f"kernels/cpp/tensor_product_{func_info['func_name']}.cpp"
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write(template.render(func_info))

def write_tp_dot_c():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("jinja/templates/"), 
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        )
    template = env.get_template(name='tensor_product.c.jinja2')
    contents = ''
    header = (
"""#include "tp_v2.h"
#include "clebsch_gordan.h"
#include <stddef.h>

typedef void (*tp_ptr)(const float*, const float*, float*);

""")

    contents += header
    LMAX_1 = 7 
    LMAX_2 = 14
    footer = f"tp_ptr tp_v2s[{LMAX_1 + 1}][{LMAX_1 + 1}][{LMAX_2 + 1}] = \u007b\n"

    for l_in1 in range(LMAX_1 + 1):
        for l_in2 in range(LMAX_1 + 1):
            for l_out in range(abs(l_in1 - l_in2), min(l_in1 + l_in2, LMAX_2)+1):
                func_info = create_cg_contraction_kernel_info(l_in1, l_in2, l_out)
                func_name = f"tp_v2_{l_in1}_{l_in2}_{l_out}"
                func_info["func_name"] = func_name
                contents += template.render(func_info)
                contents += '\n'
                footer += f"[{l_in1}][{l_in2}][{l_out}] = {func_name},\n"

    footer += (
r"""};

void tp_v2(int l1, int l2, int l3, const float* input1, const float* input2, float* output) {
tp_v2s[l1][l2][l3](input1, input2, output);
}""")

    contents += footer

    filename = f"kernels/tp_v2.c"
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write(contents)


def main():
    write_tp_dot_c()
    

if __name__ == "__main__":
    main()