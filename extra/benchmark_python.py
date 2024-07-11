import time
from threadpoolctl import threadpool_limits

# Being defensive here
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

channels = 128
warmup = 10
trials = 100
L = 8


def benchmark(func, *args, **kwargs):
    for _ in range(warmup):
        result = func(*args)

    start = time.process_time()

    with threadpool_limits(limits=1):
        for _ in range(trials):
            result = func(*args)
        # Dropping the sychronization code since one thread restriction should be good enough

    return time.process_time() - start    

def run_e3nn_jax(lmax):
    import e3nn_jax
    import jax

    irreps1 = e3nn_jax.Irreps.spherical_harmonics(lmax)
    irreps2 = irreps1 * channels

    input1 = e3nn_jax.zeros(irreps1, ())
    input2 = e3nn_jax.zeros(irreps2, ())

    tp = jax.jit(e3nn_jax.tensor_product)
    
    print(f"{lmax},", benchmark(tp, input1, input2))

def run_e3nn_torch(lmax):
    import torch
    import e3nn as e3nn_torch

    # torch.set_num_threads(1) # note this actually makes it faster
    with torch.no_grad():
        irreps1 = e3nn_torch.o3.Irreps.spherical_harmonics(lmax)
        irreps2 = (channels * irreps1).sort().irreps.simplify()


        input1 = irreps1.randn(-1) * 0
        input2 = irreps2.randn(-1) * 0

        tp = e3nn_torch.o3.FullTensorProduct(irreps1, irreps2)
        tp = e3nn_torch.util.jit.compile(tp)

        print(f"{lmax},", benchmark(tp, input1, input2))

def run_e3nn_torch2(lmax):
    import torch
    import e3nn as e3nn_torch

    import torch._inductor.config as config
    config.cpp_wrapper = True   
    # torch.set_num_threads(1) # note this actually makes it faster
    with torch.no_grad():
        irreps1 = e3nn_torch.o3.Irreps.spherical_harmonics(lmax)
        irreps2 = (channels * irreps1).sort().irreps.simplify()

        input1 = irreps1.randn(-1) * 0
        input2 = irreps2.randn(-1) * 0
        tp = e3nn_torch.o3.experimental.FullTensorProductv2(irreps1, irreps2)
        tp = torch.compile(tp, mode="reduce-overhead", fullgraph=True)

        print(f"{lmax},", benchmark(tp, input1, input2))
        
def run_e3nn_torch2_export(lmax):
    import torch
    import e3nn as e3nn_torch

    import torch._inductor.config as config
    config.cpp_wrapper = True   
    # torch.set_num_threads(1) # note this actually makes it faster
    with torch.no_grad():
        irreps1 = e3nn_torch.o3.Irreps.spherical_harmonics(lmax)
        irreps2 = (channels * irreps1).sort().irreps.simplify()

        input1 = irreps1.randn(-1) * 0
        input2 = irreps2.randn(-1) * 0
        tp = e3nn_torch.o3.experimental.FullTensorProductv2(irreps1, irreps2)
        so_path = torch._export.aot_compile(
            tp,
            (input1, input2,),
            options={"aot_inductor.output_path": os.path.join(os.getcwd(), f"extra/build/model_{lmax}.so")}
        )
        tp_compiled = torch._export.aot_load(os.path.join(os.getcwd(), f"extra/build/model_{lmax}.so"), device='cpu')
        print(f"{lmax},", benchmark(tp_compiled, input1, input2))

def run_e3nn_torch2_ipex(lmax):
    import torch
    import e3nn as e3nn_torch
    import intel_extension_for_pytorch as ipex
    # torch.set_num_threads(1) # note this actually makes it faster
    with torch.no_grad():
        irreps1 = e3nn_torch.o3.Irreps.spherical_harmonics(lmax)
        irreps2 = (channels * irreps1).sort().irreps.simplify()

        input1 = irreps1.randn(-1) * 0
        input2 = irreps2.randn(-1) * 0
        tp = e3nn_torch.o3.experimental.FullTensorProductv2(irreps1, irreps2)
        tp.eval()
        tp = ipex.optimize(tp, weights_prepack=False)
        tp = torch.compile(tp, backend='ipex')

        print(f"{lmax},", benchmark(tp, input1, input2))

def main():
    for run_fn, name in [
        (run_e3nn_jax, "e3nn-jax"),
        (run_e3nn_torch, "e3nn-torch"),
        (run_e3nn_torch2, "e3nn-torch2"),
        (run_e3nn_torch2_export, "e3nn-torch2-export"),
        # (run_e3nn_torch2_ipex, "e3nn-torch2-ipex"),
    ]:
        print(name)
        for lmax in range(1, L):
            run_fn(lmax)

if __name__ == "__main__":
    main()
