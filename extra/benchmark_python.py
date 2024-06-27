import time

import jax
import e3nn_jax

import torch
import e3nn as e3nn_torch

channels = 128
warmup = 10
trials = 100
L = 8

def main():
    print("e3nn-jax")
    for lmax in range(1, L):
        irreps1 = e3nn_jax.Irreps.spherical_harmonics(lmax)
        irreps2 = irreps1 * channels

        input1 = e3nn_jax.zeros(irreps1, ())
        input2 = e3nn_jax.zeros(irreps2, ())

        tp = jax.jit(e3nn_jax.tensor_product)

        # perform once to jit
        for _ in range(warmup):
            output = tp(input1, input2)

        start = time.process_time()

        for trial in range(trials):
            output = tp(input1, input2)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)

        print(f"{lmax},", time.process_time() - start)

    print("e3nn-torch")
    torch.set_num_threads(1) # note this actually makes it faster
    with torch.no_grad():
        for lmax in range(1, L):
            irreps1 = e3nn_torch.o3.Irreps.spherical_harmonics(lmax)
            irreps2 = (channels * irreps1).sort().irreps.simplify()


            input1 = irreps1.randn(-1) * 0
            input2 = irreps2.randn(-1) * 0

            tp = e3nn_torch.o3.FullTensorProduct(irreps1, irreps2)
            tp = e3nn_torch.util.jit.compile(tp)

            # let jit
            # perform once to jit
            for _ in range(warmup):
                output = tp(input1, input2)

            start = time.process_time()

            for _ in range(trials):
                output = tp(input1, input2)

            print(f"{lmax},", time.process_time() - start)

    print("e3nn-torch2")
    torch.set_num_threads(1) # note this actually makes it faster
    with torch.no_grad():
        for lmax in range(1, L):
            irreps1 = e3nn_torch.o3.Irreps.spherical_harmonics(lmax)
            irreps2 = (channels * irreps1).sort().irreps.simplify()

            input1 = irreps1.randn(-1) * 0
            input2 = irreps2.randn(-1) * 0
            tp = e3nn_torch.o3.experimental.FullTensorProductv2(irreps1, irreps2)
            tp = torch.compile(tp, fullgraph=True)

            # perform once to jit
            for _ in range(warmup):
                output = tp(input1, input2)

            start = time.process_time()
            for _ in range(trials):
                output = tp(input1, input2)

            print(f"{lmax},", time.process_time() - start)


if __name__ == "__main__":
    main()
