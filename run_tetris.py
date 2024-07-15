import sys

import e3nn_jax as e3nn
import jax
import jraph
from flax import serialization
from jax import numpy as jnp

from train_tetris import Model

def main():
    if len(sys.argv) != 13:  # 4 coordinates * 3 values each + 1 for the program name
        print(f"usage: {sys.argv[0]} x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4")
        return
    
    labels = ["chiral 1", "chiral 2", "square", "line", "corner", "L", "T", "zigzag"];
    
    pos = []
    for i in range(4):
        x = float(sys.argv[3 * i + 1])
        y = float(sys.argv[3 * i + 2])
        z = float(sys.argv[3 * i + 3])
        pos.append([x, y, z])

    
    model = Model()

    pos = jnp.array(pos, dtype=jnp.float32) # zigzag
    senders, receivers = e3nn.radius_graph(pos, 1.1)
    graph = jraph.GraphsTuple(
        nodes=pos.reshape((4, 3)),
        edges=None,
        senders=senders,
        receivers=receivers,
        globals=None,
        n_node=jnp.array([len(pos)]),
        n_edge=jnp.array([len(senders)]),
    ) 

    params = model.init(jax.random.PRNGKey(0), graph)
    with open("tetris.mp", "rb") as f:
        params = serialization.from_bytes(params, f.read())

    logits = model.apply(params, graph)[0]
    for (label, logit) in zip(labels, logits):
        print(f"{label:<12}{float(logit):.5f}")


if __name__ == "__main__":
    main()