import matplotlib.pyplot as plt
import lovelyplots
plt.style.use(["ipynb", "use_mathtext"])

from benchmark_python import channels, trials

with open("benchmark.txt", "r") as f:
    lines = [line.strip() for line in f.readlines()]

data = {}
for line in lines:
    if not line[0].isnumeric():
        name = line
        data[name] = {}
        data[name]["lmax"] = []
        data[name]["time"] = []
    else:
        lmax, time = line.split(", ")
        data[name]["lmax"].append(int(lmax)) 
        data[name]["time"].append(float(time) / trials * 1000) 

fig = plt.figure(figsize=(6,3))
for name in data.keys():
    plt.plot(data[name]["lmax"], data[name]["time"], label=name)
plt.yscale("log")
plt.ylabel("time [ms]")
plt.xlabel("L")
plt.legend(fontsize=8)
plt.grid()
plt.title("Single-threaded CPU Tensor Product Performance\n" + rf"({channels}x0e + {channels}x1o ...) $\otimes$ (1x0e + 1x1o ...)")
plt.savefig("extra/benchmark_versions.png", dpi=200, bbox_inches="tight", transparent=False)
plt.close(fig)

fig = plt.figure(figsize=(6,3))
for name in ["e3nn.c v3", "e3nn-jax", "e3nn-torch", "e3nn-torch2"]: # , "e3nn-torch2-ipex"]:
    label = "e3nn.c" if "e3nn.c" in name else name
    plt.plot(data[name]["lmax"], data[name]["time"], label=label)
plt.yscale("log")
plt.ylabel("time [ms]")
plt.xlabel("L")
plt.legend(fontsize=8)
plt.grid()
plt.title("Single-threaded CPU Tensor Product Performance\n" + rf"({channels}x0e + {channels}x1o ...) $\otimes$ (1x0e + 1x1o ...)")
plt.savefig("extra/benchmark.png", dpi=200, bbox_inches="tight", transparent=False)
plt.close(fig)