import torch

model = torch.nn.Sequential(
    *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
)
input = torch.rand(1024, device="cuda")
output = model(input)
output.sum().backward()

# exit cleanly if we are on a device that doesn't support torch.compile
if torch.cuda.get_device_capability() < (7, 0):
    print("Exiting because torch.compile is not supported on this device.")
    import sys

    sys.exit(0)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def eager_step():
    optimizer.step()


def compiled_step():
    optimizer.step()


compiled_step = torch.compile(compiled_step, fullgraph=False)

# Let's define a helpful benchmarking function:
import torch.utils.benchmark as benchmark


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


# Warmup runs to compile the function
for _ in range(5):
    compiled_step()

eager_runtime = benchmark_torch_function_in_microseconds(eager_step)
compiled_runtime = benchmark_torch_function_in_microseconds(compiled_step)

assert eager_runtime > compiled_runtime

print(f"eager runtime: {eager_runtime}us")
print(f"compiled runtime: {compiled_runtime}us")
