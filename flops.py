import torch
import yaml
import time
from torch.profiler import profile, ProfilerActivity
from models.diffusion import Model
from caching.cka_cache_wrapper import CKACacheModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def dict_to_object(d):
    if isinstance(d, dict):
        class Config: pass
        config = Config()
        for key, value in d.items():
            setattr(config, key, dict_to_object(value))
        return config
    elif isinstance(d, list):
        return [dict_to_object(item) for item in d]
    else:
        return d
    
with open("configs/celeba.yml", "r") as f:
    config_dict = yaml.safe_load(f)
config = dict_to_object(config_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

baseline_model = Model(config).to(device).eval()
caching_model = CKACacheModel(config).to(device).eval()


x = torch.randn(1, 3, 64, 64, device=device)
t = torch.zeros(1, dtype=torch.long, device=device)

def measure_flops(model, x, t):
    model.eval()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = model(x, t)
    total_flops = sum([e.flops for e in prof.key_averages() if e.flops is not None])
    return total_flops

baseline_flops = measure_flops(baseline_model, x, t)
print(f"Baseline model FLOPs: {baseline_flops / 1e9:.2f} GFLOPs")


print("\nWarming up caching model ...")
with torch.no_grad():
    for step in range(100):
        t_step = torch.tensor([step], device=device)
        _ = caching_model(x, t_step)

print("\nMeasuring per-timestep FLOPs ...")
timesteps = list(range(100))
cache_flops_total = 0

for step in timesteps:
    t_step = torch.tensor([step], device=device)
    flops = measure_flops(caching_model, x, t_step)
    cache_flops_total += flops

cache_avg_flops = cache_flops_total / len(timesteps)

print(f"Caching Model Avg FLOPs (over 100 steps): {cache_avg_flops/1e9:.2f} GFLOPs")
print(f"→ FLOPs reduction: {100*(1 - cache_avg_flops/baseline_flops):.2f}%")

def measure_runtime(model, steps=100):
    start = time.time()
    with torch.no_grad():
        for i in range(steps):
            t_step = torch.tensor([i % 1000], device=device)
            _ = model(x, t_step)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()
    return (end - start) / steps

avg_time_baseline = measure_runtime(baseline_model)
avg_time_cache = measure_runtime(caching_model)

print("\n--- Runtime ---")
print(f"Baseline avg forward: {avg_time_baseline*1000:.2f} ms")
print(f"Caching avg forward:  {avg_time_cache*1000:.2f} ms")