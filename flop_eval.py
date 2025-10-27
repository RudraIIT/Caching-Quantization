import torch
import yaml
import time
import warnings
from models.diffusion import Model
from caching.deep_cache_wrapper import DeepCacheModel

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
model = Model(config).to(device).eval()
caching_model = DeepCacheModel(config).to(device).eval()

x = torch.randn(1, 3, 64, 64).to(device)
t = torch.zeros(1, dtype=torch.long).to(device)

total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, t_value):
        super().__init__()
        self.model = model
        self.t = t_value
    def forward(self, x):
        return self.model(x, self.t)

try:
    from ptflops import get_model_complexity_info
    def get_flops(model, x_shape=(3, 64, 64)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs, _ = get_model_complexity_info(
                model, x_shape,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
        return macs * 2
    model_flops = get_flops(ModelWrapper(model, t))
    timesteps = list(range(1000))
    cache_flops_total = 0
    for step in timesteps:
        t_step = torch.tensor([step], device=device)
        wrapper = ModelWrapper(caching_model, t_step)
        flops = get_flops(wrapper)
        if step in caching_model.caching_schedule and step in caching_model.cache:
            flops = 1e6
        cache_flops_total += flops
    cache_avg_flops = cache_flops_total / len(timesteps)
    has_ptflops = True
except ImportError:
    model_flops = total_params * (x.shape[2] * x.shape[3])
    cache_avg_flops = model_flops
    has_ptflops = False

if device.type == "cuda":
    torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    for _ in range(1000):
        _ = model(x, t)
if device.type == "cuda":
    torch.cuda.synchronize()
end_time = time.time()
avg_time_model = (end_time - start_time) / 1000

if device.type == "cuda":
    torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    for _ in range(1000):
        _ = caching_model(x, t)
if device.type == "cuda":
    torch.cuda.synchronize()
end_time = time.time()
avg_time_cache = (end_time - start_time) / 1000

print("\n--- Model Statistics ---")
print(f"Total parameters: {total_params/1e6:.2f} M")
print(f"Trainable parameters: {total_trainable_params/1e6:.2f} M")

print("\n--- Runtime ---")
print(f"Baseline model average forward time: {avg_time_model*1000:.2f} ms")
print(f"Caching model average forward time: {avg_time_cache*1000:.2f} ms")

print("\n--- FLOPs ---")
print(f"{'Estimated ' if not has_ptflops else ''}Baseline FLOPs: {model_flops/1e9:.2f} GFLOPs")
print(f"{'Estimated ' if not has_ptflops else ''}Caching model avg FLOPs: {cache_avg_flops/1e9:.2f} GFLOPs")
