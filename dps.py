import torch
import os
import time
import networkx as nx

def load_activation_for_timestep(save_dir, timestep, name="unet", device="cuda"):
    path = os.path.join(save_dir, f'activations_t{timestep}_b1.pt')
    data = torch.load(path, map_location=device)
    if isinstance(data, dict):
        data = torch.cat([v.flatten() for v in data.values()])
    return data.to(device)

@torch.no_grad()
def compute_D_cache_chunked(save_dir, timesteps, name="unet",
                               device="cuda", chunk_size=16384):
    N = len(timesteps)
    D = torch.zeros((N, N), dtype=torch.float32, device="cpu")

    def load_centered(t):
        x = load_activation_for_timestep(save_dir, t, name, device)
        x = x.view(x.shape[0], -1).to(device, dtype=torch.float32)
        x = x - x.mean(0, keepdim=True)
        return x

    # Precompute Frobenius norms
    norms = []
    for t in timesteps:
        x = load_centered(t)
        norm = torch.norm(x.T @ x, p='fro')
        norms.append(norm)
        del x
        torch.cuda.empty_cache()

    for i in range(N):
        xi = load_centered(timesteps[i])
        for j in range(i, N):
            xj = load_centered(timesteps[j])

            # Compute numerator chunked
            num = 0.0
            xi_chunks = xi.split(chunk_size, dim=1)
            xj_chunks = xj.split(chunk_size, dim=1)
            for xi_c, xj_c in zip(xi_chunks, xj_chunks):
                dot = xi_c.T @ xj_c
                num += (dot ** 2).sum().item()
                del dot
                torch.cuda.empty_cache()

            D[i, j] = num / (norms[i] * norms[j] + 1e-12)
            D[j, i] = D[i, j]

            del xj, xj_chunks
            torch.cuda.empty_cache()

        del xi
        torch.cuda.empty_cache()

    return D


def DPS_schedule(save_dir, timesteps, device="cuda", name="unet",
                 threshold=0.98, max_group_size=None):
    D_cache = compute_D_cache_chunked(save_dir, timesteps, device=device)
    T = len(timesteps)
    A = (D_cache >= threshold)

    G = nx.Graph()
    G.add_nodes_from(range(T))
    for i in range(T):
        for j in range(i + 1, T):
            if A[i, j]:
                G.add_edge(i, j)

    components = list(nx.connected_components(G))
    final_groups = []

    for comp in components:
        comp = list(comp)
        remaining = set(comp)
        while remaining:
            # start new clique
            i = remaining.pop()
            clique = [i]
            to_remove = set()
            for j in remaining:
                if all(A[j, k] for k in clique):
                    clique.append(j)
                    to_remove.add(j)
                    # optional: stop if group too large
                    if max_group_size and len(clique) >= max_group_size:
                        break
            remaining -= to_remove
            final_groups.append([timesteps[idx] for idx in clique])

    print("✅ Strict grouping done")
    return final_groups

def cached_timesteps(final_groups, num_samples=None):
    cached = []
    
    for group in final_groups:
        group_sorted = sorted(group)
        group_size = len(group_sorted)
        
        if group_size == 0:
            continue
        
        if num_samples is not None:
            n_samples = min(num_samples, group_size)
        else:
            n_samples = max(1, group_size // 3)
        
        if n_samples == 1:
            sampled_indices = [0]
        else:
            sampled_indices = [
                int(i * (group_size - 1) / (n_samples - 1)) 
                for i in range(n_samples)
            ]
        
        sampled_timesteps = [group_sorted[idx] for idx in sampled_indices]
        
        cached.extend(sampled_timesteps)
    
    return sorted(cached)

def create_config_and_save(cached_timesteps_list):
    config = {
        "cached_timesteps": cached_timesteps_list
    }
    with open("caching_config.json", "w") as f:
        import json
        json.dump(config, f, indent=4)
    print("Configuration saved to caching_config.json")

if __name__ == "__main__":
    timesteps = list(range(0, 100, 1))
    save_dir = "./activations"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    schedule = DPS_schedule(save_dir, timesteps, name="unet", max_group_size=None, device=device)
    print("Optimal caching schedule (groups):", schedule)
    print("Time taken:", time.time() - t0)
    
    cached = cached_timesteps(schedule)
    print("Cached timesteps:", cached)
    print(f"Total cached timesteps: {len(cached)}")

