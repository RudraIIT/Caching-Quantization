#!/usr/bin/env python3
"""
Benchmark DDIM: Normal generation vs generation with caching
Generates 1000 identical images using both methods and measures throughput
"""
import os
import torch
import numpy as np
import time
import yaml
import argparse
import logging
import json
from tqdm import tqdm
from PIL import Image

# Import required modules
from models.diffusion import Model
from caching.deep_cache_wrapper import DeepCacheModel
from caching.cka_cache_wrapper import CKACacheModel
from functions.denoising import generalized_steps
from datasets import inverse_data_transform


def setup_logging():
    """Set up basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def dict2namespace(config_dict):
    """Convert a dictionary to a namespace"""
    namespace = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_timesteps):
    """Get beta schedule for diffusion process"""
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps)
    elif beta_schedule == "quad":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_timesteps)**2
    else:
        raise ValueError(f"Unknown beta schedule: {beta_schedule}")
    return torch.from_numpy(betas).float()


class BenchmarkRunner:
    def __init__(self, config_path="configs/celeba.yml", ckpt_path="ckpt.pth", num_timesteps=50):
        # Load configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Load config
        with open(config_path, "r") as f:
            self.config = dict2namespace(yaml.safe_load(f))
        self.config.device = self.device
        
        # Set parameters
        self.timesteps = num_timesteps
        self.checkpoint_path = ckpt_path
        self.sample_type = "generalized"
        self.skip_type = "uniform"
        self.eta = 0.0
        
        # Load model just to check its working
        try:
            _ = self.load_model()
            logging.info("Model loading test successful")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
        
        # Initialize beta schedule
        self._setup_betas()
        
        logging.info(f"Benchmark initialized with {num_timesteps} timesteps")
        logging.info(f"Image size: {self.config.data.image_size}x{self.config.data.image_size}")
    
    def _setup_betas(self):
        """Set up beta schedule for the diffusion process"""
        betas = get_beta_schedule(
            self.config.diffusion.beta_schedule,
            self.config.diffusion.beta_start, 
            self.config.diffusion.beta_end,
            self.config.diffusion.num_diffusion_timesteps
        )
        self.betas = betas.to(self.device)
        self.num_diffusion_timesteps = len(self.betas)
        
        # Set up timestep sequence for sampling
        skip = self.num_diffusion_timesteps // self.timesteps
        self.seq = range(0, self.num_diffusion_timesteps, skip)
        
    def _create_caching_config(self):
        """Create a deep copy of config with caching parameters added"""
        # Create a deep copy of the config structure
        config_dict = {}
        
        # Copy data section
        data_dict = {}
        for key in dir(self.config.data):
            if not key.startswith('_'):
                data_dict[key] = getattr(self.config.data, key)
        config_dict['data'] = data_dict
        
        # Copy model section
        model_dict = {}
        for key in dir(self.config.model):
            if not key.startswith('_'):
                model_dict[key] = getattr(self.config.model, key)
        config_dict['model'] = model_dict
        
        # Copy diffusion section
        diffusion_dict = {}
        for key in dir(self.config.diffusion):
            if not key.startswith('_'):
                diffusion_dict[key] = getattr(self.config.diffusion, key)
        config_dict['diffusion'] = diffusion_dict
        
        # Add caching section
        cache_steps = []
        # Cache every 50th step
        for i in range(0, self.num_diffusion_timesteps, 50):
            cache_steps.append(i)
        config_dict['caching'] = {
            'schedule': cache_steps,
            'max_size': 100,
            'use_input_hashing': True
        }
        
        # Convert to namespace
        return dict2namespace(config_dict)
    
    def load_model(self):
        """Load model from checkpoint"""
        model = Model(self.config)
        
        # Load checkpoint
        states = torch.load(self.checkpoint_path, map_location=self.device)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        
        # Apply EMA if available
        if self.config.model.ema and len(states) > 1:
            from models.ema import EMAHelper
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        
        model.eval()
        return model
    
    def generate_fixed_noise(self, batch_size, seed=1234):
        """Generate fixed noise for reproducible generation"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        return torch.randn(
            batch_size,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size,
            device=self.device
        )
    
    def generate_without_caching(self, n_images=1000, batch_size=10, output_dir="nocache_images"):
        """Generate images without caching (reload model for each batch)"""
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Generating {n_images} images WITHOUT caching (batch size: {batch_size})")
        total_batches = (n_images + batch_size - 1) // batch_size
        
        start_time = time.time()
        images_generated = 0
        
        for batch_idx in tqdm(range(total_batches), desc="Generating batches without caching"):
            # Set reproducible seed for this batch
            batch_seed = 1234 + batch_idx
            
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, n_images - batch_idx * batch_size)
            if current_batch_size <= 0:
                break
            
            # Generate noise for this batch
            batch_x = self.generate_fixed_noise(current_batch_size, seed=batch_seed)
            
            # Load a fresh model for each batch (no caching)
            model = self.load_model()
            
            # Generate samples
            with torch.no_grad():
                samples = generalized_steps(batch_x, self.seq, model, self.betas, eta=self.eta)
                samples = samples[0][-1]  # Get final samples
                
                # Convert to images
                samples = inverse_data_transform(self.config, samples)
                
                # Save each image
                for i in range(samples.shape[0]):
                    img_idx = batch_idx * batch_size + i
                    if img_idx >= n_images:
                        break
                    
                    # Convert to numpy and save
                    sample_np = samples[i].cpu().numpy().transpose(1, 2, 0)
                    sample_np = np.clip(sample_np, 0, 1)
                    sample_np = (sample_np * 255).astype(np.uint8)
                    
                    img = Image.fromarray(sample_np)
                    img.save(os.path.join(output_dir, f"img_{img_idx:04d}.png"))
                    images_generated += 1
            
            # Clean up to ensure no caching effect
            del model
            torch.cuda.empty_cache()
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = images_generated / total_time
        
        logging.info(f"Generated {images_generated} images in {total_time:.2f}s without caching")
        logging.info(f"Throughput without caching: {throughput:.2f} images/second")
        
        return total_time, throughput, images_generated
    
    def generate_with_caching(self, n_images=1000, batch_size=10, output_dir="cached_images"):
        """Generate images with caching using DeepCacheModel"""
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Generating {n_images} images WITH caching (batch size: {batch_size})")
        total_batches = (n_images + batch_size - 1) // batch_size
        
        # Create a caching version of the config
        caching_config = self._create_caching_config()
        
        # Load a DeepCacheModel model instead of the normal model
        caching_model = CKACacheModel(caching_config)
        
        # Load checkpoint
        states = torch.load(self.checkpoint_path, map_location=self.device)
        caching_model = caching_model.to(self.device)
        caching_model = torch.nn.DataParallel(caching_model)
        caching_model.load_state_dict(states[0], strict=True)
        
        # Apply EMA if available
        if self.config.model.ema and len(states) > 1:
            from models.ema import EMAHelper
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(caching_model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(caching_model)
            
        caching_model.eval()
        
        # Store the model for later stats access
        self.model_with_caching = caching_model
        
        start_time = time.time()
        images_generated = 0
        
        for batch_idx in tqdm(range(total_batches), desc="Generating batches with caching"):
            # Set reproducible seed for this batch
            batch_seed = 1234 + batch_idx
            
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, n_images - batch_idx * batch_size)
            if current_batch_size <= 0:
                break
            
            # Generate noise for this batch (must be identical to no-cache version)
            batch_x = self.generate_fixed_noise(current_batch_size, seed=batch_seed)
            
            # Generate samples using the cached model
            with torch.no_grad():
                samples = generalized_steps(batch_x, self.seq, caching_model, self.betas, eta=self.eta)
                samples = samples[0][-1]  # Get final samples
                
                # Convert to images
                samples = inverse_data_transform(self.config, samples)
                
                # Save each image
                for i in range(samples.shape[0]):
                    img_idx = batch_idx * batch_size + i
                    if img_idx >= n_images:
                        break
                    
                    # Convert to numpy and save
                    sample_np = samples[i].cpu().numpy().transpose(1, 2, 0)
                    sample_np = np.clip(sample_np, 0, 1)
                    sample_np = (sample_np * 255).astype(np.uint8)
                    
                    img = Image.fromarray(sample_np)
                    img.save(os.path.join(output_dir, f"img_{img_idx:04d}.png"))
                    images_generated += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = images_generated / total_time
        
        logging.info(f"Generated {images_generated} images in {total_time:.2f}s with caching")
        logging.info(f"Throughput with caching: {throughput:.2f} images/second")
        
        return total_time, throughput, images_generated
    
    def verify_image_identity(self, dir1, dir2, n_images):
        """Verify that images from both methods are identical"""
        logging.info(f"Verifying image identity between {dir1} and {dir2}")
        
        identical_count = 0
        different_count = 0
        total_diff = 0
        max_diff = 0
        
        for i in tqdm(range(n_images), desc="Comparing images"):
            img1_path = os.path.join(dir1, f"img_{i:04d}.png")
            img2_path = os.path.join(dir2, f"img_{i:04d}.png")
            
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                logging.warning(f"Missing image at index {i}")
                different_count += 1
                continue
            
            img1 = np.array(Image.open(img1_path))
            img2 = np.array(Image.open(img2_path))
            
            # Calculate difference
            pixel_diff = np.abs(img1.astype(float) - img2.astype(float)).mean()
            total_diff += pixel_diff
            max_diff = max(max_diff, pixel_diff)
            
            if pixel_diff < 1.0:  # Allow very small differences due to floating point
                identical_count += 1
            else:
                different_count += 1
        
        avg_diff = total_diff / n_images if n_images > 0 else 0
        identical_percent = (identical_count / n_images * 100) if n_images > 0 else 0
        
        logging.info(f"Images compared: {n_images}")
        logging.info(f"Identical images: {identical_count} ({identical_percent:.2f}%)")
        logging.info(f"Different images: {different_count}")
        logging.info(f"Average pixel difference: {avg_diff:.6f}")
        logging.info(f"Maximum pixel difference: {max_diff:.6f}")
        
        return {
            "identical_count": identical_count,
            "different_count": different_count,
            "identical_percent": identical_percent,
            "avg_diff": avg_diff,
            "max_diff": max_diff
        }


def main():
    # Set up logging
    setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark DDIM image generation with and without caching")
    parser.add_argument("--n_images", type=int, default=50000, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--timesteps", type=int, default=100, help="Number of timesteps for sampling")
    parser.add_argument("--config", type=str, default="configs/celeba.yml", help="Config file")
    parser.add_argument("--ckpt", type=str, default="ckpt.pth", help="Checkpoint file")
    parser.add_argument("--nocache_dir", type=str, default="nocache_images", help="Output directory for non-cached images")
    parser.add_argument("--cache_dir", type=str, default="cached_images_cka", help="Output directory for cached images")
    args = parser.parse_args()
    
    # Print benchmark configuration
    logging.info("="*70)
    logging.info("DDIM CACHING BENCHMARK")
    logging.info("="*70)
    logging.info(f"Number of images: {args.n_images}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Timesteps: {args.timesteps}")
    logging.info(f"Config file: {args.config}")
    logging.info(f"Checkpoint: {args.ckpt}")
    logging.info("="*70)
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        config_path=args.config,
        ckpt_path=args.ckpt,
        num_timesteps=args.timesteps
    )

    # Generate images with caching
    time_with_cache, throughput_with_cache, n_gen2 = runner.generate_with_caching(
        n_images=args.n_images,
        batch_size=args.batch_size,
        output_dir=args.cache_dir
    )

    # Generate images without caching
    # time_no_cache, throughput_no_cache, n_gen1 = runner.generate_without_caching(
    #     n_images=args.n_images,
    #     batch_size=args.batch_size,
    #     output_dir=args.nocache_dir
    # )
    
    # # Verify images are identical
    # identity_results = runner.verify_image_identity(
    #     args.nocache_dir, args.cache_dir, min(n_gen1, n_gen2)
    # )
    
    # # Calculate performance metrics
    # speedup = throughput_with_cache / throughput_no_cache
    # time_saved = time_no_cache - time_with_cache
    # time_saved_percent = (time_saved / time_no_cache) * 100
    
    # Prepare results
    results = {
        "configuration": {
            "n_images": args.n_images,
            "batch_size": args.batch_size,
            "timesteps": args.timesteps,
            "config_file": args.config,
            "checkpoint": args.ckpt,
            "device": str(runner.device)
        },
        # "without_caching": {
        #     "time": time_no_cache,
        #     "throughput": throughput_no_cache,
        #     "images_generated": n_gen1
        # },
        "with_caching": {
            "time": time_with_cache,
            "throughput": throughput_with_cache,
            "images_generated": n_gen2
        },
        "performance": {
            "speedup": speedup,
            "time_saved": time_saved,
            "time_saved_percent": time_saved_percent
        },
        "image_comparison": identity_results
    }
    
    # Get cache statistics if available
    try:
        cache_stats = runner.model_with_caching.module.get_cache_stats() if hasattr(runner, 'model_with_caching') else {'cache_size': 'N/A'}
    except:
        cache_stats = {'cache_size': 'unavailable', 'max_cache_size': 'unavailable'}
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"Images generated: {args.n_images}")
    print(f"Images identical: {identity_results['identical_count']}/{min(n_gen1, n_gen2)} " +
          f"({identity_results['identical_percent']:.2f}%)")
    print(f"Average pixel difference: {identity_results['avg_diff']:.6f}")
    # print()
    # print("WITHOUT CACHING:")
    # print(f"  Time: {time_no_cache:.2f} seconds")
    # print(f"  Throughput: {throughput_no_cache:.2f} images/second")
    print()
    print("WITH CACHING:")
    print(f"  Time: {time_with_cache:.2f} seconds")
    print(f"  Throughput: {throughput_with_cache:.2f} images/second")
    print(f"  Cache entries: {cache_stats.get('cache_size', 'N/A')}")
    print(f"  Max cache size: {cache_stats.get('max_cache_size', 'N/A')}")
    print()
    print("PERFORMANCE IMPROVEMENT:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.2f} seconds ({time_saved_percent:.1f}%)")
    print("="*70)
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Results saved to benchmark_results.json")


if __name__ == "__main__":
    main()