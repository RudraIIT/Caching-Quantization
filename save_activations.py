import torch
import yaml
import argparse
import os
import sys
from types import SimpleNamespace
from tqdm import tqdm

# Import Model from the diffusion module
from models.diffusion import Model
from collections import OrderedDict

class ActivationSaver:
    def __init__(self, model, save_dir, save_per_timestep=False):
        self.model = model
        self.save_dir = save_dir
        self.save_per_timestep = save_per_timestep
        self.activations = {}
        self.hooks = []
        self.current_timestep = None
        
        # Create directory for saving activations
        os.makedirs(save_dir, exist_ok=True)
        
        # Register hooks for each module in the model
        self._register_hooks()
        
    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                # Store the activation
                self.activations[name] = output.detach().cpu()
            return hook
        
        # Register hooks for each named module
        for name, module in self.model.named_modules():
            if name and not any(n in name for n in ["up", "down", "mid", "norm_out", "conv_out", "temb"]):
                continue  # Skip nested children to avoid redundancy
                
            if name:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
    
    def set_timestep(self, timestep):
        """Set the current timestep before running the model"""
        self.current_timestep = timestep
    
    def save_activations(self):
        """Save the activations to disk"""
        if self.current_timestep is None:
            raise ValueError("Timestep must be set before saving activations")
            
        batch_size = list(self.activations.values())[0].shape[0] if self.activations else 1
            
        if self.save_per_timestep:
            # Save activations for this specific timestep
            save_path = os.path.join(self.save_dir, f'activations_t{self.current_timestep}_b{batch_size}.pt')
            torch.save(self.activations, save_path)
            print(f"Saved activations for timestep {self.current_timestep} to {save_path}")
        else:
            # Append to a global activation dictionary
            global_acts_path = os.path.join(self.save_dir, 'all_activations.pt')
            
            # Check if file exists
            if os.path.exists(global_acts_path):
                try:
                    global_acts = torch.load(global_acts_path)
                except Exception:
                    global_acts = {}
            else:
                global_acts = {}
            
            # Store activations for this timestep
            global_acts[f'timestep_{self.current_timestep}'] = self.activations
            
            # Save updated dictionary
            torch.save(global_acts, global_acts_path)
            print(f"Appended activations for timestep {self.current_timestep} to {global_acts_path}")
            
        # Clear activations for next run
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all hooks from the model"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='celeba.yml', help='Configuration file to use')
    parser.add_argument('--ckpt', type=str, default='ckpt.pth', help='Checkpoint file path')
    parser.add_argument('--timesteps', type=str, default=','.join([str(i) for i in range(100)]), help='Comma-separated list of timesteps to run')
    parser.add_argument('--save_dir', type=str, default='activations', help='Directory to save activations')
    parser.add_argument('--save_per_timestep', action='store_true', 
                        help='Save activations per timestep (separate files) instead of all together')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for generation')
    args = parser.parse_args()

    print("=" * 80)
    print(f"Recording activations using configuration: {args.config}")
    print("=" * 80)
    
    # Load config file
    config_path = os.path.join('configs', args.config)
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to SimpleNamespace for attribute access
    config = SimpleNamespace()
    for k, v in config_dict.items():
        if isinstance(v, dict):
            setattr(config, k, SimpleNamespace(**v))
        else:
            setattr(config, k, v)
    
    # Create activation save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = Model(config)
    
    # Load checkpoint
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file {ckpt_path} not found")
        sys.exit(1)
        
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, list):
        # If checkpoint is a list (e.g., containing optimizer state), extract model state dict
        print("Checkpoint format is a list, extracting model state dict...")
        state_dict = checkpoint[0]
    elif isinstance(checkpoint, dict):
        # If checkpoint is already a state dict
        state_dict = checkpoint
    else:
        print(f"Error: Unrecognized checkpoint format: {type(checkpoint)}")
        sys.exit(1)
    
    # Remove 'module.' prefix if present in state_dict keys
    if all(k.startswith('module.') for k in state_dict.keys()):
        print("Removing 'module.' prefix from state dict keys...")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        state_dict = new_state_dict
    
    # Load state dict into model
    try:
        model.load_state_dict(state_dict)
        print("Successfully loaded checkpoint")
    except Exception as e:
        print(f"Warning: Could not load checkpoint exactly: {str(e)}")
        # Try with strict=False
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded checkpoint with strict=False")
        except Exception as e2:
            print(f"Error: Could not load checkpoint: {str(e2)}")
            sys.exit(1)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create activation saver
    activation_saver = ActivationSaver(model, args.save_dir, args.save_per_timestep)
    
    # Parse timesteps
    try:
        timesteps = [int(t) for t in args.timesteps.split(',')]
        print(f"Processing timesteps: {timesteps}")
    except ValueError:
        print(f"Error: Invalid timestep format. Please use comma-separated integers.")
        sys.exit(1)
    
    # Generate input data
    batch_size = args.batch_size
    channels = config.model.in_channels
    image_size = config.data.image_size
    
    print(f"Creating {batch_size} random images of size {image_size}x{image_size} with {channels} channels...")
    x = torch.randn(batch_size, channels, image_size, image_size, device=device)
    
    # Record all activations for each timestep
    print("\nRunning model forward pass and recording activations...")
    
    with torch.no_grad():
        for timestep in tqdm(timesteps, desc="Processing timesteps"):
            # Create tensor for current timestep
            t = torch.tensor([timestep], device=device)
            
            # Set current timestep in the activation saver
            activation_saver.set_timestep(timestep)
            
            # Run model forward pass
            output = model(x, t)
            
            # Save activations for this timestep
            activation_saver.save_activations()
            
            print(f"Processed timestep {timestep}, output shape: {output.shape}")
    
    # Clean up by removing hooks
    activation_saver.remove_hooks()
    
    # Report completion
    if args.save_per_timestep:
        print(f"\nAll activations have been saved to individual files in: {args.save_dir}")
        # List a few saved files
        saved_files = [f for f in os.listdir(args.save_dir) if f.startswith('activations_t')]
        if saved_files:
            print("Sample of saved files:")
            for i, file in enumerate(saved_files[:5]):
                print(f"  - {file}")
            if len(saved_files) > 5:
                print(f"  - ... ({len(saved_files) - 5} more files)")
    else:
        print(f"\nAll activations have been saved to: {os.path.join(args.save_dir, 'all_activations.pt')}")
        
        # Load and display information about saved activations
        try:
            saved_path = os.path.join(args.save_dir, 'all_activations.pt')
            if os.path.exists(saved_path):
                saved_activations = torch.load(saved_path)
                print(f"\nTimesteps recorded: {len(saved_activations.keys())}")
                
                # Show sample of activation information for first timestep
                first_ts = list(saved_activations.keys())[0]
                timestep_activations = saved_activations[first_ts]
                
                # Count total activations and memory usage
                total_params = 0
                for name, tensor in timestep_activations.items():
                    total_params += torch.numel(tensor)
                
                print(f"Total activations stored per timestep: {len(timestep_activations)}")
                print(f"Total parameters per timestep: {total_params:,}")
                print(f"Estimated memory per timestep: {total_params * 4 / (1024**2):.2f} MB (32-bit float)")
                
                # Show sample of activation names and shapes
                print("\nSample of recorded activations:")
                for i, (name, activation) in enumerate(timestep_activations.items()):
                    print(f"  - {name}: {activation.shape}")
                    if i >= 9:  # Show only first 10
                        remaining = len(timestep_activations) - 10
                        if remaining > 0:
                            print(f"  - ... ({remaining} more activations)")
                        break
        except Exception as e:
            print(f"Error loading saved activations: {str(e)}")
    
    print("\nActivation recording complete!")

if __name__ == "__main__":
    main()