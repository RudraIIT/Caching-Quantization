import torch
import sys
import os

def main():
    # Get the path to the activation file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'activations/activations_t0_b1.pt'
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load the activations
    print(f"Loading activations from {file_path}...")
    activations = torch.load(file_path)
    
    # Print summary
    print(f"\nNumber of recorded activations: {len(activations)}")
    print("\nMemory usage:")
    total_elements = sum(tensor.numel() for tensor in activations.values())
    memory_mb = total_elements * 4 / (1024 * 1024)  # assuming float32 (4 bytes)
    print(f"Total elements: {total_elements:,}")
    print(f"Estimated memory: {memory_mb:.2f} MB")
    
    # Print sample of activations
    print("\nSample of activation layers and shapes:")
    for i, (name, tensor) in enumerate(activations.items()):
        print(f"{i+1}. {name}: {tensor.shape}")
        if i >= 9:
            remaining = len(activations) - 10
            if remaining > 0:
                print(f"... and {remaining} more layers")
            break

if __name__ == "__main__":
    main()