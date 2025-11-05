import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple tensor
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(x)
