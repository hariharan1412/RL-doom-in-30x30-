import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from own_env import EYES, BossShooterEnv

# Load the model (define the EYES class first as in your original code)
model = EYES(img_height=30, img_width=30, num_actions=3)
model.load_state_dict(torch.load('own-model-doom-20k.pth'))
model.eval()

# Global variables to store feature maps
c1_features = None
c2_features = None

def hook_c1(module, input, output):
    global c1_features
    c1_features = output.detach().cpu()

def hook_c2(module, input, output):
    global c2_features
    c2_features = output.detach().cpu()

# Register hooks
model.c1.register_forward_hook(hook_c1)
model.c2.register_forward_hook(hook_c2)


# Create environment and get sample input
env = BossShooterEnv()
observation = env.reset()
sample_input = torch.from_numpy(observation).unsqueeze(0).float()  # Add batch dimension


# Forward pass through the model
with torch.no_grad():
    output = model(sample_input)
    print("Output shape:", output.shape)

def save_feature_maps(features, layer_name):
    # Convert to numpy and normalize
    features = features.numpy()
    batch_size, n_channels, h, w = features.shape
    
    # Create directory
    os.makedirs(f"feature_maps/{layer_name}", exist_ok=True)
    
    for i in range(n_channels):
        feature_map = features[0, i]
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        plt.figure(figsize=(w/5, h/5))
        plt.imshow(feature_map, cmap='viridis')
        plt.axis('off')
        plt.savefig(f"feature_maps/{layer_name}/channel_{i:02d}.png", 
                   bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

# Save both layers
save_feature_maps(c1_features, "c1")
save_feature_maps(c2_features, "c2")