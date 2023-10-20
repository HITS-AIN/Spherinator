import os
import sys

import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))

import models

CHECKPOINT_PATH = "/home/doserbd/ain-space/local/shapes-power/spherinator/w5z7hffp/checkpoints/epoch=60-train_loss=19.32.ckpt"

model = models.RotationalVariationalAutoencoderPower(z_dim=3)
model.load_state_dict(torch.load(CHECKPOINT_PATH)["state_dict"])
model.eval()

# Test the model with a dummy input
model(model.example_input_array)

# Convert the model to TorchScript
script = model.to_torchscript()

# Save the model
script.save("model.pt")
