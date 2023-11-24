import os
import sys

import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))

import models

model = models.RotationalVariationalAutoencoderPower(z_dim=3)
input = model.example_input_array

model(input)
model.encode(input)

compiled_encode = torch.compile(model.encode)
compiled_encode(input)
