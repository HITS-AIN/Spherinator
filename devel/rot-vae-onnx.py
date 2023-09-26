import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))

import models

model = models.RotationalVariationalAutoencoder()

# Test the model with a dummy input
model(model.example_input_array)

filepath = "RotationalVAE.onnx"
model.to_onnx(filepath, export_params=True, verbose=True)
