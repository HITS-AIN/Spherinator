import torch
from models import RotationalSphericalProjectingAutoencoder

model = RotationalSphericalProjectingAutoencoder()
filepath = "RotationalSphericalProjectingAutoencoder.onnx"
input_sample = (torch.randn((424, 424)), float(0))
model.to_onnx(filepath, input_sample, export_params=True,
              input_names=["input", "rotation"],
              output_names=["output", "coordinates"])
