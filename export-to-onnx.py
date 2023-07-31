import torch
from models import RotationalSphericalProjectingAutoencoder

model = RotationalSphericalProjectingAutoencoder()
filepath = "RotationalSphericalProjectingAutoencoder.onnx"
input_sample = torch.randn((64, 3, 424, 424))

output = model(input_sample)

model.to_onnx(filepath, input_sample, export_params=True, verbose=True)
