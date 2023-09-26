import models

model = models.rotational_variational_autoencoder()

# Test the model with a dummy input
model(model.example_input_array)

filepath = "RotationalVAE.onnx"
model.to_onnx(filepath, export_params=True, verbose=True)
