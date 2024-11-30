from spherinator.models import (
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
    VariationalAutoencoder,
)


def test_forward():

    z_dim = 3
    encoder = ConvolutionalEncoder1D(input_dim=128, output_dim=256)
    decoder = ConvolutionalDecoder1D(input_dim=256, output_dim=128)
    model = VariationalAutoencoder(encoder=encoder, decoder=decoder, z_dim=z_dim)
    input = model.example_input_array

    (z_mean, z_var), (_, _), _, recon = model(input)

    batch_size = input.shape[0]
    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape
