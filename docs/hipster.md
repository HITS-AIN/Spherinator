# Generate HiPS and catalog

The following command generates a HiPS representation and a catalog showing the real images located on the latent space using the trained model.

```bash
hipster --checkpoint <checkpoint-file>.ckpt
```

Call `hipster --help` for more information.

For visualization, a Python HTTP server can be started by executing `python3 -m http.server 8082` within the HiPSter output file.


## The HiPSter workflow

![](assets/HiPSter.drawio.svg)
