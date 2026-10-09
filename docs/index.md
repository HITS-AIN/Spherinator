# Spherinator

Spherinator is a Python package providing variational autoencoders (VAE) to reduce generic data to a
spherical latent space. It is designed to be used with
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

```{figure} assets/P404_f2.png
---
name: fig:latent_space
width: 300px
align: center
---
```

## Project X

Spherinator is a module of the higher-level framework `Project X` for the analysis of astrophysical
data, together with PEST and HiPSter. The modular design allows using them independently or in
combination. [Apache Parquet](https://parquet.apache.org/) is used as internal data format, which
allows storing large amounts of data efficiently.

```{figure} assets/projectx_v2.svg
---
name: fig:projectx
align: center
---
```

- [PEST](https://github.com/HITS-AIN/PEST)
  ([documentation](https://pest.readthedocs.io/en/latest/))
  preprocesses simulation data and generates training data for Spherinator and HiPSter, including
  arbitrary single- and multi-channel images, 3D PPP and PPV cubes, and point clouds.

- [Spherinator](https://github.com/HITS-AIN/Spherinator)
  (this documentation)
  trains models to compress generic data to a spherical latent space.

- [HiPSter](https://github.com/HITS-AIN/HiPSter)
  (documentation coming soon)
  creates the HiPS tilings and catalogs which can be visualized interactively on the
  surface of a sphere with [Aladin Lite](https://github.com/cds-astro/aladin-lite).

```{toctree}
:maxdepth: 2

install.md
spherinator.md
api.md
workflow_orchestration.md
contributing.md
references.md
```
