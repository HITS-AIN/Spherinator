# Spherinator & HiPSter

Spherinator, HiPster and PEST are modules of an higher-level framework `Project X` for the analysis of
astrophysical data. The modulare design allows to use them independently or in combination.
[Apache Parquet](https://parquet.apache.org/) is used as internal data format, which allows to store
large amounts of data efficiently.

![](assets/projectx_v2.svg)


- [PEST](https://github.com/HITS-AIN/PEST)
  preprocess simulation data and generate training data for Spherinator and HiPSter, including
  arbitrary single- and multi-channel images, 3D PPP and PPV cubes, and point clouds.

- [Spherinator](https://github.com/HITS-AIN/Spherinator)
  is a Python package providing variational autoencoders (VAE) reduction generic data to a spherical
  latent space. It is designed to be used with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

- [HiPSter](https://github.com/HITS-AIN/HiPSter)
  creates the HiPS tilings and catalogs which can be visualized interactively on the
  surface of a sphere with [Aladin Lite](https://github.com/cds-astro/aladin-lite).



![](assets/P404_f2.png)

```{toctree}
:maxdepth: 2

pest.md
spherinator.md
hipster.md
contributing.md
workflow_orchestration.md
```
