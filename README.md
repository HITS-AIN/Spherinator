[![Build Status](https://github.com/HITS-AIN/Spherinator/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/HITS-AIN/Spherinator/actions/workflows/python-package.yml?branch=main)
[![Documentation Status](https://readthedocs.org/projects/spherinator/badge/?version=latest)](https://spherinator.readthedocs.io/en/latest/?badge=latest)
![versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

# Spherinator

[Spherinator](https://github.com/HITS-AIN/Spherinator) and
[HiPSter](https://github.com/HITS-AIN/HiPSter) are tools that provide explorative access
and visualization for multimodal data from extremely large astrophysical datasets, ranging from
exascale cosmological simulations to multi-billion object observational galaxy surveys.

A variational autoencoder (VAE) will be trained using
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
to compress the structural data into a low-dimensional spherical latent space.


<p align="center">
  <img src="docs/assets/P404_f2.png" width="400" height="400">
</p>


## Installation

```bash
pip install spherinator
```

## Documentation

[Read The Docs](https://spherinator.readthedocs.io/en/latest/index.html)


## Acknowledgments

Funded by the European Union. This work has received funding from the European High-Performance Computing Joint Undertaking (JU) and Belgium, Czech Republic, France, Germany, Greece, Italy, Norway, and Spain under grant agreement No 101093441.

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European High Performance Computing Joint Undertaking (JU) and Belgium, Czech Republic, France, Germany, Greece, Italy, Norway, and Spain. Neither the European Union nor the granting authority can be held responsible for them.


## License

This project is licensed under the [Apache-2.0 License](http://www.apache.org/licenses/LICENSE-2.0).


## Citation

If you use Spherinator in your research, we provide a [citation](./CITATION.cff) to use:

```bibtex
@article{Polsterer_Spherinator_and_HiPSter_2024,
author = {Polsterer, Kai Lars and Doser, Bernd and Fehlner, Andreas and Trujillo-Gomez, Sebastian},
title = {{Spherinator and HiPSter: Representation Learning for Unbiased Knowledge Discovery from Simulations}},
url = {https://arxiv.org/abs/2406.03810},
doi = {10.48550/arXiv.2406.03810},
year = {2024}
}
```
