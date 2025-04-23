# PEST: Preprocessing Engine for Spherinator Training

[PEST](https://github.com/HITS-AIN/PEST) preprocess simulation data and generate training data for
Spherinator & HiPSter, including arbitrary single- and multi-channel images, 3D PPP and PPV cubes,
and point clouds.

## Installation

PEST can be installed via `pip`:

```bash
pip install git+https://github.com/HITS-AIN/PEST
```

## Usage

## Example 1: Convert Gaia CSV to Parquet

```python
from pest import GaiaConverter

gaia_converter = GaiaConverter(
    with_flux_error=True,
    number_of_workers=1,
)
gaia_converter.convert_all("data/gaia/csv", "data/gaia/parquet")
```


## Example 2: Convert Illustris TNG SKIRT fits images to Parquet

```python
from pest import FitsConverter

FitsConverter(image_size=128).convert_all(
    "data/illustris/fits/TNG100/sdss/snapnum_099/data", "data/illustris/parquet"
)
```
