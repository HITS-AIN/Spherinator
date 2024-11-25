# General format for datasets

## Requirements

- Store different dataset formats
  - Images
  - Time series
  - Point clouds
  - Graphs
  - Data cubes
  - Spectra
- Store metadata
  - Labels
  - IDs
  - Simulation parameters

- Compressible
- Streamable
- Random access
- Versioning

## Possible dataset formats

- Parquet
    - not designed for storing binary data
    - optimized for columnar storage of structured data
- HDF5
    - complex data structures
- Arrow
    - in-memory columnar data storage
- CSV
    - row-based
- JSON
- XML


## parquet-tools

Command line tool to inspect Parquet files.

```bash
pip install parquet-tools
```

```bash
parquet-tools inspect XpContinuousMeanSpectrum_000000-003111.parquet
parquet-tools show --head 10 XpContinuousMeanSpectrum_000000-003111.parquet
```
