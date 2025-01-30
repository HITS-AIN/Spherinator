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

- Parquet (https://parquet.apache.org/docs/)
    - columnar storage (structured data)
    - fast by vectorized processing
- HDF5
    - complex data structures
- CSV
    - row-based storage (structured data)
- JSON
- XML

## Apache Arrow

In-memory data processing framework

Docu: https://arrow.apache.org/docs/index.html
Tracker: https://issues.apache.org/jira/projects/ARROW


## parquet-tools

Command line tool to inspect Parquet files.

```bash
pip install parquet-tools
```

```bash
parquet-tools inspect XpContinuousMeanSpectrum_000000-003111.parquet
parquet-tools show --head 10 XpContinuousMeanSpectrum_000000-003111.parquet
```
