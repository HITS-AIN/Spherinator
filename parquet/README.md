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
