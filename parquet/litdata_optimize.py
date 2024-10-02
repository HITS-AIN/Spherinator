import io
import os

import pyarrow.parquet as pq
from lightning.data import optimize
from PIL import Image

# 1. List the parquet files
input_dir = f"/teamspace/studios/this_studio/dataframe_data"
output_dir = "/teamspace/datasets/lightning_data"
parquet_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])


# 2. Define the function to read parquet files and yield the samples
def convert_parquet_to_lightning_data(parquet_file):
    parquet_file = pq.ParquetFile(parquet_file)
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=32):
        df = batch.to_pandas()
        df["image"] = df["image"].apply(lambda x: Image.open(io.BytesIO(x)))
        for row in df.itertuples():
            sample = (
                row.id,
                row.timestamp.strftime("%m/%d/%Y %I:%M %p"),
                row.description,
                row.image,
            )
            yield sample  # -> encode the sample into binary chunks


# 3. Apply the optimize operator over the parquet files
optimize(
    convert_parquet_to_lightning_data,
    parquet_files[:10],
    output_dir,
    num_workers=os.cpu_count(),
    chunk_bytes="64MB",
)
