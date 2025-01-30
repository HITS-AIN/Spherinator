import io
import os

import pyarrow.parquet as pq
from lightning.data import optimize
from PIL import Image

root_dir = "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images"
input_dir = root_dir + "/parquet"
output_dir = root_dir + "lightning"
parquet_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])


def convert_parquet_to_lightning_data(parquet_file):
    parquet_file = pq.ParquetFile(parquet_file)
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=32):
        df = batch.to_pandas()
        df["image"] = df["image"].apply(lambda x: Image.open(io.BytesIO(x)))
        for row in df.itertuples():
            sample = (row.image,)
            yield sample


if __name__ == "__main__":
    optimize(
        convert_parquet_to_lightning_data,
        parquet_files[:10],
        output_dir,
        num_workers=1,
        chunk_bytes="64MB",
    )
