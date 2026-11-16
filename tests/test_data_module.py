import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from datasets import Image
from PIL import Image as PILImage
from spherinator.data import DataModule


def _parquet_file(tmp_path, size=20):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"data": [np.array([i], dtype=np.float32) for i in range(size)]}), path)
    return path


def test_no_validation_split_keeps_test_disjoint_from_training(tmp_path):
    data = DataModule(str(_parquet_file(tmp_path)), validation_size=0.0, test_size=0.25, batch_size=20)
    data.setup()

    train = {float(value) for batch in data.train_dataloader() for value in batch.flatten()}
    test = {float(value) for batch in data.test_dataloader() for value in batch.flatten()}

    assert len(train) == 15
    assert len(test) == 5
    assert train.isdisjoint(test)


def test_no_validation_split_skips_validation_loop(tmp_path):
    data = DataModule(str(_parquet_file(tmp_path)), validation_size=0.0, test_size=0.0)
    data.setup("fit")

    assert data.val_dataloader() == []


def test_default_single_column_batch_is_tensor(tmp_path):
    data = DataModule(str(_parquet_file(tmp_path)), validation_size=0.2, test_size=0.0)
    data.setup("fit")

    assert isinstance(next(iter(data.train_dataloader())), torch.Tensor)


def _shaped_parquet_dir(tmp_path, files=2, rows=10):
    """Directory of parquet files with a flat 12-element ``data`` column and ``data_shape: (1,12)`` metadata."""
    directory = tmp_path / "ds"
    directory.mkdir()
    for i in range(files):
        table = pa.table(
            {"data": [np.random.rand(12).astype(np.float32) for _ in range(rows)]},
            metadata={"data_shape": "(1,12)"},
        )
        pq.write_table(table, directory / f"part-{i}.parquet")
    return directory


@pytest.mark.parametrize(
    ("path", "rows"),
    [
        ("ds/part-0.parquet", 10),
        ("ds", 20),
        ("ds/*.parquet", 20),
        ("ds/**/*.parquet", 20),
    ],
    ids=["file", "dir", "glob", "recursive-glob"],
)
def test_shape_metadata_is_applied_for_every_path_form(tmp_path, path, rows):
    _shaped_parquet_dir(tmp_path)
    data = DataModule(str(tmp_path / path), validation_size=0.0, test_size=0.0, batch_size=5)
    data.setup("fit")

    assert len(data._train_ds) == rows
    assert next(iter(data.train_dataloader())).shape == (5, 1, 12)


def _image_parquet(path, shape=(8, 6), large=False, rows=4):
    """Parquet file without HF metadata: PNG images as ``{bytes, path}`` structs in ``image`` and ``other``."""
    if large:
        struct_type = pa.struct([("bytes", pa.large_binary()), ("path", pa.large_string())])
    else:
        struct_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    images = [np.random.randint(0, 256, size=shape, dtype=np.uint8) for _ in range(rows)]

    def png(array):
        buffer = io.BytesIO()
        PILImage.fromarray(array).save(buffer, format="PNG")
        return {"bytes": buffer.getvalue(), "path": None}

    table = pa.table(
        {
            "image": pa.array([png(image) for image in images], type=struct_type),
            "other": pa.array([png(image) for image in images], type=struct_type),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return images


def _image_data_module(path):
    data = DataModule(str(path), columns=["image"], validation_size=0.0, test_size=0.0, shuffle=False, batch_size=2)
    data.setup("fit")
    return data


@pytest.mark.parametrize(
    ("shape", "channels"),
    [((8, 6), 1), ((8, 6, 3), 3)],
    ids=["grayscale", "rgb"],
)
@pytest.mark.parametrize("large", [False, True], ids=["binary", "large_binary"])
def test_image_struct_column_is_decoded(tmp_path, shape, channels, large):
    images = _image_parquet(tmp_path / "images.parquet", shape, large)
    data = _image_data_module(tmp_path / "images.parquet")

    batch = next(iter(data.train_dataloader()))
    assert batch.shape == (2, channels, 8, 6)
    assert batch.dtype == torch.float32

    expected = torch.from_numpy(images[0]).reshape(8, 6, channels).permute(2, 0, 1).float() / 255.0
    torch.testing.assert_close(data._train_ds[0], expected)


def test_image_struct_is_decoded_for_nested_directory(tmp_path):
    _image_parquet(tmp_path / "ds" / "sub" / "part-0.parquet")
    data = _image_data_module(tmp_path / "ds")

    assert next(iter(data.train_dataloader())).shape == (2, 1, 8, 6)


def test_unrequested_struct_column_is_not_decoded(tmp_path):
    _image_parquet(tmp_path / "images.parquet")
    data = _image_data_module(tmp_path / "images.parquet")

    features = data._train_ds.dataset.features
    assert isinstance(features["image"], Image)
    assert isinstance(features["other"], dict)


@pytest.mark.parametrize(
    ("validation_size", "test_size"),
    [(-0.1, 0.0), (1.0, 0.0), (0.2, -0.1), (0.2, 1.0)],
)
def test_invalid_split_sizes_are_rejected(validation_size, test_size):
    with pytest.raises(ValueError):
        DataModule("unused", validation_size=validation_size, test_size=test_size)
