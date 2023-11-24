from data import ShapesDataset


def test_init():
    dataset = ShapesDataset("tests/data/shapes")

    assert len(dataset) == 4000

    data = dataset[0]

    assert data.shape == (64, 64)
    assert dataset.get_metadata(0) == {
        "simulation": "shapes",
        "snapshot": "0",
        "subhalo_id": "0",
    }
