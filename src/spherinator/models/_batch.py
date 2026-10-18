from typing import Any


def split_reconstruction_batch(batch: Any):
    """Return the model input and reconstruction target from a supported batch."""
    if isinstance(batch, dict):
        if len(batch) != 1:
            raise ValueError("Dictionary batches must contain exactly one reconstruction column")
        value = next(iter(batch.values()))
        return value, value
    if isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise ValueError("Sequence batches must contain exactly (augmented, original)")
        return batch[0], batch[1]
    return batch, batch
