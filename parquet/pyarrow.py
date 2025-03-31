import numpy as np
import pyarrow as pa

array = np.random.rand(2, 3, 4)
tensor = pa.Tensor.from_numpy(array)

data = [
    pa.array([1, 2, 3]),
    pa.array(["foo", "bar", None]),
    pa.array([True, None, True]),
    pa.array([tensor, None, tensor]),
]

print(data)

my_schema = pa.schema(
    [
        ("f0", pa.int8()),
        ("f1", pa.string()),
        ("f2", pa.bool_()),
        ("tensors_int", pa.int32()),
    ]
)

table = pa.Table.from_arrays(data, schema=my_schema)
