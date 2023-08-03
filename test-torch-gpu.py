import torch
print("PyTorch version: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
print("Number of CUDA devices: ", torch.cuda.device_count())

import torch_xla
print(torch_xla.__version__)