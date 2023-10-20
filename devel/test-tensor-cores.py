import torch
import torch.nn

bsz , inf, outf = 256, 1024, 2048
tensor = torch.randn(bsz, inf).cuda().half()
layer = torch.nn.Linear(inf, outf).cuda().half()
layer(tensor)
