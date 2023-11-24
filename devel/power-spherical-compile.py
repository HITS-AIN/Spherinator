import os
import sys

import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../external/power_spherical/'))
from power_spherical import PowerSpherical

loc = torch.tensor([1.0, 0.0])
scale = torch.tensor([600.0])
dist = PowerSpherical(loc, scale)
compiled_dist = torch.compile(dist.rsample)

output = compiled_dist()
print(output)
