import lightning.pytorch as pl
import torch

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=4)
        self.example_input_array = torch.randn(1, 64)


    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))
