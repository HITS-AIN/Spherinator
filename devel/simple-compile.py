import torch
import lightning.pytorch as pl

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        return torch.relu(self.l1(x))


# create the model
model = SimpleModel()
torch.compile(model)

input = torch.randn(1, 64)
output = model(input)
print(output)
