import torch
import lightning.pytorch as pl

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))


# create the model
model = SimpleModel()
script = model.to_torchscript()

# save for use in production environment
torch.jit.save(script, "model.pt")
