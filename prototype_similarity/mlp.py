import torch
from torch import nn
import pytorch_lightning as pl

class OODClassiferDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size=25, output_size=2):
        super(MLP, self).__init__()
        self.automatic_optimization = False
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    
    def training_step(self, train_batch, batch_idx):
        opt_reg = self.optimizers()
        x, y = train_batch
        
        y_pred = self.forward(x)
        train_loss_mlp = self.loss(y_pred, y)
        self.log('train_loss_mlp', train_loss_mlp)

        opt_reg.zero_grad()
        self.manual_backward(train_loss_mlp)
        opt_reg.step()
    
    def configure_optimizers(self):
        opt_reg = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [opt_reg]