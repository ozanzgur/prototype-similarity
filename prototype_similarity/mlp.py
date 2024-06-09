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
    def __init__(self, reconstructor, input_size, hidden_size=25, output_size=1, learning_rate=1e-2, dropout_rate=0.8):
        super(MLP, self).__init__()
        self.automatic_optimization = True
        self.learning_rate = learning_rate
        self.reconstructor = reconstructor
        self.reconstructor.eval()
        for param in self.reconstructor.parameters():
            param.requires_grad = False

        self.bn = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x):
        with torch.no_grad():
            sim_scores = self.reconstructor.prototype_scores(x).detach()

        out = self.bn(sim_scores)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    
    def training_step(self, train_batch, batch_idx):
        #opt_reg = self.optimizers()
        x, y = train_batch
        
        y_pred = self.forward(x)
        train_loss_mlp = self.loss(y_pred, y)
        self.log('train_loss_mlp', train_loss_mlp)

        #opt_reg.zero_grad()
        #self.manual_backward(train_loss_mlp)
        #opt_reg.step()
        return train_loss_mlp
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        #preds = torch.argmax(logits, dim=1)
        #acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        #self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):

        params = list(self.bn.parameters()) +\
            list(self.fc1.parameters()) +\
            list(self.fc2.parameters())
        opt_reg = torch.optim.Adam(params, lr=self.learning_rate)
        return [opt_reg]