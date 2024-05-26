import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeMatchingModel(nn.Module):
    def __init__(self, input_dim, num_prototypes):
        super(PrototypeMatchingModel, self).__init__()
        self.prototype_bank = nn.Parameter(torch.randn(num_prototypes, input_dim).abs(), requires_grad=True) # *0.01
        self.input_dim = input_dim
        self.prototype_usage_counts = torch.zeros(num_prototypes, dtype=torch.int32).cuda()
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1)  # Reshape to 2D for easier computation

        # Normalize input and prototypes
        x = F.normalize(x, p=2, dim=1)
        prototypes_normalized = F.normalize(self.prototype_bank, p=2, dim=1)
        #prototypes_normalized = self.prototype_bank

        # Calculate cosine similarity
        #x = x.unsqueeze(2)  # Add singleton dimension for broadcasting
        # 
        x = x.unsqueeze(3) #.transpose(1, 2) # batch_size, n_ch, h*w, 1
        prototypes_normalized = prototypes_normalized.t().unsqueeze(0).unsqueeze(2) # 1, n_ch, 1, n_proto
        similarities = (x * prototypes_normalized).sum(dim=1) # batch_size, h*w, n_proto
        #similarities = torch.sqrt(torch.square(x - prototypes_normalized).sum(dim=1)) # batch_size, h*w, n_proto

        # Find index of closest prototype for each position
        _, indices = torch.max(similarities, dim=2) # batch_size, h*w
        #_, indices = torch.min(similarities, dim=2) # batch_size, h*w
        indices_flat = indices.flatten()
        self.prototype_usage_counts.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.int32).cuda())

        # Replace each position with its closest prototype
        reconstructed = torch.index_select(self.prototype_bank, 0, indices.view(-1))
        #reconstructed = reconstructed.view(batch_size, channels, height, width)
        #reconstructed = reconstructed.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        #reconstructed, indices = replace_with_weighted_average(self.prototype_bank, similarities, k=5)
        reconstructed = reconstructed.view(batch_size, channels, height, width)
        reconstructed = reconstructed.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return reconstructed, indices

    def reinit_unused(self, n=100):
        # Reinit bottom N prototypes
        idx_to_reinit = torch.argsort(self.prototype_usage_counts, dim=0)[:n]
        print(f"Reinitialize: {idx_to_reinit.cpu().detach().numpy()}")

        bank_detach = self.prototype_bank.detach()
        bank_detach[idx_to_reinit] = torch.randn(len(idx_to_reinit), self.prototype_bank.shape[1], device="cuda").abs()*0.01
        self.prototype_usage_counts = torch.zeros(len(self.prototype_bank), dtype=torch.int32).cuda()

        with torch.no_grad():
            self.prototype_bank.copy_(bank_detach)
            #self.prototype_bank.grad.data.zero_()