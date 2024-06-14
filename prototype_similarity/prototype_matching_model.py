import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBatchNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(CustomBatchNorm2D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_covariance', torch.zeros(num_features, num_features))
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        if self.training:
            # Compute batch statistics per channel
            batch_mean = x.mean(dim=[0, 2, 3], keepdim=True)  # shape: (1, num_features, 1, 1)
            x_centered = x - batch_mean  # shape: (batch_size, num_features, height, width)
            x_flat = x_centered.view(x_centered.size(0), x_centered.size(1), -1)  # shape: (batch_size, num_features, height * width)
            #print(x_flat.shape)
            batch_covariance = torch.einsum('ijk,iab->ja', x_flat, x_flat) / (x_flat.size(-1))
            
            # Update running statistics
            self.running_mean.mul_(self.momentum).add_(batch_mean.squeeze() * (1 - self.momentum))
            self.running_covariance.mul_(self.momentum).add_(batch_covariance.mean(dim=0) * (1 - self.momentum))
        else:
            
            batch_mean = self.running_mean
            batch_covariance = self.running_covariance
        
        # Normalize
        x_normalized = (x - batch_mean.view(1, -1, 1, 1)) * torch.abs(torch.diag(batch_covariance)).view(1, -1, 1, 1) # / torch.sqrt( + self.eps)
        
        # Scale and shift
        out = self.gamma.view(1, -1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1)
        return out

class PrototypeMatchingModel(nn.Module):
    def __init__(self, input_dim, num_prototypes):
        super(PrototypeMatchingModel, self).__init__()
        self.num_prototypes = num_prototypes
        prot_init = torch.randn(num_prototypes, input_dim)

        self.prototype_bank = nn.Parameter(prot_init, requires_grad=True) # *0.01 # .abs()
        self.input_dim = input_dim
        self.prototype_usage_counts = None
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()

        x = x.view(batch_size, channels, -1)  # Reshape to 2D for easier computation

        # Normalize input and prototypes
        x = F.normalize(x, p=2, dim=1) # TODO: try without norm
        prototypes_normalized = F.normalize(self.prototype_bank, p=2, dim=1)

        # Calculate cosine similarity
        x = x.unsqueeze(3) #.transpose(1, 2) # batch_size, n_ch, h*w, 1
        prototypes_normalized = prototypes_normalized.t().unsqueeze(0).unsqueeze(2) # 1, n_ch, 1, n_proto
        similarities = (x * prototypes_normalized).sum(dim=1) # batch_size, h*w, n_proto

        # Find index of closest prototype for each position
        _, indices = torch.max(similarities, dim=2) # batch_size, h*w
        if self.prototype_usage_counts is None:
            self.prototype_usage_counts = torch.zeros((self.num_prototypes, height * width), dtype=torch.int32).cuda()
        self.prototype_usage_counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.int32).cuda())

        # Replace each position with its closest prototype
        reconstructed = torch.index_select(self.prototype_bank, 0, indices.view(-1))
        reconstructed = reconstructed.view(batch_size, channels, height, width)
        reconstructed = reconstructed.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return reconstructed, indices

    def reinit_unused(self, frac=0.33):
        # Reinit bottom N prototypes
        idx_to_reinit = torch.argsort(self.prototype_usage_counts, dim=0)[:int(self.num_prototypes*frac)]
        print(f"Reinitialize: {idx_to_reinit.cpu().detach().numpy()}")

        bank_detach = self.prototype_bank.detach()
        bank_detach[idx_to_reinit] = torch.randn(len(idx_to_reinit), self.prototype_bank.shape[1], device="cuda").abs()*0.01
        self.prototype_usage_counts = torch.zeros(len(self.prototype_bank), dtype=torch.int32).cuda()

        with torch.no_grad():
            self.prototype_bank.copy_(bank_detach)

    def init_with_acts(self, acts):
        init_std = torch.std(acts)
        print(f"Initialization std: {init_std}")
        nn.init.normal_(self.prototype_bank, mean=0.0, std=init_std * 3)
