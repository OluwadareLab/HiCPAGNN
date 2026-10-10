import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== With CPA and channel attention =====
class CPA_SAGEConv(nn.Module):
    """Mass-Aware, Normalized CPA SAGE Layer with Channel Attention"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        
        # Feature-Space Scaling (The Mass trick)
        self.mass_mlp = nn.Sequential(
            nn.Linear(1, in_channels // 2),
            nn.GELU(), 
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid() 
        )
        
        # The Channel Attention Gate
        # Squeezes the features down and expands them to find the most important channels
        self.attention_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.GELU(),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, adj_matrix, degrees):
        log_degree = torch.log1p(degrees).unsqueeze(1)
        source_mass = self.mass_mlp(log_degree)
        x_weighted = x * source_mass
        
        # Sparse Matrix Multiplication (Crash-proof)
        if adj_matrix.is_sparse:
            with torch.amp.autocast('cuda', enabled=False):
                out = torch.sparse.mm(adj_matrix.to(torch.float32), x_weighted.to(torch.float32))
            out = out.to(x.dtype)
        else:
            out = torch.matmul(adj_matrix, x_weighted)
            
        # SAGE Update + Skip Connection
        out = self.lin_l(out) + self.lin_r(x)
        
        # Apply the Channel Attention Mechanism
        # The network learns a dynamic attention weight between 0 and 1 for every channel
        attention_weights = self.attention_gate(out)
        out = out * attention_weights
        
        # Stabilize
        out = self.norm(out)
        
        return out



class HiCPAGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CPA_SAGEConv(512, 128)
        self.conv2 = CPA_SAGEConv(128, 128)
        self.conv3 = CPA_SAGEConv(128, 64)
        self.conv4 = CPA_SAGEConv(64, 64)
        
        self.proj1 = nn.Linear(128, 64)
        self.proj2 = nn.Linear(128, 64)
        self.layer_weights = nn.Parameter(torch.ones(4)) 
        self.dec = nn.Linear(64, 3)
    
    def forward(self, x, adj, degrees):
        x1 = F.relu(self.conv1(x, adj, degrees))
        x1_drop = F.dropout(x1, p=0.3, training=self.training)
        
        x2 = F.relu(self.conv2(x1_drop, adj, degrees))
        x2_drop = F.dropout(x2, p=0.3, training=self.training)
        
        x3 = F.relu(self.conv3(x2_drop, adj, degrees))
        x3_drop = F.dropout(x3, p=0.3, training=self.training)
        
        x4 = self.conv4(x3_drop, adj, degrees)
        
        x1_proj = self.proj1(x1)
        x2_proj = self.proj2(x2)
        
        w = F.softmax(self.layer_weights, dim=0)
        x_combined = (w[0] * x1_proj) + (w[1] * x2_proj) + (w[2] * x3) + (w[3] * x4)
        final_struct = self.dec(x_combined)
        return torch.cdist(final_struct, final_struct, p=2)

    def get_structure(self, x, adj, degrees):
        x1 = F.relu(self.conv1(x, adj, degrees))
        x2 = F.relu(self.conv2(x1, adj, degrees))
        x3 = F.relu(self.conv3(x2, adj, degrees))
        x4 = self.conv4(x3, adj, degrees)
        
        x1_proj = self.proj1(x1)
        x2_proj = self.proj2(x2)
        
        w = F.softmax(self.layer_weights, dim=0)
        x_combined = (w[0] * x1_proj) + (w[1] * x2_proj) + (w[2] * x3) + (w[3] * x4)
        return self.dec(x_combined)


class CompartmentHiCPAGNN(HiCPAGNN):
    """
    Inherits CPA_SAGEConv layers from HiCPAGNN.
    Changes the decoder to output a 1D A/B Compartment score.
    """
    def __init__(self):
        super().__init__()
        # Replace the 3D coordinate decoder with a 1D score decoder
        self.dec = nn.Linear(64, 1)
        
    def forward(self, x, adj, degrees):
        x1 = F.relu(self.conv1(x, adj, degrees))
        x1_drop = F.dropout(x1, p=0.3, training=self.training)
        
        x2 = F.relu(self.conv2(x1_drop, adj, degrees))
        x2_drop = F.dropout(x2, p=0.3, training=self.training)
        
        x3 = F.relu(self.conv3(x2_drop, adj, degrees))
        x3_drop = F.dropout(x3, p=0.3, training=self.training)
        
        x4 = self.conv4(x3_drop, adj, degrees)
        
        x1_proj = self.proj1(x1)
        x2_proj = self.proj2(x2)
        
        w = F.softmax(self.layer_weights, dim=0)
        x_combined = (w[0] * x1_proj) + (w[1] * x2_proj) + (w[2] * x3) + (w[3] * x4)
        
        # Directly return the 1D PC1 scores (No cdist distance calculation)
        return self.dec(x_combined)