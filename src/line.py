import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset

class LINEModel(nn.Module):
    """
    PyTorch implementation of LINE (Second Order Proximity).
    Conceptually similar to Skip-gram with Negative Sampling.
    """
    def __init__(self, num_nodes, embedding_dim=128):
        super(LINEModel, self).__init__()
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.vertex_embeddings = nn.Embedding(num_nodes, embedding_dim)

        # Initialize with 0 mean and small std dev (Word2Vec style)
        # This helps convergence compared to Xavier uniform for embeddings
        nn.init.normal_(self.context_embeddings.weight, std=0.01)
        nn.init.normal_(self.vertex_embeddings.weight, std=0.01)

    def forward(self, target_nodes, context_nodes, neg_nodes):
        # Target: (batch_size, dim)
        v = self.vertex_embeddings(target_nodes)
        
        # Context (Positive Neighbors): (batch_size, dim)
        u_pos = self.context_embeddings(context_nodes)
        
        # Negative Samples: (batch_size, k, dim)
        u_neg = self.context_embeddings(neg_nodes)
        
        # --- Positive Score (Maximize dot product) ---
        pos_score = torch.mul(v, u_pos).sum(dim=1)
        pos_loss = F.logsigmoid(pos_score) 
       
        # --- Negative Score (Minimize dot product -> Maximize negative dot) ---
        v_unsqueezed = v.unsqueeze(1) 
        neg_score = torch.mul(v_unsqueezed, u_neg).sum(dim=2) 
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1) 

        # Total loss (Maximize Likelihood -> Minimize Negative Log Likelihood)
        return -(torch.mean(pos_loss + neg_loss))

def train_line(adjacency_matrix, embedding_dim=512, device="cuda", epochs=50, batch_size=4096, neg_ratio=5):
    """
    Main function to train LINE embeddings.
    """
    num_nodes = adjacency_matrix.shape[0]
    
    # --- Safe Data Extraction ---
    # Convert to COO to cleanly separate edges and weights.
    if not sp.issparse(adjacency_matrix):
        coo = sp.coo_matrix(adjacency_matrix)
    else:
        coo = adjacency_matrix.tocoo()
        
    # Strictly filter out zero-weights to prevent multinomial crashes
    mask = coo.data > 0
    rows = coo.row[mask]
    cols = coo.col[mask]
    weights = coo.data[mask]
    
    # Clean weights (remove NaNs) and calculate positive probabilities
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    probs = weights / weights.sum()
    
    # --- Safe Negative Probabilities ---
    node_degrees = np.array(coo.sum(axis=1)).flatten()
    node_degrees = np.nan_to_num(node_degrees, nan=0.0, posinf=0.0, neginf=0.0)
    node_degrees[node_degrees <= 0] = 0.001 # Prevent 0 degree nodes from breaking the power calculation
    
    neg_probs = np.power(node_degrees, 0.75)
    neg_probs = neg_probs / neg_probs.sum()
    
    model = LINEModel(num_nodes, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)
    
    
    # --- Custom 64-bit Cumulative Sums for >16.7 Million Edges ---
    try:
        edges_tensor = torch.tensor(np.column_stack((rows, cols)), dtype=torch.long, device=device)
        
        # 1. Positive Probabilities (64-bit to prevent precision loss on massive arrays)
        probs_tensor = torch.tensor(probs, dtype=torch.float64, device=device)
        probs_tensor = torch.nan_to_num(probs_tensor, nan=0.0)
        probs_tensor = torch.clamp(probs_tensor, min=0.0)
        probs_cumsum = probs_tensor.cumsum(dim=0)
        probs_cumsum = probs_cumsum / probs_cumsum[-1] 
        
        # 2. Negative Probabilities (64-bit)
        neg_probs_tensor = torch.tensor(neg_probs, dtype=torch.float64, device=device)
        neg_probs_tensor = torch.nan_to_num(neg_probs_tensor, nan=0.0)
        neg_probs_tensor = torch.clamp(neg_probs_tensor, min=0.0)
        neg_probs_cumsum = neg_probs_tensor.cumsum(dim=0)
        neg_probs_cumsum = neg_probs_cumsum / neg_probs_cumsum[-1] 
    except:
        # Fallback to CPU if VRAM issue
        edges_tensor = torch.tensor(np.column_stack((rows, cols)), dtype=torch.long, device="cpu")
        probs_cumsum = torch.tensor(probs, dtype=torch.float64, device="cpu").cumsum(dim=0)
        probs_cumsum = probs_cumsum / probs_cumsum[-1]
        neg_probs_cumsum = torch.tensor(neg_probs, dtype=torch.float64, device="cpu").cumsum(dim=0)
        neg_probs_cumsum = neg_probs_cumsum / neg_probs_cumsum[-1]

    print(f"   [LINE] Training for {epochs} epochs on {device}...")
    
    num_edges = len(rows)
    batches_per_epoch = max(1, num_edges // batch_size)
        
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for _ in range(batches_per_epoch):
            optimizer.zero_grad()
            
            # --- O(log N) Binary Search Sampling (Bypasses 2^24 limit) ---
            # Sample Positive Edges
            rand_pos = torch.rand(batch_size, dtype=torch.float64, device=device)
            idx = torch.searchsorted(probs_cumsum, rand_pos)
            idx = torch.clamp(idx, max=len(probs_cumsum) - 1) 
            batch_edges = edges_tensor[idx]
            
            target_nodes = batch_edges[:, 0] 
            context_nodes = batch_edges[:, 1]
            
            # Sample Negative Edges
            rand_neg = torch.rand(batch_size * neg_ratio, dtype=torch.float64, device=device)
            neg_indices = torch.searchsorted(neg_probs_cumsum, rand_neg)
            neg_indices = torch.clamp(neg_indices, max=len(neg_probs_cumsum) - 1)
            neg_nodes = neg_indices.view(batch_size, neg_ratio)

            # Optimize
            loss = model(target_nodes, context_nodes, neg_nodes)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / batches_per_epoch
            print(f"     Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    return model.vertex_embeddings.weight.data.cpu().numpy()
