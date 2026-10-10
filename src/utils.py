import torch
import numpy as np
import random
import os
import re
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr

def set_seed(seed=42):
    """
    Locks all random number generators to a fixed seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def contact_to_distance(contacts, alpha, is_single_cell=False):
    """Converts contact freq to distance, branching logic based on data type."""
    c = torch.tensor(contacts, dtype=torch.float32) if not isinstance(contacts, torch.Tensor) else contacts.clone()
    valid_mask = c > 0
    d = torch.zeros_like(c)
    
    if is_single_cell:
        # Normalize raw contacts to [0, 1]
        max_contact = torch.max(c[valid_mask])
        c_norm = c.clone()
        if max_contact > 0:
            c_norm[valid_mask] = c[valid_mask] / max_contact
            
        # Apply inverse power law (Strictly using alpha=0.5 later)
        d[valid_mask] = (1.0 / c_norm[valid_mask]) ** alpha
        
        # Unscaled penalty for missing edges
        max_dist = torch.max(d)
        d[~valid_mask] = max_dist * 1.5 if max_dist > 0 else 1.5
        
    else:
        # --- ORIGINAL BULK DEFAULT ---
        d[valid_mask] = (1.0 / c[valid_mask]) ** alpha
        max_dist = torch.max(d)
        if max_dist > 0:
            d = d / max_dist 
        d[~valid_mask] = 1.5
        
    d.fill_diagonal_(0)
    return d



def write_pdb(positions, pdb_file):
    """Saves the 3D coordinates, allowing the structure to natively expand."""
    # Center the structure at the origin so PyMOL can focus on it easily
    positions = positions - np.mean(positions, axis=0)
    
    # Direct Expansion (No max_val squashing!)
    # We multiply by 100 to spread the nodes out, exactly like the original code,
    # which prevents the main structure from looking like a dense, solid tube.
    positions = positions * 50.0 
    
    with open(pdb_file, "w") as o_file:
        o_file.write("\n")
        bin_num = len(positions)
        
        # Write Coordinates
        for i in range(1, bin_num+1):
            line = "ATOM  %5d  CA  MET A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n" % (i, i, positions[i-1][0], positions[i-1][1], positions[i-1][2])
            o_file.write(line)
            
        # Write Explicit Backbone Connections
        for i in range(1, bin_num):
            col1 = "CONECT"
            col2 = str(i).rjust(5)
            col3 = str(i+1).rjust(5)
            o_file.write(f"{col1}{col2}{col3}\n")
            
        o_file.write("END")



def spearmanr_gpu(x, y):
    """
    Computes Spearman correlation on CPU using float64 to prevent overflow 
    and safely handle rank ties for large contact matrices.
    """
    # Detach from graph, move to CPU, convert to numpy, and FORCE float64
    x_np = x.detach().cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().astype(np.float64)
    
    # scipy's spearmanr calculates the correlation on the ranks correctly (handling ties)
    corr, _ = spearmanr(x_np, y_np)
    
    # In case of mathematically undefined correlation (e.g., all NaNs or a completely flat array)
    if np.isnan(corr):
        return 0.0
        
    return corr


def pearson_correlation_loss(pred_dist, target_dist, idx):
    """
    Differentiable rank-aware loss.
    Extracts the upper triangle to perfectly mirror the Spearman evaluation.
    """
    pred = pred_dist[idx[0], idx[1]]
    target = target_dist[idx[0], idx[1]]
    
    # Subtract means
    pred_mean = pred - pred.mean()
    target_mean = target - target.mean()
    
    # Calculate covariance and variances
    covariance = (pred_mean * target_mean).sum()
    pred_var = torch.sqrt((pred_mean ** 2).sum() + 1e-8)
    target_var = torch.sqrt((target_mean ** 2).sum() + 1e-8)
    
    # Pearson correlation coefficient (ranges from -1 to 1)
    pearson_corr = covariance / (pred_var * target_var)
    
    # We MINIMIZE loss, so we return (1 - corr)
    return 1.0 - pearson_corr


def prepare_tensors(norm_matrix, embeddings, device, use_sparse=True):
    """Memory-efficient tensor preparation, exporting raw Cardinality (degrees)."""
    adj = torch.tensor(norm_matrix, dtype=torch.float32)
    
    # Add self-loops so nodes pay attention to their own features
    adj = adj + torch.eye(adj.shape[0])
    
    # Capture raw Cardinality (Node Degrees) 
    degree = adj.sum(dim=1)
    
    # Normalize the matrix for standard message passing
    degree_inv = degree.pow(-1)
    degree_inv[torch.isinf(degree_inv)] = 0
    norm_adj_tensor = adj * degree_inv.unsqueeze(1)
    
    if use_sparse: 
        norm_adj_tensor = norm_adj_tensor.to_sparse()
        
    # Return the 3 required variables for our custom CPA layers
    return (
        torch.tensor(embeddings, dtype=torch.float32).to(device), 
        norm_adj_tensor.to(device),
        degree.to(device) 
    )



# The first function "get_resolution_from_filename" uses standard Python Regular Expressions (re) to hunt 
# for strings like "10kb" or "1mb" and convert them into mathematical 
# base pairs (10000 or 1000000). 
# The second function uses that number to generate the BED file.

def get_resolution_from_filename(filename):
    """Automatically extracts the integer base-pair resolution from a filename."""
    name = filename.lower()
    
    # Search for 'kb' (e.g., '10kb' -> 10000)
    match_kb = re.search(r'(\d+)kb', name)
    if match_kb:
        return int(match_kb.group(1)) * 1000
        
    # Search for 'mb' (e.g., '1mb' -> 1000000)
    match_mb = re.search(r'(\d+)mb', name)
    if match_mb:
        return int(match_mb.group(1)) * 1000000
        
    # Fallback if the filename is poorly named
    print(f" [!] Warning: Could not detect 'kb' or 'mb' in '{filename}'. Defaulting to 10kb.")
    return 10000

def save_coordinate_mapping(filename, start_bin, num_nodes, resolution=None):
    """Creates a strict, headerless 3-column BED file (chr, start, end)."""
    
    # Auto-detect resolution if not explicitly provided
    if resolution is None:
        resolution = get_resolution_from_filename(filename)
        
    # Extract chromosome name (e.g., 'chr8_10kb' -> 'chr8')
    chr_name = filename.split('_')[0] if 'chr' in filename.lower() else "unknown_chr"
    
    # Changed extension to .bed (the bioinformatics standard)
    map_path = f"Outputs/{filename}_coordinate_map.bed"
    
    with open(map_path, "w") as f:
        # NO header. Just standard 3-column BED format.
        for node_id in range(num_nodes):
            original_bin = start_bin + node_id
            start_bp = original_bin * resolution
            end_bp = start_bp + resolution
            
            # Format: chr1    0       100000
            f.write(f"{chr_name}\t{start_bp}\t{end_bp}\n")
            
    print(f" -> Saved coordinate mapping to {map_path} (Auto-Res: {resolution} bp)")