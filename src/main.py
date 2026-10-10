# python -m src.main ../data/single_cell/HumanCell1/testChr/320kb_res/chr2_320kb.txt --single_cell


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import pandas as pd
from scipy.stats import spearmanr

from .normalization import preprocess_hic_file, kr_normalization
from .embeddings import generate_embeddings
from .utils import prepare_tensors, contact_to_distance, save_coordinate_mapping, write_pdb, set_seed, spearmanr_gpu, pearson_correlation_loss
from .models import HiCPAGNN

def train_hic_gnn(filepath, device_name="cuda", is_single_cell=False):
    # --- Start Timer ---
    start_time = time.time()

    # --- Setup Directories ---
    if not os.path.exists('Outputs'):
        os.makedirs('Outputs')
        print(" -> Created 'Outputs/' directory.")
        
    filename = os.path.splitext(os.path.basename(filepath))[0]

    # --- Setup Device ---
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # --- Load & Normalize Data ---
    print(" -> Loading Data...")
    raw_matrix, _ = preprocess_hic_file(filepath)
    np.fill_diagonal(raw_matrix, 0)
    
    # --- SINGLE CELL TOGGLE: Normalization ---
    if is_single_cell:
        print("   [!] Single-Cell Mode: Bypassing KR Normalization.")
        norm_matrix = raw_matrix
    else:
        print(" -> Normalizing (KR)...")
        norm_matrix = kr_normalization(raw_matrix)

    # --- 3. Generate Embeddings ---
    print(" -> Generating Embeddings...")
    embeddings = generate_embeddings(raw_matrix)
    
    # Save the embeddings immediately so they aren't lost if training fails
    # Ensure it is a numpy array (in case generate_embeddings returns a tensor)
    if torch.is_tensor(embeddings):
        emb_to_save = embeddings.cpu().numpy()
    else:
        emb_to_save = embeddings
        
    embed_path = f"Outputs/{filename}_embeddings.txt"
    np.savetxt(embed_path, emb_to_save, delimiter='\t')
    print(f" -> Saved embeddings to {embed_path}")

    # --- Save Coordinate Mapping (Auto-Resolution) ---
    # Automatically extracts resolution from 'filename' and sets start_bin to 0
    num_nodes = raw_matrix.shape[0]
    save_coordinate_mapping(filename, start_bin=0, num_nodes=num_nodes, resolution=None)
    
    # --- Prepare Tensors ---
    features, adj_tensor, degrees = prepare_tensors(norm_matrix, embeddings, device)
    
    # --- Training Loop ---
    conversions = np.arange(0.1, 2.0, 0.1)
    
    best_score = -1
    best_loss = float('inf')
    best_struct = None
    best_model_state = None
    best_alpha = None
    
    print(" -> Starting Training...")
       
    # Auto-detect input dimension
    input_dim = embeddings.shape[1]

    for alpha in conversions:
        # Seeding for reproducibility
        set_seed(42)

        model = HiCPAGNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Ground Truth
        truth = contact_to_distance(raw_matrix, alpha, is_single_cell=is_single_cell).to(device)
        
        # --- SINGLE CELL TOGGLE: Dropout Masking ---
        if is_single_cell:
            norm_tensor = torch.tensor(norm_matrix, device=device)
            # Only select upper triangle indices where a true contact exists
            valid_mask = torch.triu(norm_tensor > 0, diagonal=1)
            idx = valid_mask.nonzero(as_tuple=True)
        else:
            # Bulk mode evaluates the entire matrix structure
            idx = torch.triu_indices(truth.shape[0], truth.shape[1], offset=1, device=device)
        
        # Setup AMP Scaler
        scaler = torch.amp.GradScaler('cuda')
        
        best_epoch_loss = float('inf')
        patience_limit = 20
        patience_counter = 0
        
        model.train()
        for epoch in range(2000): 
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output_dist = model(features, adj_tensor, degrees)
                
                # Differentiable Correlation Loss (Forces relative ranking)
                corr_loss = pearson_correlation_loss(output_dist, truth, idx)
                
                loss = corr_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # --- Evaluate Patience ---
            current_loss = loss.item()
            if current_loss < best_epoch_loss - 1e-6:
                best_epoch_loss = current_loss
                patience_counter = 0 
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                break
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            struct = model.get_structure(features, adj_tensor, degrees)
            dist_out = torch.cdist(struct, struct)
            
            # Compute correlation using the pre-calculated idx
            score = spearmanr_gpu(dist_out[idx[0], idx[1]], truth[idx[0], idx[1]])
            
            print(f"   Alpha: {alpha:.1f} | Loss: {loss.item():.6f} | dSCC: {score:.4f} | Epochs: {epoch}")
            
            if score > best_score:
                best_score = score
                best_loss = loss.item()
                best_struct = struct.cpu().numpy()
                best_model_state = model.state_dict()
                best_alpha = alpha
    
    # --- Stop Timer ---
    end_time = time.time() 
    total_time = end_time - start_time

    # --- 6. Save Results ---
    print(f"\nFinal Result -> Best dSCC: {best_score:.4f} (Alpha: {best_alpha:.1f})")
    print(f"Total time sec: {total_time:.4f}")

    # Save Weights
    weight_path = f"Outputs/{filename}_weights.pt"
    torch.save(best_model_state, weight_path)
    print(f" -> Saved weights to {weight_path}")
    
    # Save Structure (PDB)
    pdb_path = f"Outputs/{filename}_structure.pdb"
    write_pdb(best_struct, pdb_path)
    print(f" -> Saved structure to {pdb_path}")

    # Save Log File (inside Output folder)
    log_path = f"Outputs/{filename}_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Optimal conversion factor: {best_alpha}\n")
        f.write(f"Optimal dSCC: {best_score}\n")
        f.write(f"Final loss: {best_loss}\n")
    print(f" -> Saved logs to {log_path}")
    
    return best_score





def train_compartment_gnn(filepath, label_path, device_name="cuda"):
    from .models import CompartmentHiCPAGNN
    
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Running A/B Compartment Task on: {device}")
    
    # Load Data (Reuses our exact pipeline)
    print(" -> Loading Data...")
    raw_matrix, unique_nodes = preprocess_hic_file(filepath) # Changed _ to unique_nodes
    np.fill_diagonal(raw_matrix, 0)
    
    print(" -> Normalizing (KR)...")
    norm_matrix = kr_normalization(raw_matrix)

    print(" -> Generating LINE Embeddings...")
    embeddings = generate_embeddings(raw_matrix)
    
    features, adj_tensor, degrees = prepare_tensors(norm_matrix, embeddings, device)
    
    # Load Ground Truth 1D Labels
    print(f" -> Loading Ground Truth Labels from {label_path}")
    df_labels = pd.read_csv(label_path, sep='\t')
    
    # --- Align absolute matrix nodes with relative label nodes ---
    if unique_nodes is not None:
        # Convert absolute unique_nodes to relative 0-based nodes dynamically
        node_offset = min(unique_nodes)
        relative_unique_nodes = [node - node_offset for node in unique_nodes]
        
        # Now they perfectly match our 0-based label files!
        df_labels = df_labels[df_labels['bin_start'].isin(relative_unique_nodes)]
        
    truth_pc1 = torch.tensor(df_labels['pc1'].values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Setup Model
    model = CompartmentHiCPAGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    
    best_loss = float('inf')
    best_model_state = None
    best_predictions = None
    
    print(" -> Starting Training...")
    model.train()
    for epoch in range(1, 1001): 
        optimizer.zero_grad()
        
        pred_pc1 = model(features, adj_tensor, degrees)
        loss = criterion(pred_pc1, truth_pc1)
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            best_predictions = pred_pc1.detach().cpu().numpy()
            
        if epoch % 100 == 0:
            print(f"   Epoch: {epoch:04d} | MSE Loss: {loss.item():.6f}")

    # Save Outputs
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Save the predicted PC1 scores
    df_out = df_labels.copy()
    df_out['predicted_pc1'] = best_predictions
    out_csv = f"Outputs/{filename}_predicted_compartments.csv"
    df_out.to_csv(out_csv, index=False)
    
    print(f"\nFinal Result -> Best MSE Loss: {best_loss:.4f}")
    print(f" -> Saved predicted compartments to {out_csv}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to input sparse matrix file")
    
    # optional arguments
    parser.add_argument("--task", type=str, choices=['3d', 'compartment'], default='3d', 
                        help="Which task to run. Default is 3d.")
    parser.add_argument("--labels", type=str, default=None, 
                        help="Path to PC1 labels (required if task is compartment)")
    parser.add_argument("--single_cell", action="store_true", 
                        help="Enable single-cell mode: Locks alpha to 0.5 and bypasses KR normalization.")

    args = parser.parse_args()
    
    if args.task == '3d':
        train_hic_gnn(args.file, is_single_cell=args.single_cell)
    elif args.task == 'compartment':
        if not args.labels:
            print("[!] Error: You must provide --labels when running the compartment task.")
        else:
            train_compartment_gnn(args.file, args.labels)