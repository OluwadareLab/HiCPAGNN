import os
import sys
import time
import shutil
import argparse
import subprocess

import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes

from src.models import HiCPAGNN
from src.normalization import preprocess_hic_file, kr_normalization
from src.embeddings import generate_embeddings
from src.utils import (
    prepare_tensors,
    contact_to_distance,
    spearmanr_gpu,
    write_pdb,
)


def domain_alignment(list1, list2, embeddings1, embeddings2):
    idx1 = np.unique(list1[:, 0]).astype(int)
    idx2 = np.unique(list2[:, 0]).astype(int)

    if len(idx1) == 0 or len(idx2) == 0:
        raise ValueError("Empty coordinate map.")

    A_list, B_list = [], []
    used_idx2 = set()

    for i in range(len(idx2) - 1):
        for j in range(len(idx1) - 1):
            if (
                idx2[i] >= idx1[j]
                and idx2[i + 1] <= idx1[j + 1]
                and i not in used_idx2
            ):
                A_list.append(embeddings2[i, :])
                B_list.append(embeddings1[j, :])
                used_idx2.add(i)

        if i not in used_idx2:
            closest = np.argmin(np.abs(idx1 - idx2[i]))
            A_list.append(embeddings2[i, :])
            B_list.append(embeddings1[closest, :])
            used_idx2.add(i)

    # Include the final evaluation bin.
    i = len(idx2) - 1
    if i not in used_idx2:
        closest = np.argmin(np.abs(idx1 - idx2[i]))
        A_list.append(embeddings2[i, :])
        B_list.append(embeddings1[closest, :])

    if not A_list:
        raise ValueError("No bins available for domain alignment.")

    A = np.vstack(A_list)
    B = np.vstack(B_list)

    transform = orthogonal_procrustes(A, B)[0]
    aligned = embeddings2 @ transform

    return torch.as_tensor(aligned, dtype=torch.float32)


def infer_resolution(path, chromosome):
    filename = os.path.basename(path)
    prefix = chromosome + "_"

    if filename.startswith(prefix) and filename.endswith(".txt"):
        return filename[len(prefix):-4]

    return "unknown"


def load_weights(path, device):
    try:
        return torch.load(
            path, map_location=device, weights_only=True
        )
    except TypeError:
        return torch.load(path, map_location=device)


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def run_evaluation(args):
    train_path = os.path.abspath(args.train_chromosome)
    eval_path = os.path.abspath(args.test_chromosome)

    for path, label in [
        (train_path, "Training"),
        (eval_path, "Evaluation"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    train_chrom = os.path.basename(train_path).split("_")[0]
    eval_chrom = os.path.basename(eval_path).split("_")[0]

    if not train_chrom.startswith("chr"):
        raise ValueError("Training filename must look like chr1_1mb.txt.")
    if not eval_chrom.startswith("chr"):
        raise ValueError("Evaluation filename must look like chr1_250kb.txt.")

    train_res = args.train_res or infer_resolution(
        train_path, train_chrom
    )
    eval_res = args.eval_res or infer_resolution(
        eval_path, eval_chrom
    )

    if "unknown" in (train_res, eval_res):
        raise ValueError(
            "Could not infer resolutions from filenames. "
            "Use --train_res and --eval_res."
        )

    if train_chrom == eval_chrom and train_res != eval_res:
        experiment_type = "CROSS-RESOLUTION GENERALIZATION"
    elif train_chrom == eval_chrom:
        experiment_type = "SAME-RESOLUTION EVALUATION"
    elif train_res != eval_res:
        experiment_type = "CROSS-CHROMOSOME AND CROSS-RESOLUTION"
    else:
        experiment_type = "CROSS-CHROMOSOME EVALUATION"

    output_dir = os.path.join(
        args.weights_dir,
        f"GM12878_GM12878_Eval_{train_res}_to_{eval_res}",
    )
    os.makedirs(output_dir, exist_ok=True)

    weight_path = os.path.join(
        output_dir, f"{train_chrom}_{train_res}_weights.pt"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    start_time = time.time()

    # ================================================================
    # PHASE 0: CONFIGURATION
    # ================================================================
    section("HiCPAGNN SINGLE-CHROMOSOME EVALUATION")

    print(f"Experiment type       : {experiment_type}")
    print(f"Training chromosome   : {train_chrom}")
    print(f"Evaluation chromosome : {eval_chrom}")
    print(f"Training resolution   : {train_res}")
    print(f"Evaluation resolution : {eval_res}")
    print(f"Training input        : {train_path}")
    print(f"Evaluation input      : {eval_path}")
    print(f"Compute device        : {device}")
    print(f"Expected weights      : {weight_path}")

    print("\nPROTOCOL")
    print("  1. Train on the reference file only if weights are missing.")
    print("  2. Load the reference model weights.")
    print("  3. Generate embeddings and align the evaluation data.")
    print("  4. Predict the structure without retraining HiCPAGNN.")

    # ================================================================
    # PHASE 1: TRAIN OR LOAD WEIGHTS
    # ================================================================
    section("PHASE 1: MODEL TRAINING / WEIGHT SELECTION")

    if os.path.exists(weight_path):
        training_status = "EXISTING WEIGHTS LOADED"
        print("Existing trained weights found.")
        print(f"Loading: {weight_path}")
        print("New model training: NO")

    else:
        training_status = "NEW MODEL TRAINED"
        print("Weights not found; automatic training is required.")
        print(f"Training input: {train_path}")
        print("Evaluation input passed to training: NO")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.main", train_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print("\n--- TRAINING LOG ---")
            print(result.stdout)
            print("--- END TRAINING LOG ---")

        except subprocess.CalledProcessError as exc:
            print("[ERROR] Training failed:")
            print(exc.stdout or str(exc))
            return

        generated_files = [
            f"{train_chrom}_{train_res}_coordinate_map.bed",
            f"{train_chrom}_{train_res}_embeddings.txt",
            f"{train_chrom}_{train_res}_log.txt",
            f"{train_chrom}_{train_res}_structure.pdb",
            f"{train_chrom}_{train_res}_weights.pt",
        ]

        for filename in generated_files:
            src = os.path.join(args.training_output_dir, filename)
            dst = os.path.join(output_dir, filename)

            if os.path.exists(src):
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.move(src, dst)
                print(f"Saved training artifact: {dst}")

        if not os.path.exists(weight_path):
            print(f"[ERROR] Expected weights not found: {weight_path}")
            print("Check training output paths and filenames.")
            return

        print("Training completed successfully.")

    # ================================================================
    # PHASE 2: LOAD TRAINED MODEL
    # ================================================================
    section("PHASE 2: LOAD TRAINED MODEL")

    model = HiCPAGNN().to(device)

    try:
        model.load_state_dict(load_weights(weight_path, device))
        model.eval()
    except Exception as exc:
        print(f"[ERROR] Loading model weights failed: {exc}")
        return

    print(f"Weights loaded: {weight_path}")
    print("Model retraining during inference: DISABLED")

    # ================================================================
    # PHASE 3: DATA PREPARATION
    # ================================================================
    section("PHASE 3: DATA PREPARATION AND EMBEDDINGS")

    try:
        print(f"\n[REFERENCE] {os.path.basename(train_path)}")
        raw_train = np.loadtxt(train_path)
        raw_matrix_train, _ = preprocess_hic_file(train_path)
        np.fill_diagonal(raw_matrix_train, 0)

        print("Generating reference embeddings...")
        embeddings_train = generate_embeddings(raw_matrix_train)

        print(f"\n[EVALUATION] {os.path.basename(eval_path)}")
        raw_eval = np.loadtxt(eval_path)
        raw_matrix_eval, _ = preprocess_hic_file(eval_path)
        np.fill_diagonal(raw_matrix_eval, 0)

        max_edges = 15000000
        current_edges = np.count_nonzero(raw_matrix_eval)

        if current_edges > max_edges:
            flat_mat = raw_matrix_eval.flatten()
            threshold = np.partition(
                flat_mat, -max_edges
            )[-max_edges]
            raw_matrix_eval[raw_matrix_eval < threshold] = 0

        norm_matrix_eval = kr_normalization(raw_matrix_eval)

        print("Generating evaluation embeddings...")
        embeddings_eval = generate_embeddings(raw_matrix_eval)

        print("\nAligning evaluation embeddings to reference...")
        aligned_embeddings = domain_alignment(
            raw_train,
            raw_eval,
            embeddings_train,
            embeddings_eval,
        )

        features, adj, degrees = prepare_tensors(
            norm_matrix_eval, aligned_embeddings, device
        )

        print("Data preparation completed.")
        print("HiCPAGNN model training in this phase: NO")

    except Exception as exc:
        print(f"[ERROR] Data preparation failed: {exc}")
        return

    # ================================================================
    # PHASE 4: ZERO-SHOT INFERENCE AND SCORING
    # ================================================================
    section("PHASE 4: STRUCTURE INFERENCE")

    print(f"Using weights: {weight_path}")
    print("Gradient computation: DISABLED")
    print("Model retraining: DISABLED")

    try:
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    struct = model.get_structure(features, adj, degrees)
                    dist_out = torch.cdist(struct, struct)
            else:
                struct = model.get_structure(features, adj, degrees)
                dist_out = torch.cdist(struct, struct)

        # Alpha selection uses target contacts; the resulting score
        # is therefore target-optimized rather than strictly held out.
        conversions = np.arange(0.1, 2.0, 0.1)
        best_dSCC = -float("inf")
        best_alpha = None

        idx = torch.triu_indices(
            dist_out.shape[0],
            dist_out.shape[1],
            offset=1,
            device=device,
        )

        print("\nScoring conversion factors:")

        for alpha in conversions:
            truth = contact_to_distance(
                raw_matrix_eval, alpha
            ).to(device)

            score = spearmanr_gpu(
                dist_out[idx[0], idx[1]],
                truth[idx[0], idx[1]],
            )

            score_value = (
                score.item()
                if isinstance(score, torch.Tensor)
                else float(score)
            )

            print(f"  alpha={alpha:.1f} | dSCC={score_value:.4f}")

            if np.isfinite(score_value) and score_value > best_dSCC:
                best_dSCC = score_value
                best_alpha = float(alpha)

        if best_alpha is None:
            raise ValueError("No valid dSCC score was obtained.")

        structure_path = os.path.join(
            output_dir, f"{eval_chrom}_structure.pdb"
        )
        write_pdb(struct.cpu().numpy(), structure_path)

        log_path = os.path.join(
            output_dir, f"{eval_chrom}_log.txt"
        )

        runtime = time.time() - start_time

        with open(log_path, "w") as f:
            f.write("HiCPAGNN Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Experiment type: {experiment_type}\n")
            f.write(f"Training chromosome: {train_chrom}\n")
            f.write(f"Evaluation chromosome: {eval_chrom}\n")
            f.write(f"Training resolution: {train_res}\n")
            f.write(f"Evaluation resolution: {eval_res}\n")
            f.write(f"Training status: {training_status}\n")
            f.write(f"Weights: {weight_path}\n")
            f.write("Model retraining during inference: NO\n")
            f.write("Domain alignment: YES\n")
            f.write("Alpha selected using target contacts: YES\n")
            f.write(f"Optimal alpha: {best_alpha:.1f}\n")
            f.write(f"Target-optimized dSCC: {best_dSCC:.4f}\n")
            f.write(f"Runtime seconds: {runtime:.2f}\n")

        # ============================================================
        # PHASE 5: EXPERIMENT REPORT
        # ============================================================
        section("FINAL EVALUATION SUMMARY")

        print(f"Experiment type : {experiment_type}")
        print(f"Training status : {training_status}")
        print(f"Training input  : {os.path.basename(train_path)}")
        print(f"Evaluation input: {os.path.basename(eval_path)}")
        print("Model retraining during inference: NO")
        print(f"Optimal alpha   : {best_alpha:.1f}")
        print(f"Best dSCC       : {best_dSCC:.4f}")
        print(f"Runtime         : {runtime:.2f} seconds")
        print(f"Structure       : {structure_path}")
        print(f"Evaluation log  : {log_path}")

        report_path = os.path.join(
            output_dir, "experiment_report.txt"
        )

        with open(report_path, "w") as f:
            f.write("HiCPAGNN EXPERIMENT REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("[1] EXPERIMENT CONFIGURATION\n")
            f.write(f"Experiment type: {experiment_type}\n")
            f.write(f"Training chromosome: {train_chrom}\n")
            f.write(f"Evaluation chromosome: {eval_chrom}\n")
            f.write(f"Training resolution: {train_res}\n")
            f.write(f"Evaluation resolution: {eval_res}\n")
            f.write(f"Training input: {train_path}\n")
            f.write(f"Evaluation input: {eval_path}\n\n")

            f.write("[2] MODEL TRAINING\n")
            f.write(f"Training status: {training_status}\n")
            f.write(f"Model weights: {weight_path}\n")
            f.write("Evaluation input passed to src.main: NO\n")
            f.write("Model retraining during inference: NO\n\n")

            f.write("[3] EVALUATION\n")
            f.write("Evaluation embedding generation: YES\n")
            f.write("Domain alignment: YES\n")
            f.write("HiCPAGNN inference: YES\n")
            f.write("Target contacts used for alpha selection: YES\n\n")

            f.write("[4] RESULTS\n")
            f.write(f"Optimal alpha: {best_alpha:.1f}\n")
            f.write(f"Target-optimized dSCC: {best_dSCC:.4f}\n")
            f.write(f"Runtime seconds: {runtime:.2f}\n")
            f.write(f"Runtime minutes: {runtime / 60:.2f}\n")

        print(f"Experiment report: {report_path}")
        print("\nEVALUATION COMPLETED")

    except Exception as exc:
        print(f"[ERROR] Structure inference failed: {exc}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-chromosome HiCPAGNN generalization evaluation."
    )

    parser.add_argument(
        "train_chromosome",
        help="Training/reference file, e.g. chr1_1mb.txt",
    )
    parser.add_argument(
        "test_chromosome",
        help="Evaluation file, e.g. chr1_250kb.txt",
    )
    parser.add_argument("--train_res", default=None)
    parser.add_argument("--eval_res", default=None)
    parser.add_argument("--weights_dir", default="Outputs")
    parser.add_argument("--training_output_dir", default="Outputs")

    run_args = parser.parse_args()
    run_evaluation(run_args)