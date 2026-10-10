![Visitors](https://visitor-badge.laobi.icu/badge?page_id=OluwadareLab.HiCPAGNN)

# HiCPAGNN
------------------------------------------------------------------------------------------------------------------------------------
Description is coming soon!

## 📂 Repository Structure

```text
HiCPAGNN/
├── data/                 # Input Hi-C contact maps (e.g., GM12878_1mb_chr19_list.txt)
├── src/                  # Core source code
│   ├── main.py           # Main training and prediction script
│   ├── models.py         # Custom GraphSAGE layers and Pure PyTorch GNN model definitions
│   ├── embeddings.py     # Embedding logic
│   ├── normalization.py  # Python implementation of Knight-Ruiz (KR) normalization
│   └── utils.py          # Data loading, PDB writing, and seeding tools
├── Outputs/              # Generated results (PDB structures, model weights, logs)
├── environment.yml       # Conda environment configuration
└── README.md             # Project documentation
```

## 🛠️ Installation

We recommend using **Conda** to manage dependencies.

### 1. Clone the Repository
```bash
git clone https://github.com/OluwadareLab/HiCPAGNN.git
cd HiCPAGNN
```

### 2. Create the Environment
Create the `environment.yml` file with the following content (or use the file provided in the repo):

**`environment.yml`**
```yaml
name: hicpagnn
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pandas
  - networkx
  - scikit-learn
  - pip
```

Run the creation command:
```bash
conda env create -f environment.yml
conda activate hicpagnn
```

### 3. Install PyTorch (Hardware Specific)
Since drivers vary between servers (e.g., H100 vs. RTX 4090), install the version of PyTorch that matches your hardware:

**Option A: Standard GPUs (H100, A100, RTX 3090/4090)**
*For systems with CUDA 11.8 - 12.x drivers.*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Option B: Bleeding Edge GPUs (Blackwell / RTX 6000 Ada)**
*For new systems requiring CUDA 13.0+.*
```bash
pip install torch torchvision --pre --extra-index-url https://download.pytorch.org/whl/cu130
```

**Option C: CPU Only**
*For laptops or non-GPU servers.*
```bash
pip install torch torchvision
```

## 🚀 Usage

### Bulk Hi-C Training 
To generate a 3D structure from a contact map, run the main module. The script automatically handles normalization, embedding generation, and training.

```bash
# Syntax: python -m src.main <path_to_data_file>
python -m src.main data/GM12878/1mb_res/chr19_1mb.txt
```

### Generalization
This script evaluates the generalization capability of HiCPAGNN by training the model on a reference Hi-C dataset and evaluating the trained model on a target Hi-C dataset without retraining the model on the target data.

```bash
Syntax: python -m src.generalize <path_to_train_file> <path_to_test_file>
```

Example A: Cross-resolution generalization

```bash
python -m src.generalize ./data/GM12878/1mb_res/chr1_1mb.txt ./data/GM12878/500kb_res/chr1_1mb.txt 
```

Example B: Cross-cell-type generalization

```bash
python -m src.generalize ./data/GM12878/1mb_res/chr1_1mb.txt ./data/K562/1mb_res/chr1_1mb.txt 
```

Example C: Cross-cell and cross-resolution generalization

```bash
python -m src.generalize ./data/GM12878/1mb_res/chr1_1mb.txt ./data/K562/500kb_res/chr1_1mb.txt 
```

### Single-Cell Hi-C Training
To generate a 3D structure from a sparse single-cell Hi-C contact map, run the main module with the --single_cell flag. The script handles single-cell data preprocessing, normalization, embedding generation, and model training to predict the 3D chromatin structure.

```bash
# Syntax: python -m src.main <path_to_data_file> --single_cell
python -m src.main ./data/single_cell/320kb_res/chr4_320kb.txt --single_cell
```


## 📊 Output Files
After a successful run, the following files are saved in the `Outputs/` directory:

* **`*_structure.pdb`**: The predicted 3D genome structure. Capable of being visualized in PyMOL, UCSF Chimera, or other molecular viewers.
* **`*_weights.pt`**: The trained PyTorch model weights.
* **Logs**: Training logs containing the Loss and Distance Spearman Correlation Coefficient (dSCC) for the optimal Alpha parameter.

---

### Developers:

Md Fahimul Kabir Chowdhury<br>
Department of Computer Science and Engineering<br>
University of North Texas<br>
Email: mdfahimulkabirchowdhury@my.unt.edu<br>
<br>

### Contact:

Dr. Oluwatosin Oluwadare <br>
Department of Computer Science and Engineering<br>
University of North Texas<br>
Email: Oluwatosin.Oluwadare@unt.edu <br>

***
### [OluwadareLab, University of North Texas](https://oluwadarelab.com/)

<div style="background-color: black; padding: 20px; text-align: start; border-radius: 8px;">
  <img src="https://webassets.unt.edu/assets/branding/unt-mobile-logo.svg" 
       alt="UNT" 
       style="max-width: 50%; height: auto; margin-top: 10px;">
</div>

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
