# Deep Learning Fraud Detection Project

This project implements an advanced fraud detection system using a combination of unsupervised feature extraction and supervised classification techniques. It leverages **AutoEncoders (AE)** to learn compressed representations of transaction data, which are then used as augmented features for **Gated MLPs**, **CatBoost**, **XGBoost**, and **LightGBM** models.

## Project Structure

- `Models/`: Core architecture definitions for AutoEncoders, Gated MLPs, and Ensemble wrappers.
- `Utils/`: Utility functions for preprocessing, data loading, training loops, and model restoration.
- `checkpoints/`: Directory containing trained model weights and evaluation histories.
- `data/`: (Local only) Directory for raw and processed datasets (e.g., `train_transaction.csv`).
- `test/`: Unit tests for verifying dataset integrity and model architectures.
- `*.ipynb`: Jupyter Notebooks for experimentation, training, and benchmarking.

## Key Features

- **Unsupervised Pre-training:** Uses AutoEncoders to capture complex patterns in transaction data without labels.
- **Gated MLP Architecture:** Employs a gated linear unit (GLU) based MLP for robust binary classification.
- **Ensemble Learning:** Supports both weighted blending and stacking of multiple base learners (Trees + Neural Networks).
- **Threshold Optimization:** Automated search for optimal probability thresholds to maximize the F2-score.

## Model Loading and Usage

For a comprehensive guide on how to load and use the trained models (including AutoEncoders, MLPs, and the final Stacked Ensemble), please refer to the **`ModelLoadingExample.ipynb`** notebook.

### Quick Example: Loading Models

```python
from Utils.ModelLoaders import load_autoencoder, load_fraud_mlp, load_stacked_ensemble
from Utils.TrainUtils import get_device

DEVICE = get_device()

# Load an AutoEncoder
ae_path = "checkpoints/Autoencoder/ae_best_L16.pt"
autoencoder = load_autoencoder(path=ae_path, device=DEVICE)

# Load a Gated MLP that uses the AutoEncoder features
mlp_path = "checkpoints/GatedMLP_AE16/best.pt"
model = load_fraud_mlp(path=mlp_path, encoder=autoencoder, device=DEVICE)
```

For more complex examples, such as loading the full stacked ensemble with gradient-boosted tree members, see the notebook mentioned above.

## Getting Started

1.  **Environment Setup:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Training:**
    Start by training an AutoEncoder using `TrainEncoder.ipynb`, then move to classifier training in `TrainMLP.ipynb` or the tree-based notebooks.
