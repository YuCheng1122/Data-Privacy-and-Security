# Differential Privacy Training for Image Classification

This project demonstrates how to train image classification models using Differential Privacy (DP). The code supports training on multiple datasets including CIFAR-10, MNIST, and SVHN, and compares DP and non-DP training modes.

## Key Features

- **Differential Privacy Training:**  
  Implements per-sample gradient computation using the Backpack library, with gradient clipping and calibrated Gaussian noise addition to ensure differential privacy.

- **Multi-Dataset Support:**  
  The code supports:
  - **CIFAR-10:** Standard 10-class colored image dataset.
  - **MNIST:** Handwritten digit recognition dataset.
  - **SVHN:** Street View House Numbers dataset.
  
- **Model Architecture:**  
  Uses a ResNet20 model architecture for CIFAR-10, with similar setups available for MNIST and SVHN.

- **Utility Functions:**  
  - **Data Loading:** Loads and preprocesses datasets (CIFAR-10, MNIST, SVHN) with appropriate normalization.
  - **Noise Scale Calculation:** Computes the noise standard deviation (sigma) based on a given privacy budget (ε, δ).
  - **Checkpointing:** Saves model parameters and training state.
  - **Learning Rate Adjustment:** Dynamically adjusts learning rate as training progresses.
  - **Plotting:** Generates learning curves and final accuracy comparison charts.

## Environment Requirements

- Python 3.x
- PyTorch
- TorchVision
- Backpack
- NumPy
- Matplotlib
- scikit-learn
- rdp_accountant

## Usage

1. **Install Dependencies:**  
   Use pip or conda to install the required packages:
   ```bash
   pip install torch torchvision backpack numpy matplotlib scikit-learn rdp_accountant
