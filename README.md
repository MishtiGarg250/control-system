# Control System Project

## Overview

This project implements a control system using neural networks and Model Predictive Control (MPC) for managing system dynamics, potentially in the context of solar energy systems with Direct Normal Irradiance (DNI) profiles. The system includes data processing, model training, evaluation, and MPC-based control strategies.

## Features

- **Data Processing**: Handles raw DNI profiles and processes them into training datasets
- **Neural Network Training**: Trains models for system prediction and control
- **MPC Teacher**: Implements Model Predictive Control algorithms
- **Evaluation**: Tools for assessing model performance
- **Plant Simulation**: Simulates the controlled system

## Project Structure

```
control-system/
├── services/
│   ├── requirements.txt          # Python dependencies
│   ├── data/
│   │   ├── X.npy                 # Processed input data
│   │   ├── y.npy                 # Processed output data
│   │   └── processed/            # Additional processed data
│   │       ├── X.npy
│   │       └── y.npy
│   └── raw/
│       └── dni_profiles.npy      # Raw DNI profile data
├── results/
│   ├── model.pt                  # Trained model weights
│   └── figures/                  # Generated plots and figures
└── src/
    ├── dataset.py                # Data loading and preprocessing
    ├── dni_generator.py          # DNI profile generation
    ├── evaluate.py               # Model evaluation scripts
    ├── model.py                  # Neural network model definitions
    ├── mpc_teacher.py            # MPC implementation
    ├── plant.py                  # System plant simulation
    ├── train_nn.py               # Neural network training script
    ├── data/                     # Local data copies
    └── results/                  # Local results copies
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd control-system
   ```

2. Install dependencies:
   ```bash
   pip install -r services/requirements.txt
   ```

## Usage

### Training the Model

Run the training script:
```bash
python src/train_nn.py
```

### Evaluating the Model

Evaluate the trained model:
```bash
python src/evaluate.py
```

### Running MPC Control

Execute the MPC teacher:
```bash
python src/mpc_teacher.py
```

### Data Processing

Process raw data:
```bash
python src/dataset.py
```

## Data

The project uses DNI (Direct Normal Irradiance) profiles for solar energy applications. Raw data is stored in `services/data/raw/`, processed data in `services/data/processed/`.

## Dependencies

Key dependencies include:
- PyTorch (for neural networks)
- NumPy (for numerical computations)
- Matplotlib (for plotting)
- Other dependencies listed in `services/requirements.txt`


