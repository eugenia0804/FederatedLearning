# Federated Learning 

This repository contains a custom simulation framework for **Federated Learning (FL)** experiments. It is designed to study the effects of **client participation rates** and **Differential Privacy (Laplace Noise)** on model convergence and accuracy.

## 📂 Repository Structure

The project is organized as follows:

```bash
├── data/               # Directory for training and testing datasets
├── deliverables/       # Jupyter Notebooks for analysis and visualization
│   ├── ablation_base.ipynb   # Analysis of client fraction (C) impact
│   ├── ablation_noisy.ipynb  # Analysis of Laplace noise impact
│   └── dataset_vis.ipynb     # Visualization of dataset and noise levels
├── runs/               # Output directory for experiment logs and plots
├── client.py           # Client-side logic (local training)
├── clientActor.py      # Actor wrapper for simulating multiple clients
├── model.py            # Neural Network architecture definition
├── server.py           # Server-side logic (aggregation and coordination)
├── run.py              # Main entry point for running simulations
└── requirements.txt    # Python dependencies
```