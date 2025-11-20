## Federated Learning 

### Repository Structure

The project is organized as follows:

```bash
├── data/               # Raw datasets 
├── deliverables/       # Notebooks for experiment analysis and visualization
│   ├── ablation_base.ipynb   # Analysis of client fraction (c) impact
│   ├── ablation_noisy.ipynb  # Analysis of Laplace noise (d) impact
│   └── dataset_vis.ipynb     # Visualization of dataset distributions and noise levels
├── runs/               # Output directory for plots
├── client.py           # Client-side logic (local training)
├── clientActor.py      # Actor wrapper for simulating multiple clients for parallelization
├── model.py            # Neural Network architecture
├── server.py           # Server-side logic (aggregation and coordination)
├── run.py              # Main entry point for running simulations
└── requirements.txt    # Python dependencies
```

### How to run the codebase

1. Install all required packages: 

    `bash pip install -r requirements.txt`

2. To run the default setting (server and all other client run on a single GPU card)

    `bash python run.py`

3. To run the default setting (clients run on other GPU cards)

    `bash python run.py --parallel`

### Training Parameters

The dafault training are set at follows:
```python
num_rounds = 800 # Numbers of communication rounds
batch_size = 64 # Number of training samples processed before the model is updated
lr = 5e-3 # Model Learning Rate
c = 0.1 # Percentage of client to preform local updates in each round
local_epoch = 10 # The number of epochs for one client update 
```

### Data Visualization

Macro-level analysis of data distribution among classes for the overall dataset and specific selected clients is available in `deliverables/dataset_vis.ipynb`.This notebook also demonstrates the visual impact of Differential Privacy, showing how input training images are altered when adding Laplace noise at various scales ($b$).

### Ablation Studies

3 Ablations studies has been conducted:

- **c (Client Participation)**: `ablation_base.ipynb`
    - Increasing $c$ generally improves convergence stability and speed (in terms of rounds), as the aggregated update is more representative of the global population. However, this comes with higher computation and communication overheads per round.
- **E (Client Training Epochs)**:  `ablation_base.ipynb`
    - Increasing $E$ allows clients to learn more from their local data before communicating. While this can reduce the total number of communication rounds needed, setting $E$ too high can lead to "client drift," where local models diverge too much from the global objective, harming final accuracy.
- **b (Noise Scale on input images)**:  `ablation_noisy.ipynb`
    - As the scale of Laplace noise ($b$) increases, the privacy guarantee becomes stronger (Differential Privacy). However, this introduces variance that degrades the model's utility, reduces final testing accuracy.
