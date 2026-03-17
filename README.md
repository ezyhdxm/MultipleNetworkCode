# Multiple Weighted Network Analysis (`mnw`)

Estimate shared low-rank structure and identify node-level perturbations across multiple weighted networks observed under different conditions.

This tool implements Algorithm 1 from:

> Yan & Levin (2026). *Estimating Multiple Weighted Networks with Node-Sparse Differences and Shared Low-Rank Structure.*

## Overview

Given networks from a **control group** (baseline) and one or more **treatment groups** (with potential perturbations), the pipeline:

1. **Spectral initialization** — estimates the shared low-rank structure from control-group networks.
2. **Coherence filtering** — excludes high-leverage nodes that could bias support recovery.
3. **Support recovery** — identifies which nodes have group-specific perturbations via SDP or Group Lasso.
4. **Debiased refinement** — produces an improved estimate of the shared structure using all available data.

## Installation

```bash
pip install -r requirements.txt
```

**MOSEK license**: The SDP method requires [MOSEK](https://www.mosek.com/). Obtain a license (free for academic use) and ensure it is accessible. Alternatively, use `--method glasso` which requires no commercial solver.

## Quick Start

### Python API

```python
import numpy as np
from mnw import MultipleNetworkPipeline

# Load your adjacency matrices (n x n, symmetric)
ctrl_1 = np.load("data/control_1.npy")
ctrl_2 = np.load("data/control_2.npy")
treat_A = np.load("data/patient_A.npy")
treat_B = np.load("data/patient_B.npy")

# Configure and run the pipeline
pipeline = MultipleNetworkPipeline(
    rank=3,              # rank of the shared low-rank structure
    support_size=10,     # expected number of perturbed nodes
    support_method="sdp" # or "glasso"
)

result = pipeline.fit(
    Y_control=[ctrl_1, ctrl_2],
    Y_treatment={
        "patient_A": [treat_A],
        "patient_B": [treat_B],
    },
)

# Inspect results
print(result.summary())
print(result.perturbed_nodes)  # {"patient_A": array([...]), "patient_B": array([...])}

# Visualize
result.plot_shared_structure()
result.plot_perturbations()

# Save for later
result.save("output/")
```

### Command Line

```bash
# Basic usage
python analyze.py \
    --control data/ctrl_1.npy data/ctrl_2.npy \
    --treatment data/treat_A.npy data/treat_B.npy \
    --rank 3 --support-size 10 \
    --output results/ --plot

# Using a config file
python analyze.py --config examples/example_config.json --output results/

# Using Group Lasso (no MOSEK needed)
python analyze.py \
    --control data/ctrl_1.csv \
    --treatment data/treat_A.csv \
    --rank 3 --support-size 10 \
    --method glasso --output results/
```

### Running the Demo

Validate the installation with a synthetic example:

```bash
python examples/demo.py
python examples/demo.py --n 500 --rank 3 --support-size 8
```

## API Reference

### `MultipleNetworkPipeline`

```python
MultipleNetworkPipeline(
    rank: int,                          # Rank of shared structure
    support_size: int | dict[str, int], # Perturbed nodes per group
    support_method: str = "sdp",        # "sdp" or "glasso"
    coherence_threshold: float = None,  # Row-norm threshold for filtering
    verbose: bool = True,               # Log progress
)
```

**`fit(Y_control, Y_treatment) -> NetworkAnalysisResult`**

- `Y_control`: List of n x n symmetric numpy arrays (control group, no perturbations).
- `Y_treatment`: Dict mapping group IDs to lists of n x n symmetric numpy arrays.

### `NetworkAnalysisResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `M_hat` | `ndarray (n,n)` | Estimated shared low-rank structure |
| `U_hat` | `ndarray (n,r)` | Estimated eigenspace |
| `Lambda_hat` | `ndarray (r,)` | Estimated eigenvalues |
| `perturbed_nodes` | `dict[str, ndarray]` | Perturbed node indices per group |
| `B_hats` | `dict[str, ndarray]` | Estimated perturbation matrices per group |
| `coherent_nodes` | `ndarray` | High-coherence nodes excluded from analysis |

| Method | Description |
|--------|-------------|
| `summary()` | Human-readable text summary |
| `plot_shared_structure()` | Heatmap of estimated M* |
| `plot_perturbations()` | Heatmap of estimated B* per group |
| `save(path)` | Save all results to directory |
| `load(path)` | Load previously saved results |

## CLI Reference

```
python analyze.py [OPTIONS]

Options:
  --config PATH          JSON config file (see examples/example_config.json)
  --control PATH ...     Control-group matrix files (.npy or .csv)
  --treatment PATH ...   Treatment-group matrix files (.npy or .csv)
  --rank INT             Rank of shared structure
  --support-size INT     Number of perturbed nodes to detect
  --method {sdp,glasso}  Support recovery method (default: sdp)
  --output DIR           Output directory (default: results/)
  --plot                 Generate and save plots
  --quiet                Suppress progress messages
```

## Input Format

Adjacency matrices can be provided as:
- **NumPy files** (`.npy`): saved via `np.save()`.
- **CSV files** (`.csv`): comma-separated values, no header, no index.
- **TSV files** (`.tsv`): tab-separated values.

All matrices must be square, symmetric, and of the same dimension n.

## Model

The pipeline assumes networks follow:

- **Control group**: Y_s = M* + W_s (shared structure + noise)
- **Treatment group**: Y_s = M* + B*_s + W_s (shared structure + node-sparse perturbation + noise)

where M* is a rank-r matrix and each B*_s is node-sparse (only m rows/columns are nonzero, corresponding to perturbed nodes).

## Project Structure

```
MultipleNetworkCode/
  README.md              # This file
  requirements.txt       # Python dependencies
  analyze.py             # CLI entry point
  mnw/                   # Core package
    __init__.py          # Public API exports
    pipeline.py          # MultipleNetworkPipeline (main orchestrator)
    results.py           # NetworkAnalysisResult (output container)
    spectral.py          # Low-rank spectral methods
    support_recovery.py  # SDP and Group Lasso for node support
    refinement.py        # Debiased eigenspace estimators
    utils.py             # Matrix norms and input validation
    io.py                # File I/O utilities
  examples/
    demo.py              # Synthetic demo and validation
    example_config.json  # Sample configuration file
```

## License

See the accompanying paper for citation information.
