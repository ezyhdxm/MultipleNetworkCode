"""
Data loading and saving utilities.

Supports reading network adjacency matrices from .npy and .csv files,
loading JSON configuration files, and saving analysis results.
"""

import json
import os
from pathlib import Path

import numpy as np


def load_matrix(path: str) -> np.ndarray:
    """
    Load a square matrix from a .npy or .csv file.

    Args:
        path: Path to the file (.npy or .csv/.tsv).

    Returns:
        Loaded matrix as a numpy array.

    Raises:
        ValueError: If the file format is unsupported or the matrix is not square.
    """
    p = Path(path)
    if p.suffix == ".npy":
        M = np.load(path)
    elif p.suffix in (".csv", ".tsv"):
        delimiter = "\t" if p.suffix == ".tsv" else ","
        M = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .npy, .csv, or .tsv.")

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {M.shape} from {path}.")

    return M


def load_matrices(paths: list[str]) -> list[np.ndarray]:
    """Load multiple matrices from a list of file paths."""
    return [load_matrix(p) for p in paths]


def load_config(path: str) -> dict:
    """
    Load a JSON configuration file for the analysis pipeline.

    Expected format:
    {
        "rank": 3,
        "support_size": 10,
        "support_method": "sdp",
        "control": ["path/to/control_1.npy", ...],
        "treatment": {
            "group_A": ["path/to/A_1.npy", ...],
            "group_B": ["path/to/B_1.npy", ...]
        },
        "node_labels": "path/to/labels.csv"  (optional)
    }

    Args:
        path: Path to JSON config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        config = json.load(f)

    required = ["rank", "support_size", "control", "treatment"]
    for key in required:
        if key not in config:
            raise ValueError(f"Config file missing required key: '{key}'.")

    return config


def save_results(result, output_dir: str) -> None:
    """
    Save analysis results to a directory.

    Creates the following files:
    - M_hat.npy: Estimated shared structure.
    - U_hat.npy: Estimated eigenspace.
    - Lambda_hat.npy: Estimated eigenvalues.
    - perturbed_nodes.json: Per-group perturbed node indices.
    - B_hat_{group_id}.npy: Per-group perturbation matrices.
    - summary.txt: Human-readable summary.

    Args:
        result: NetworkAnalysisResult instance.
        output_dir: Directory to write output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "M_hat.npy"), result.M_hat)
    np.save(os.path.join(output_dir, "U_hat.npy"), result.U_hat)
    np.save(os.path.join(output_dir, "Lambda_hat.npy"), result.Lambda_hat)

    nodes_serializable = {
        k: v.tolist() for k, v in result.perturbed_nodes.items()
    }
    with open(os.path.join(output_dir, "perturbed_nodes.json"), "w") as f:
        json.dump(nodes_serializable, f, indent=2)

    for group_id, B in result.B_hats.items():
        np.save(os.path.join(output_dir, f"B_hat_{group_id}.npy"), B)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(result.summary())
