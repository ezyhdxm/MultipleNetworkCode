"""
Result container for multiple network analysis.

Stores all outputs from the pipeline and provides methods for
summarization, visualization, and serialization.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NetworkAnalysisResult:
    """
    Complete results from the multiple weighted network analysis pipeline.

    Attributes:
        M_hat: Estimated shared low-rank structure (n x n).
        U_hat: Estimated eigenspace of the shared structure (n x r).
        Lambda_hat: Estimated eigenvalues of the shared structure (r,).
        perturbed_nodes: Mapping from treatment group ID to an array of
            node indices identified as having group-specific perturbations.
        B_hats: Mapping from treatment group ID to the estimated
            perturbation matrix (n x n) for that group.
        coherent_nodes: Node indices excluded by the coherence filter
            (high-leverage nodes whose rows in U_hat are too large).
        diagnostics: Additional information from intermediate pipeline
            stages, e.g. residual norms and spectral gaps.
    """

    M_hat: np.ndarray
    U_hat: np.ndarray
    Lambda_hat: np.ndarray
    perturbed_nodes: dict[str, np.ndarray]
    B_hats: dict[str, np.ndarray]
    coherent_nodes: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary of the analysis results."""
        n = self.M_hat.shape[0]
        r = len(self.Lambda_hat)
        lines = [
            "=" * 60,
            "  Multiple Weighted Network Analysis -- Results Summary",
            "=" * 60,
            "",
            f"  Network size (n):                {n}",
            f"  Estimated rank (r):              {r}",
            f"  Eigenvalues:                     {np.array2string(self.Lambda_hat, precision=3)}",
            f"  High-coherence nodes excluded:   {len(self.coherent_nodes)}",
            "",
            "  --- Perturbed Nodes by Group ---",
        ]

        for group_id, nodes in self.perturbed_nodes.items():
            sorted_nodes = np.sort(nodes)
            lines.append(f"  {group_id}: {len(nodes)} nodes")
            lines.append(f"    Indices: {sorted_nodes.tolist()}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def plot_shared_structure(self, node_labels=None, ax=None):
        """
        Plot a heatmap of the estimated shared structure M_hat.

        Args:
            node_labels: Optional list of node labels for axis ticks.
            ax: Optional matplotlib Axes. If None, creates a new figure.

        Returns:
            The matplotlib Axes object.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))

        sns.heatmap(
            self.M_hat,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            xticklabels=node_labels or False,
            yticklabels=node_labels or False,
        )
        ax.set_title("Estimated Shared Structure (M̂)")
        return ax

    def plot_perturbations(self, network_id: str = None, node_labels=None):
        """
        Plot heatmaps of estimated perturbation matrices B_hat.

        If network_id is specified, plots only that group. Otherwise, plots
        all treatment groups in a row of subplots.

        Args:
            network_id: Optional group ID to plot. If None, plots all.
            node_labels: Optional list of node labels for axis ticks.

        Returns:
            The matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        groups = (
            {network_id: self.B_hats[network_id]}
            if network_id
            else self.B_hats
        )
        n_groups = len(groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 6))
        if n_groups == 1:
            axes = [axes]

        for ax, (gid, B) in zip(axes, groups.items()):
            perturbed = self.perturbed_nodes.get(gid, np.array([]))
            mask = np.zeros_like(B, dtype=bool)
            if len(perturbed) > 0:
                mask[np.ix_(perturbed, perturbed)] = True

            sns.heatmap(
                B,
                ax=ax,
                cmap="RdBu_r",
                center=0,
                xticklabels=node_labels or False,
                yticklabels=node_labels or False,
            )
            ax.set_title(f"Perturbation B̂ — {gid}")

        fig.tight_layout()
        return fig

    def save(self, path: str) -> None:
        """
        Save results to a directory.

        Args:
            path: Output directory (created if it does not exist).
        """
        from mnw.io import save_results
        save_results(self, path)

    @classmethod
    def load(cls, path: str) -> NetworkAnalysisResult:
        """
        Load previously saved results from a directory.

        Args:
            path: Directory containing saved result files.

        Returns:
            Reconstructed NetworkAnalysisResult.
        """
        M_hat = np.load(os.path.join(path, "M_hat.npy"))
        U_hat = np.load(os.path.join(path, "U_hat.npy"))
        Lambda_hat = np.load(os.path.join(path, "Lambda_hat.npy"))

        with open(os.path.join(path, "perturbed_nodes.json"), "r") as f:
            nodes_raw = json.load(f)
        perturbed_nodes = {k: np.array(v) for k, v in nodes_raw.items()}

        B_hats = {}
        for fname in os.listdir(path):
            if fname.startswith("B_hat_") and fname.endswith(".npy"):
                gid = fname[len("B_hat_"):-len(".npy")]
                B_hats[gid] = np.load(os.path.join(path, fname))

        coherent_nodes = np.array([])
        coherent_path = os.path.join(path, "coherent_nodes.npy")
        if os.path.exists(coherent_path):
            coherent_nodes = np.load(coherent_path)

        return cls(
            M_hat=M_hat,
            U_hat=U_hat,
            Lambda_hat=Lambda_hat,
            perturbed_nodes=perturbed_nodes,
            B_hats=B_hats,
            coherent_nodes=coherent_nodes,
        )
