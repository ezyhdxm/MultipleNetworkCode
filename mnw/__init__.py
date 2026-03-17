"""
mnw: Multiple Weighted Network Analysis

Estimate shared low-rank structure and identify node-level perturbations
across multiple weighted networks observed under different conditions.

Reference:
    Yan & Levin (2026). "Estimating Multiple Weighted Networks with
    Node-Sparse Differences and Shared Low-Rank Structure." JMLR.
"""

from mnw.pipeline import MultipleNetworkPipeline
from mnw.results import NetworkAnalysisResult

__all__ = ["MultipleNetworkPipeline", "NetworkAnalysisResult"]
