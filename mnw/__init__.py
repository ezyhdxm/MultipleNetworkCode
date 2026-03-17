"""
mnw: Multiple Weighted Network Analysis

Estimate shared low-rank structure and identify node-level perturbations
across multiple weighted networks observed under different conditions.

Reference:
    Yan & Levin (2025). "Estimating Multiple Weighted Networks with
    Node-Sparse Differences and Shared Low-Rank Structure."
    arXiv:2506.15915. https://arxiv.org/abs/2506.15915
"""

from mnw.pipeline import MultipleNetworkPipeline
from mnw.results import NetworkAnalysisResult

__all__ = ["MultipleNetworkPipeline", "NetworkAnalysisResult"]
