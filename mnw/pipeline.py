"""
Multiple Weighted Network Analysis Pipeline.

Implements Algorithm 1 from Yan & Levin (2025, arXiv:2506.15915):
  1. Spectral initialization from control-group networks.
  2. Low-coherence row filtering.
  3. Residual computation for treatment-group networks.
  4. Node-level support recovery via SDP or Group Lasso.
  5. Debiased refinement of the shared low-rank structure.
"""

from __future__ import annotations

import logging
import warnings
from typing import Union

import numpy as np

from mnw.spectral import rank_r_approximation
from mnw.support_recovery import recover_support_sdp, recover_support_glasso
from mnw.refinement import debiased_estimate
from mnw.results import NetworkAnalysisResult
from mnw.utils import validate_inputs

logger = logging.getLogger(__name__)

_SUPPORT_METHODS = {
    "sdp": recover_support_sdp,
    "glasso": recover_support_glasso,
}


class MultipleNetworkPipeline:
    """
    Analyze multiple weighted networks with shared low-rank structure
    and node-sparse group-specific perturbations.

    Given control-group networks (baseline, no perturbation) and
    treatment-group networks (with potential node-level perturbations),
    this pipeline:
      - Estimates the shared low-rank structure M*.
      - Identifies which nodes are perturbed in each treatment network.
      - Estimates the perturbation matrices B*.

    Args:
        rank: Rank of the shared low-rank structure (r).
        support_size: Number of perturbed nodes to detect (m).
            If a dict, maps group IDs to per-group support sizes.
        support_method: Method for support recovery.
            'sdp' (default) uses semidefinite programming (requires MOSEK).
            'glasso' uses Group Lasso via ADMM.
        coherence_threshold: Threshold C for filtering high-coherence rows.
            Rows of U with ||U_i|| > C * n^{-1/4} are excluded from
            support recovery. If None, defaults to a data-driven value.
        verbose: If True, log progress messages.

    Example:
        >>> import numpy as np
        >>> from mnw import MultipleNetworkPipeline
        >>>
        >>> # Y_control: list of control-group adjacency matrices
        >>> # Y_treatment: dict of treatment-group adjacency matrices
        >>> pipeline = MultipleNetworkPipeline(rank=3, support_size=10)
        >>> result = pipeline.fit(
        ...     Y_control=[ctrl_matrix_1, ctrl_matrix_2],
        ...     Y_treatment={"patient_A": [treat_matrix_A]},
        ... )
        >>> print(result.summary())
        >>> result.save("output/")
    """

    def __init__(
        self,
        rank: int,
        support_size: Union[int, dict[str, int]],
        support_method: str = "sdp",
        coherence_threshold: float = None,
        verbose: bool = True,
    ):
        if support_method not in _SUPPORT_METHODS:
            raise ValueError(
                f"Unknown support_method '{support_method}'. "
                f"Choose from {list(_SUPPORT_METHODS.keys())}."
            )

        self.rank = rank
        self.support_size = support_size
        self.support_method = support_method
        self.coherence_threshold = coherence_threshold
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

    def fit(
        self,
        Y_control: list[np.ndarray],
        Y_treatment: dict[str, list[np.ndarray]],
    ) -> NetworkAnalysisResult:
        """
        Run the full analysis pipeline (Algorithm 1).

        Args:
            Y_control: List of n x n symmetric matrices from the control
                group (group G0). These are assumed to contain only the
                shared structure plus noise.
            Y_treatment: Dict mapping group/subject IDs to lists of n x n
                symmetric matrices from treatment groups (group G1).
                Each treatment network may contain node-level perturbations.

        Returns:
            NetworkAnalysisResult containing all estimated quantities.
        """
        n = validate_inputs(Y_control, Y_treatment)
        self._log(f"Input validated: n={n}, {len(Y_control)} control, "
                  f"{sum(len(v) for v in Y_treatment.values())} treatment matrices.")

        return self._run_stages(Y_control, Y_treatment, n)

    def _run_stages(self, Y_control, Y_treatment, n):
        """Execute all pipeline stages with numerical warnings suppressed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self._run_stages_inner(Y_control, Y_treatment, n)

    def _run_stages_inner(self, Y_control, Y_treatment, n):
        # Stage A0: spectral initialization
        M0, U, Lambda = self._spectral_init(Y_control)
        self._log(f"Spectral init complete. Eigenvalues: "
                  f"{np.array2string(Lambda, precision=3)}")

        # Low-coherence row selection
        cal_U, excluded = self._select_low_coherence_rows(U, n)
        n_tilde = len(cal_U)
        self._log(f"Coherence filter: {n_tilde}/{n} rows retained, "
                  f"{len(excluded)} high-coherence nodes excluded.")

        # Stage A1: support recovery per treatment group
        perturbed_nodes = {}
        B_hats = {}
        recover_fn = _SUPPORT_METHODS[self.support_method]

        for group_id, Y_list in Y_treatment.items():
            m = self._get_support_size(group_id)
            self._log(f"Recovering support for '{group_id}' (m={m})...")

            Y_residuals = self._compute_residuals(Y_list, M0, cal_U)

            if len(Y_residuals) == 1:
                Y_tilde = Y_residuals[0]
            else:
                Y_tilde = np.mean(Y_residuals, axis=0)

            I_hat_local = recover_fn(Y_tilde, m)
            I_hat_global = cal_U[I_hat_local]
            perturbed_nodes[group_id] = I_hat_global

            B_hat = np.zeros((n, n))
            Y_avg = np.mean(Y_list, axis=0)
            for idx in I_hat_global:
                B_hat[idx, :] = Y_avg[idx, :] - M0[idx, :]
                B_hat[:, idx] = Y_avg[:, idx] - M0[:, idx]
            B_hats[group_id] = B_hat

            self._log(f"  '{group_id}': {len(I_hat_global)} perturbed nodes found.")

        # Stage A2: debiased refinement
        M_hat, U_hat, Lambda_hat = self._debiased_refinement(
            Y_control, Y_treatment, perturbed_nodes, M0, U, Lambda
        )
        self._log("Debiased refinement complete.")

        return NetworkAnalysisResult(
            M_hat=M_hat,
            U_hat=U_hat,
            Lambda_hat=Lambda_hat,
            perturbed_nodes=perturbed_nodes,
            B_hats=B_hats,
            coherent_nodes=excluded,
            diagnostics={
                "spectral_init_eigenvalues": Lambda,
                "n_coherent_excluded": len(excluded),
            },
        )

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _spectral_init(self, Y_list):
        """Average control matrices and compute rank-r spectral estimate."""
        Y_avg = np.mean(Y_list, axis=0)
        M0, U, Lambda = rank_r_approximation(Y_avg, self.rank)
        return M0, U, Lambda

    def _select_low_coherence_rows(self, U, n):
        """
        Return indices of low-coherence rows and excluded high-coherence rows.
        """
        threshold = self.coherence_threshold
        if threshold is None:
            threshold = n ** (-0.25)

        row_norms = np.linalg.norm(U, axis=1)
        cal_U = np.where(row_norms <= threshold)[0]
        excluded = np.where(row_norms > threshold)[0]
        return cal_U, excluded

    def _compute_residuals(self, Y_list, M0, cal_U):
        """Subtract spectral init and restrict to low-coherence rows."""
        residuals = []
        for Y in Y_list:
            Y_sub = Y[np.ix_(cal_U, cal_U)]
            M0_sub = M0[np.ix_(cal_U, cal_U)]
            residuals.append(Y_sub - M0_sub)
        return residuals

    def _debiased_refinement(
        self, Y_control, Y_treatment, perturbed_nodes, M0, U, Lambda
    ):
        """
        Refine the shared structure estimate using all available observations,
        with perturbation supports zeroed out.

        Prioritizes control-group matrices for debiased estimation (they have
        no zeroed rows). Falls back to spectral averaging when fewer than 4
        clean control matrices are available.
        """
        n = M0.shape[0]

        cleaned_treatment = []
        for group_id, Y_list in Y_treatment.items():
            I_hat = perturbed_nodes[group_id]
            for Y in Y_list:
                Y_clean = Y.copy()
                Y_clean[I_hat, :] = 0.0
                Y_clean[:, I_hat] = 0.0
                cleaned_treatment.append(Y_clean)

        all_cleaned = [Y.copy() for Y in Y_control] + cleaned_treatment

        if len(Y_control) >= 4:
            try:
                U_hat, M_hat = debiased_estimate(
                    [Y.copy() for Y in Y_control[:4]], self.rank
                )
                Lambda_hat = np.diag(U_hat.T @ M_hat @ U_hat)
                return M_hat, U_hat, Lambda_hat
            except Exception as e:
                self._log(f"Debiased refinement failed ({e}), falling back to spectral.")

        Y_avg = np.mean(all_cleaned, axis=0)
        M_hat, U_hat, Lambda_hat = rank_r_approximation(Y_avg, self.rank)
        return M_hat, U_hat, Lambda_hat

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_support_size(self, group_id: str) -> int:
        """Resolve support size for a given group."""
        if isinstance(self.support_size, dict):
            if group_id not in self.support_size:
                raise ValueError(
                    f"support_size dict missing key '{group_id}'. "
                    f"Available keys: {list(self.support_size.keys())}"
                )
            return self.support_size[group_id]
        return self.support_size

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)
