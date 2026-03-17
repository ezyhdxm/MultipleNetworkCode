#!/usr/bin/env python3
"""
Example: Analyze ABIDE brain connectivity data.

Downloads fMRI ROI time series from the ABIDE Preprocessed dataset
(autism vs. control subjects), computes correlation-based connectivity
matrices, and runs the multiple network analysis pipeline to identify
brain regions with group-specific connectivity differences.

Dataset: ABIDE Preprocessed (http://preprocessed-connectomes-project.org/abide/)
Atlas: CC200 (200 ROIs)

Usage:
    pip install nilearn
    python examples/run_abide.py
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnw import MultipleNetworkPipeline


def fetch_abide_timeseries(n_control=5, n_asd=5):
    """Download ABIDE ROI time series for control and ASD subjects."""
    from nilearn.datasets import fetch_abide_pcp

    print("Downloading ABIDE data (CC200 atlas, CPAC pipeline)...")
    print(f"  Fetching {n_control} control subjects...")
    ctrl = fetch_abide_pcp(
        DX_GROUP=2,
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_cc200"],
        n_subjects=n_control,
    )

    print(f"  Fetching {n_asd} ASD subjects...")
    asd = fetch_abide_pcp(
        DX_GROUP=1,
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_cc200"],
        n_subjects=n_asd,
    )

    return ctrl, asd


def timeseries_to_correlation(ts: np.ndarray) -> np.ndarray:
    """
    Convert an ROI time series matrix to a correlation-based connectivity matrix.

    Removes ROIs with zero variance (no signal), computes Pearson correlation,
    then re-embeds into the full ROI space.
    """
    n_timepoints, n_rois = ts.shape

    variances = np.var(ts, axis=0)
    valid = variances > 1e-10

    if valid.sum() < n_rois:
        ts_valid = ts[:, valid]
        corr_valid = np.corrcoef(ts_valid.T)
        corr = np.zeros((n_rois, n_rois))
        idx = np.where(valid)[0]
        corr[np.ix_(idx, idx)] = corr_valid
    else:
        corr = np.corrcoef(ts.T)

    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def main():
    print("=" * 65)
    print("  ABIDE Brain Connectivity Analysis")
    print("  Control vs. Autism Spectrum Disorder (ASD)")
    print("=" * 65)

    n_control, n_asd = 5, 5
    ctrl_data, asd_data = fetch_abide_timeseries(n_control, n_asd)

    print("\nComputing correlation matrices (200 x 200)...")
    Y_control = []
    for i, ts in enumerate(ctrl_data.rois_cc200):
        if isinstance(ts, str):
            ts = np.loadtxt(ts)
        corr = timeseries_to_correlation(ts)
        Y_control.append(corr)
        print(f"  Control {i+1}: time series {ts.shape} -> corr {corr.shape}")

    Y_asd = []
    for i, ts in enumerate(asd_data.rois_cc200):
        if isinstance(ts, str):
            ts = np.loadtxt(ts)
        corr = timeseries_to_correlation(ts)
        Y_asd.append(corr)
        print(f"  ASD {i+1}:     time series {ts.shape} -> corr {corr.shape}")

    n = Y_control[0].shape[0]
    print(f"\nNetwork size: {n} brain regions (CC200 atlas)")
    print(f"Control group: {len(Y_control)} subjects")
    print(f"ASD group:     {len(Y_asd)} subjects")

    rank = 5
    support_size = 15

    print(f"\nRunning pipeline (rank={rank}, support_size={support_size}, method=glasso)...")
    pipeline = MultipleNetworkPipeline(
        rank=rank,
        support_size=support_size,
        support_method="glasso",
        verbose=True,
    )

    result = pipeline.fit(
        Y_control=Y_control,
        Y_treatment={"ASD": Y_asd},
    )

    print("\n" + result.summary())

    perturbed = np.sort(result.perturbed_nodes["ASD"])
    print("\n  --- Interpretation ---")
    print(f"  {len(perturbed)} brain regions (CC200 ROI indices) show")
    print(f"  ASD-specific connectivity differences:")
    print(f"  ROI indices: {perturbed.tolist()}")

    B_asd = result.B_hats["ASD"]
    perturbed_strength = np.linalg.norm(B_asd[perturbed, :], axis=1)
    ranking = np.argsort(-perturbed_strength)
    print("\n  Ranked by perturbation strength:")
    for i, idx in enumerate(ranking):
        roi = perturbed[idx]
        strength = perturbed_strength[idx]
        print(f"    {i+1}. ROI {roi:3d}  (||B*_row|| = {strength:.4f})")

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results_abide",
    )
    result.save(output_dir)
    print(f"\n  Results saved to {output_dir}/")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
