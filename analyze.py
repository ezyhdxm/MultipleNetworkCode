#!/usr/bin/env python3
"""
Command-line interface for multiple weighted network analysis.

Usage:
    python analyze.py \\
        --control data/ctrl_1.npy data/ctrl_2.npy \\
        --treatment data/treat_A.npy data/treat_B.npy \\
        --rank 3 --support-size 10 \\
        --output results/

    python analyze.py --config analysis_config.json --output results/

See README.md for full documentation.
"""

import argparse
import sys

import numpy as np

from mnw import MultipleNetworkPipeline
from mnw.io import load_matrices, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multiple weighted networks: identify node-level "
                    "perturbations and estimate shared low-rank structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file (overrides other flags).",
    )
    parser.add_argument(
        "--control", nargs="+", type=str, default=None,
        help="Paths to control-group adjacency matrices (.npy or .csv).",
    )
    parser.add_argument(
        "--treatment", nargs="+", type=str, default=None,
        help="Paths to treatment-group adjacency matrices (.npy or .csv).",
    )
    parser.add_argument("--rank", type=int, help="Rank of shared structure.")
    parser.add_argument("--support-size", type=int, help="Number of perturbed nodes.")
    parser.add_argument(
        "--method", type=str, default="sdp", choices=["sdp", "glasso"],
        help="Support recovery method (default: sdp).",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory (default: results/).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate and save visualization plots.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        rank = config["rank"]
        support_size = config["support_size"]
        method = config.get("support_method", "sdp")

        Y_control = load_matrices(config["control"])

        Y_treatment = {}
        treatment_cfg = config["treatment"]
        if isinstance(treatment_cfg, dict):
            for gid, paths in treatment_cfg.items():
                Y_treatment[gid] = load_matrices(paths)
        elif isinstance(treatment_cfg, list):
            for i, path in enumerate(treatment_cfg):
                Y_treatment[f"treatment_{i}"] = [load_matrices([path])[0]]
    else:
        if not args.control or not args.treatment:
            parser.error("Provide either --config or both --control and --treatment.")
        if args.rank is None or args.support_size is None:
            parser.error("--rank and --support-size are required without --config.")

        rank = args.rank
        support_size = args.support_size
        method = args.method

        Y_control = load_matrices(args.control)
        Y_treatment = {}
        for i, path in enumerate(args.treatment):
            Y_treatment[f"treatment_{i}"] = load_matrices([path])

    pipeline = MultipleNetworkPipeline(
        rank=rank,
        support_size=support_size,
        support_method=method,
        verbose=not args.quiet,
    )

    result = pipeline.fit(Y_control, Y_treatment)

    result.save(args.output)
    print(result.summary())

    if args.plot:
        import matplotlib.pyplot as plt
        import os

        fig_dir = os.path.join(args.output, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        ax = result.plot_shared_structure()
        ax.figure.savefig(os.path.join(fig_dir, "shared_structure.png"), dpi=150, bbox_inches="tight")
        plt.close()

        fig = result.plot_perturbations()
        fig.savefig(os.path.join(fig_dir, "perturbations.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Plots saved to {fig_dir}/")

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
