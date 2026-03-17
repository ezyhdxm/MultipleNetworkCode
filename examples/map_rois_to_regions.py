#!/usr/bin/env python3
"""
Map CC200 ROI indices to anatomical brain region names.

Given a set of perturbed node indices from the mnw pipeline, this script:
  1. Downloads the CC200 atlas used by the CPAC/ABIDE preprocessing pipeline.
  2. Computes the center-of-mass (MNI coordinates) of each selected ROI.
  3. Looks up the nearest AAL3 region at those coordinates.

Requirements:
    pip install nibabel nilearn numpy

Usage:
    python examples/map_rois_to_regions.py
    python examples/map_rois_to_regions.py --results-dir results_abide
    python examples/map_rois_to_regions.py --rois 147 44 85 154 186
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib


CC200_URL = (
    "https://github.com/FCP-INDI/C-PAC_templates/raw/main/"
    "atlases/label/Human/CC200.nii.gz"
)


def download_cc200(cache_dir=None):
    """Download the CPAC CC200 atlas and return the file path."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".mnw_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "CC200.nii.gz")
    if os.path.exists(path):
        return path

    print(f"Downloading CC200 atlas to {path} ...")
    import urllib.request
    urllib.request.urlretrieve(CC200_URL, path)
    return path


def load_aal3():
    """Load the AAL3 atlas via nilearn and return (data, affine_inv, info)."""
    from nilearn import datasets

    aal = datasets.fetch_atlas_aal()
    img = nib.load(aal["maps"])
    data = img.get_fdata()
    affine_inv = np.linalg.inv(img.affine)
    return data, affine_inv, aal


def lookup_aal3(mni_coord, aal_data, aal_affine_inv, aal_info, search_radius=7):
    """Return the AAL3 region name for a given MNI coordinate."""
    vox = np.round((aal_affine_inv @ np.append(mni_coord, 1))[:3]).astype(int)

    def _label_at(v):
        if all(0 <= v[i] < aal_data.shape[i] for i in range(3)):
            return int(aal_data[v[0], v[1], v[2]])
        return 0

    lbl = _label_at(vox)
    if lbl > 0:
        return aal_info["labels"][aal_info["indices"].index(str(lbl))]

    for r in range(1, search_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != r:
                        continue
                    lbl = _label_at(vox + np.array([dx, dy, dz]))
                    if lbl > 0:
                        return aal_info["labels"][aal_info["indices"].index(str(lbl))]

    return "Not in AAL3"


def map_rois(roi_indices, cc200_path=None):
    """
    Map a list of 0-indexed CC200 ROI indices to anatomical labels.

    Returns a list of dicts with keys:
        roi, label, mni_x, mni_y, mni_z, side, aal3_region
    """
    if cc200_path is None:
        cc200_path = download_cc200()

    atlas_img = nib.load(cc200_path)
    atlas_data = atlas_img.get_fdata()
    cc200_affine = atlas_img.affine

    aal_data, aal_affine_inv, aal_info = load_aal3()

    results = []
    for roi_idx in roi_indices:
        atlas_label = roi_idx + 1  # CC200 labels are 1-indexed
        mask = atlas_data == atlas_label
        if mask.sum() == 0:
            results.append({
                "roi": roi_idx, "aal3_region": "No voxels found",
                "mni_x": None, "mni_y": None, "mni_z": None, "side": "?"
            })
            continue

        center_vox = np.array(np.where(mask)).mean(axis=1)
        mni = (cc200_affine @ np.append(center_vox, 1))[:3]

        region = lookup_aal3(mni, aal_data, aal_affine_inv, aal_info)
        side = "L" if mni[0] < -5 else ("R" if mni[0] > 5 else "midline")

        results.append({
            "roi": roi_idx,
            "mni_x": round(float(mni[0]), 1),
            "mni_y": round(float(mni[1]), 1),
            "mni_z": round(float(mni[2]), 1),
            "side": side,
            "aal3_region": region,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Map CC200 ROI indices to AAL3 anatomical region names."
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Path to pipeline results directory containing perturbed_nodes.json. "
             "If omitted, uses --rois or the default results_abide/ directory.",
    )
    parser.add_argument(
        "--rois", type=int, nargs="+", default=None,
        help="Explicit list of 0-indexed ROI indices to map.",
    )
    parser.add_argument(
        "--cc200", default=None,
        help="Path to CC200.nii.gz atlas file. Downloaded automatically if omitted.",
    )
    args = parser.parse_args()

    B_hat = None
    if args.rois is not None:
        roi_indices = args.rois
        group_name = "user-specified"
    else:
        results_dir = args.results_dir
        if results_dir is None:
            results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results_abide",
            )
        nodes_path = os.path.join(results_dir, "perturbed_nodes.json")
        if not os.path.exists(nodes_path):
            print(f"Error: {nodes_path} not found. Run the pipeline first or use --rois.")
            sys.exit(1)

        with open(nodes_path) as f:
            nodes = json.load(f)

        group_name = list(nodes.keys())[0]
        roi_indices = nodes[group_name]

        B_path = os.path.join(results_dir, f"B_hat_{group_name}.npy")
        B_hat = np.load(B_path) if os.path.exists(B_path) else None

    print(f"Mapping {len(roi_indices)} ROIs for group '{group_name}'...\n")
    mapped = map_rois(roi_indices, cc200_path=args.cc200)

    if B_hat is not None:
        row_norms = np.linalg.norm(B_hat, axis=1)
        for m in mapped:
            m["norm"] = round(float(row_norms[m["roi"]]), 4)
        mapped.sort(key=lambda x: -x["norm"])
    else:
        B_hat = None

    header = f"{'Rank':>4}  {'ROI':>4}  {'MNI (x, y, z)':>20}  {'Side':>7}  "
    if B_hat is not None:
        header += f"{'||B*||':>8}  "
    header += "AAL3 Region"
    print(header)
    print("-" * len(header))

    for rank, m in enumerate(mapped, 1):
        mni_str = f"({m['mni_x']:5.1f}, {m['mni_y']:5.1f}, {m['mni_z']:5.1f})"
        line = f"{rank:4d}  {m['roi']:4d}  {mni_str:>20s}  {m['side']:>7s}  "
        if B_hat is not None:
            line += f"{m['norm']:8.4f}  "
        line += m["aal3_region"]
        print(line)


if __name__ == "__main__":
    main()
