# Example: Identifying ASD-Specific Brain Connectivity Differences

This example demonstrates the `mnw` pipeline on real neuroimaging data from the [ABIDE Preprocessed](http://preprocessed-connectomes-project.org/abide/) dataset, comparing resting-state functional connectivity between individuals with Autism Spectrum Disorder (ASD) and typically developing controls.

## Setup

```bash
pip install -r requirements.txt
pip install nilearn   # for downloading ABIDE data
```

## Data

We use the **CC200 atlas** (Craddock et al., 2012), which parcellates the brain into **200 regions of interest (ROIs)**. For each subject, resting-state fMRI time series (196 timepoints) are extracted per ROI, and a **200 x 200 Pearson correlation matrix** serves as the weighted adjacency matrix for that subject's brain network.

From the ABIDE dataset (University of Pittsburgh site):
- **Control group**: 5 typically developing subjects
- **Treatment group**: 5 individuals diagnosed with ASD

## Running the Analysis

```python
import numpy as np
from mnw import MultipleNetworkPipeline

# (After computing 200x200 correlation matrices for each subject)
pipeline = MultipleNetworkPipeline(
    rank=5,              # shared low-rank structure has rank 5
    support_size=15,     # detect up to 15 perturbed brain regions
    support_method="glasso",
)

result = pipeline.fit(
    Y_control=Y_control,           # list of control correlation matrices
    Y_treatment={"ASD": Y_asd},    # list of ASD correlation matrices
)

print(result.summary())
```

The full runnable script is in [`examples/run_abide.py`](run_abide.py).

## Results

### Pipeline Output

```
Input validated: n=200, 5 control, 5 treatment matrices.
Spectral init complete. Eigenvalues: [60.081 14.94  10.591  7.902  7.318]
Coherence filter: 200/200 rows retained, 0 high-coherence nodes excluded.
Recovering support for 'ASD' (m=15)...
  'ASD': 15 perturbed nodes found.
Debiased refinement complete.
```

The spectral initialization reveals a clear low-rank structure in the shared brain connectivity, with a dominant first eigenvalue (60.1) and four additional components. All 200 ROIs passed the coherence filter, indicating the shared structure is well-distributed across brain regions.

### Identified Brain Regions

The pipeline identified **15 brain regions** whose functional connectivity patterns differ between ASD and control groups. Using the AAL atlas to map ROI indices to anatomical labels:

| Rank | ROI | Brain Region | Side | Perturbation Strength |
|------|-----|-------------|------|-----------------------|
| 1 | 147 | Middle Temporal Gyrus | L | 3.74 |
| 2 | 44 | Temporal Pole (middle) | R | 2.93 |
| 3 | 85 | Fusiform Gyrus | L | 2.74 |
| 4 | 154 | Middle Frontal Gyrus | R | 2.71 |
| 5 | 186 | Orbitofrontal Cortex (anterior) | R | 2.61 |
| 6 | 17 | Cerebellum Crus I | L | 2.61 |
| 7 | 189 | Postcentral Gyrus | R | 2.60 |
| 8 | 161 | Cerebellar Vermis (IV/V) | midline | 2.59 |
| 9 | 26 | Rolandic Operculum | R | 2.47 |
| 10 | 31 | Cerebellum Crus I | R | 2.46 |
| 11 | 29 | Inferior Frontal Gyrus (pars tri.) | L | 2.43 |
| 12 | 86 | Middle Occipital Gyrus | R | 2.38 |
| 13 | 141 | Middle Occipital Gyrus | L | 2.37 |

### Interpretation

The identified regions align with well-established findings from the ASD neuroimaging literature and span several brain systems known to be affected in autism:

**Social brain and language network** (4 regions, including the top 3):
- The **left Middle Temporal Gyrus** (rank 1) is a core node of the social brain and language comprehension network. Altered MTG connectivity is among the most consistently reported findings in ASD functional connectivity studies.
- The **right Temporal Pole** (rank 2) is critical for theory of mind --- the ability to infer others' mental states. Temporal pole dysfunction is robustly linked to social cognition deficits in ASD.
- The **left Fusiform Gyrus** (rank 3) houses the fusiform face area (FFA), one of the most well-validated neuroimaging findings in ASD. Reduced FFA activation and connectivity during face processing is a hallmark of the condition.
- The **left Inferior Frontal Gyrus** (rank 11) overlaps with Broca's area, essential for language production and a region implicated in the social communication deficits core to ASD.

**Cerebellum** (3 regions):
- **Bilateral Cerebellum Crus I** (ranks 6 and 10) and the **Cerebellar Vermis** (rank 8) are identified. Cerebellar abnormalities are among the oldest and most replicated neuroanatomical findings in ASD, dating back to the 1980s. Crus I has been specifically linked to higher-order social and cognitive functions.

**Executive function and decision-making** (2 regions):
- The **right Middle Frontal Gyrus** (rank 4, dorsolateral prefrontal cortex) supports executive function and working memory, both commonly impaired in ASD.
- The **right Orbitofrontal Cortex** (rank 5) is involved in reward processing and social decision-making, with well-documented alterations in ASD.

**Sensory processing** (3 regions):
- The **right Postcentral Gyrus** (rank 7, primary somatosensory cortex) and **bilateral Middle Occipital Gyrus** (ranks 12--13) relate to sensory processing. Sensory differences are now a DSM-5 diagnostic criterion for ASD.

### Saved Outputs

The pipeline saves all results to the output directory:

```
results_abide/
  M_hat.npy              # 200x200 estimated shared connectivity structure
  U_hat.npy              # Eigenspace of shared structure (200 x 5)
  Lambda_hat.npy         # Eigenvalues of shared structure (5,)
  B_hat_ASD.npy          # 200x200 estimated ASD perturbation matrix
  perturbed_nodes.json   # Indices of the 15 identified brain regions
  summary.txt            # Human-readable summary
```

These can be loaded back for further analysis:

```python
from mnw import NetworkAnalysisResult
result = NetworkAnalysisResult.load("results_abide/")
result.plot_shared_structure()
result.plot_perturbations(network_id="ASD")
```

## Caveats

This demonstration uses a small sample (5 + 5 subjects) from a single acquisition site for illustration purposes. A rigorous study would use larger samples, explore sensitivity to the rank and support size parameters, and incorporate cross-validation or permutation testing for statistical inference.

## References

- Yan & Levin (2025). [*Estimating Multiple Weighted Networks with Node-Sparse Differences and Shared Low-Rank Structure.*](https://arxiv.org/abs/2506.15915) arXiv:2506.15915.
- Craddock et al. (2012). A whole brain fMRI atlas generated via spatially constrained spectral clustering. *Human Brain Mapping*, 33, 1914--1928.
- Di Martino et al. (2014). The Autism Brain Imaging Data Exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*, 19, 659--667.
