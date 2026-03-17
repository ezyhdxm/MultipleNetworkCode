# Example: Identifying Brain Connectivity Differences in Autism

This example applies the `mnw` pipeline to real neuroimaging data, comparing brain connectivity between individuals with **Autism Spectrum Disorder (ASD)** and typically developing controls. The goal is to identify a small set of brain regions whose connectivity patterns differ between the two groups --- precisely the kind of structured, low-rank-plus-sparse signal the model in [1] is designed to recover.

## Background for Non-Neuroscientists

A brain connectivity network is a weighted graph where each **node** is a brain region and each **edge weight** measures how strongly two regions co-activate over time. When a subject lies in an MRI scanner at rest (no task), the scanner records a time series of blood-oxygen signals for each region. Computing the Pearson correlation between every pair of time series yields a symmetric matrix --- the **functional connectivity matrix** --- which we treat as a weighted adjacency matrix.

The statistical question is: given a collection of such matrices from two groups (control and ASD), can we (a) estimate a **shared low-rank structure** common to all subjects, and (b) identify a **sparse set of nodes** whose connectivity is systematically perturbed in the ASD group? This is the problem addressed by the `mnw` pipeline [1].

## Setup

```bash
pip install -r requirements.txt
pip install nilearn   # for downloading ABIDE data
```

## Data

**Source.** We use the [ABIDE Preprocessed](http://preprocessed-connectomes-project.org/abide/) dataset [3], a publicly available collection of resting-state fMRI scans from individuals with ASD and matched controls across multiple sites.

**Brain parcellation.** The brain is divided into **200 regions of interest (ROIs)** using the CC200 atlas [2], a data-driven parcellation obtained via spatially constrained spectral clustering of voxel-level fMRI data. Each ROI is a contiguous cluster of voxels.

**From scans to matrices.** For each subject, the scanner records a time series of 196 measurements at each of the 200 ROIs. We compute the 200 x 200 Pearson correlation matrix across all ROI pairs, set the diagonal to zero, and use this as the subject's weighted adjacency matrix. This gives us one 200 x 200 symmetric matrix per subject.

**Sample.** From the University of Pittsburgh site, we use:
- **Control group**: 5 typically developing subjects
- **ASD group**: 5 individuals diagnosed with ASD

## Running the Analysis

```python
import numpy as np
from mnw import MultipleNetworkPipeline

# After computing 200x200 correlation matrices for each subject:
pipeline = MultipleNetworkPipeline(
    rank=5,              # rank of the shared low-rank structure
    support_size=15,     # number of perturbed nodes to detect
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

**Reading the output.** The spectral initialization finds a rank-5 approximation to the shared connectivity structure. The top eigenvalue (60.1) is well-separated from the rest, indicating a strong leading component. All 200 nodes pass the coherence filter (none have disproportionate leverage on the low-rank estimate). The support recovery step then selects 15 nodes whose row-norms in the estimated perturbation matrix are largest, and a debiased refinement step produces the final estimates.

### ROI-to-Region Mapping

The CC200 atlas defines ROIs by data-driven clustering [2], so its parcels do not carry anatomical names. To give the results a neuroscientific interpretation, we map each ROI to a named brain region in three steps:

1. **Atlas source.** We use the CC200 atlas distributed with the [C-PAC pipeline](https://github.com/FCP-INDI/C-PAC_templates) (`CC200.nii.gz`), the same file used during ABIDE preprocessing. This atlas is a 3D volume in standardized MNI space where each voxel is labeled 1--200.
2. **Center-of-mass.** For each selected ROI, we average the spatial coordinates of all voxels carrying that label, yielding a single (x, y, z) point in MNI space (a standardized coordinate system for the brain).
3. **Anatomical lookup.** We query the **AAL3** atlas [4] at that coordinate. AAL3 partitions the brain into 166 named regions --- including fine-grained thalamic nuclei, cerebellar lobules, and brainstem structures --- providing a human-readable label for each ROI.

The script [`examples/map_rois_to_regions.py`](map_rois_to_regions.py) automates this mapping. After running the pipeline, simply call:

```bash
# Map the perturbed nodes saved in results_abide/
python examples/map_rois_to_regions.py

# Or map specific ROI indices directly
python examples/map_rois_to_regions.py --rois 147 44 85 154 186
```

### Identified Brain Regions

The figure below shows where the identified regions sit in the brain. The left panel is a sagittal (midline cross-section) view revealing deep structures such as the thalamus, hippocampus, and cerebellum. The right panel is a ventral (bottom-up) view showing the underside of the brain, where the fusiform gyrus and temporal pole are visible.

![Brain regions with ASD-specific connectivity differences](brain_regions_asd.png)

The pipeline identified **15 nodes** (brain regions) with the largest perturbation between groups. The column **||B\*||** reports the row-wise L2-norm of the estimated perturbation matrix, measuring how much that node's entire connectivity profile differs between ASD and control groups. Higher values indicate stronger group differences.

| Rank | ROI | Brain Region (AAL3) | Side | MNI (x, y, z) | \|\|B\*\|\| |
|------|-----|---------------------|------|----------------|-------------|
| 1 | 147 | Thalamus (mediodorsal, medial) | R | (2, −4, 5) | 3.74 |
| 2 | 44 | Thalamus (mediodorsal, lateral) | L | (−8, −18, 11) | 2.93 |
| 3 | 85 | Cerebellum (lobule IX) | R | (9, −44, −37) | 2.74 |
| 4 | 154 | Hippocampus | R | (21, −12, −16) | 2.71 |
| 5 | 186 | Supplementary Motor Area | R | (14, 22, 60) | 2.61 |
| 6 | 17 | Thalamus (anterior pulvinar) | R | (12, −20, 10) | 2.61 |
| 7 | 189 | Cerebellum (lobule III) | R | (16, −35, −21) | 2.60 |
| 8 | 161 | Cerebellum (lobule IV/V) | L | (−9, −44, −24) | 2.59 |
| 9 | 26 | Fusiform Gyrus | L | (−31, −5, −33) | 2.47 |
| 10 | 31 | Temporal Pole (middle) | R | (44, 10, −36) | 2.46 |
| 11 | 197 | Fusiform Gyrus | R | (31, −1, −36) | 2.43 |
| 12 | 29 | Brainstem (locus coeruleus region) | midline | (3, −28, −36) | 2.43 |
| 13 | 86 | Hippocampus | R | (38, −13, −26) | 2.38 |
| 14 | 141 | Superior Occipital Gyrus | R | (18, −90, 23) | 2.37 |
| 15 | 196 | Precuneus | midline | (−4, −53, 57) | 2.33 |

### Interpretation

To assess whether these results are scientifically meaningful, we compare them against the existing ASD neuroimaging literature. A key question for any variable-selection method is: *do the selected variables correspond to known biology?* Below we summarize how each identified region relates to prior findings.

**Thalamus** (3 of 15 nodes, including the top 2 --- ranks 1, 2, 6):

The **thalamus** is a deep-brain relay station: nearly all sensory and motor signals pass through it on the way to the cerebral cortex. Think of it as the brain's central switchboard. The mediodorsal nucleus (ranks 1--2) connects to the prefrontal cortex and is involved in executive function, decision-making, and social cognition. The anterior pulvinar (rank 6) routes visual and multisensory information. That thalamic nodes show the strongest perturbation aligns with Nair et al. [5], who used combined fMRI and diffusion imaging to demonstrate impaired thalamus-to-cortex connectivity in children with ASD, concluding that disrupted thalamic relay may be upstream of many cortical-level ASD symptoms.

**Cerebellum** (3 nodes --- ranks 3, 7, 8):

The **cerebellum** ("little brain") sits at the back and bottom of the skull. Traditionally associated with motor coordination and balance, it is now known to participate in cognitive and social functions via loops connecting it to the cerebral cortex. Cerebellar abnormalities are among the earliest and most replicated neuroanatomical findings in ASD, dating back to Courchesne et al.'s [6] landmark report. D'Mello & Stoodley [7] review how specific cerebellar lobules map onto motor, cognitive, and social circuits --- all domains affected in ASD.

**Hippocampus** (2 nodes --- ranks 4, 13):

The **hippocampus**, located in the inner fold of the temporal lobe, is best known for its role in forming new memories. It also supports spatial navigation and social memory (e.g., remembering faces and past social interactions). Cooper et al. [10] showed that hippocampal functional connectivity is reduced during memory retrieval in adults with ASD, linking hippocampal network disruption to the episodic memory difficulties observed in the disorder.

**Fusiform Gyrus** (2 nodes --- ranks 9, 11):

The **fusiform gyrus**, on the bottom surface of the brain, contains the "fusiform face area" (FFA) --- a region that activates strongly when people view faces. Reduced FFA activation and connectivity during face processing is one of the most well-validated neuroimaging findings in ASD [8] and is thought to underlie the face-recognition and social-perception difficulties characteristic of the condition. The pipeline detected bilateral (left and right) fusiform perturbations.

**Temporal Pole** (1 node --- rank 10):

The **temporal pole**, at the front tip of the temporal lobe, is involved in social cognition --- particularly "theory of mind," the ability to infer what others are thinking or feeling [9]. Temporal pole dysfunction has been consistently linked to the social-communication difficulties that define ASD.

**Supplementary Motor Area** (1 node --- rank 5):

The **supplementary motor area (SMA)**, on the top of the brain near the midline, coordinates motor planning and sequencing. Motor difficulties are pervasive in ASD: a meta-analysis of 83 studies found a large effect size (d = 1.20) for motor coordination deficits across the autism spectrum [11].

**Brainstem** (1 node --- rank 12):

This parcel overlaps with the **locus coeruleus (LC)**, a small brainstem nucleus that produces norepinephrine and regulates arousal and attention. Atypical LC activity --- manifested as elevated resting pupil diameter and impaired attentional disengagement --- has been documented in children with ASD [12], suggesting a low-level arousal dysregulation that may cascade into higher-order attentional difficulties.

**Visual Cortex and Default Mode Network** (2 nodes --- ranks 14, 15):

The **superior occipital gyrus** (rank 14) is part of the visual processing hierarchy, consistent with the atypical visual perception frequently reported in ASD. The **precuneus** (rank 15) is a hub of the "default mode network" (DMN) --- a set of brain regions that activate during rest and self-referential thought. Assaf et al. [13] showed that reduced precuneus connectivity within the DMN correlates with social and communication symptom severity in ASD.

### Summary

Across all 15 selected nodes, every identified region has independent literature support linking it to ASD. The findings span subcortical relay structures (thalamus, brainstem), memory systems (hippocampus), social perception areas (fusiform gyrus, temporal pole), motor coordination regions (cerebellum, SMA), and higher-order association networks (precuneus/DMN, visual cortex). This convergence with established neuroscience suggests that the `mnw` pipeline is recovering scientifically meaningful signal rather than noise --- even with a small sample of 5 + 5 subjects.

### Saved Outputs

The pipeline saves all results to the output directory:

```
results_abide/
  M_hat.npy              # 200x200 estimated shared connectivity (M*)
  U_hat.npy              # Leading eigenvectors of M* (200 x 5)
  Lambda_hat.npy         # Leading eigenvalues of M* (5,)
  B_hat_ASD.npy          # 200x200 estimated perturbation matrix (B*)
  perturbed_nodes.json   # Indices of the 15 selected nodes
  summary.txt            # Human-readable summary
```

These can be loaded for further analysis:

```python
from mnw import NetworkAnalysisResult
result = NetworkAnalysisResult.load("results_abide/")
result.plot_shared_structure()
result.plot_perturbations(network_id="ASD")
```

## Caveats

This demonstration uses a small sample (5 + 5 subjects) from a single acquisition site for illustration purposes. A production analysis would use larger samples, explore sensitivity to the rank and support-size hyperparameters, and incorporate cross-validation or permutation testing for formal statistical inference.

## References

[1] Yan & Levin (2025). [Estimating Multiple Weighted Networks with Node-Sparse Differences and Shared Low-Rank Structure.](https://arxiv.org/abs/2506.15915) arXiv:2506.15915.

[2] Craddock et al. (2012). A whole brain fMRI atlas generated via spatially constrained spectral clustering. *Human Brain Mapping*, 33, 1914--1928.

[3] Di Martino et al. (2014). The Autism Brain Imaging Data Exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*, 19, 659--667.

[4] Rolls et al. (2020). Automated anatomical labelling atlas 3. *NeuroImage*, 206, 116189.

[5] Nair et al. (2013). Impaired thalamocortical connectivity in autism spectrum disorder: a study of functional and anatomical connectivity. *Brain*, 136(6), 1942--1955.

[6] Courchesne et al. (1988). Hypoplasia of cerebellar vermal lobules VI and VII in autism. *New England Journal of Medicine*, 318(21), 1349--1354.

[7] D'Mello & Stoodley (2015). Cerebro-cerebellar circuits in autism spectrum disorder. *Frontiers in Neuroscience*, 9, 408.

[8] Schultz (2005). Developmental deficits in social perception in autism: the role of the amygdala and fusiform face area. *International Journal of Developmental Neuroscience*, 23(2--3), 125--141.

[9] Olson et al. (2007). The enigmatic temporal pole: a review of findings on social and emotional processing. *Neuropsychologia*, 45(11), 2515--2524.

[10] Cooper et al. (2017). Reduced hippocampal functional connectivity during episodic memory retrieval in autism. *Cerebral Cortex*, 27(2), 888--902.

[11] Fournier et al. (2010). Motor coordination in autism spectrum disorders: a synthesis and meta-analysis. *Journal of Autism and Developmental Disorders*, 40, 1227--1240.

[12] Bast et al. (2021). Attentional disengagement and the locus coeruleus--norepinephrine system in children with autism spectrum disorder. *Frontiers in Integrative Neuroscience*, 15, 716447.

[13] Assaf et al. (2010). Abnormal functional connectivity of default mode sub-networks in autism spectrum disorder patients. *NeuroImage*, 53, 247--256.
