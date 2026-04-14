# Few-Shot Brain Tumor Segmentation with Meta-Learning

**CS7150 Deep Learning — Final Project**
**Yuzhe Li & Sherwin Vahidimowlavi | Northeastern University | Spring 2026**

Can we segment brain tumors with only 1–20 labeled examples? This project compares three few-shot adaptation strategies on the BraTS 2020 dataset: fine-tuning baseline, prototypical networks with gated attention, and MAML (first-order approximation).

---

## Key Results

| Method | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| Baseline Fine-tuning | 0.595 ± 0.137 | 0.732 ± 0.083 | 0.736 ± 0.078 | 0.735 ± 0.078 |
| **Prototypical Networks** | **0.799 ± 0.052** | 0.795 ± 0.086 | 0.797 ± 0.061 | **0.829 ± 0.059** |
| MAML (FOMAML) | 0.775 ± 0.070 | **0.800 ± 0.067** | **0.800 ± 0.061** | 0.823 ± 0.072 |

*Mean Tumor Dice Score (classes 1–3), 50 episodes per k value.*

Both meta-learning methods significantly outperform the baseline at every k, with the largest gap at k=1 (+0.204 for prototypical). Full supervision Dice: 0.737.

![Method Comparison](results/method_comparison.png)

---

## Context: Published Benchmarks

Direct comparison with published results is limited because most prior work on BraTS uses binary segmentation (tumor vs. background) rather than multi-subregion classification, and many use all 4 MRI modalities. Our task — 4-class segmentation from only FLAIR and T1CE — is strictly harder.

| Reference | Dataset | Task | Setup | Dice |
|---|---|---|---|---|
| **Ours (Prototypical)** | **BraTS 2020** | **4-class, 2ch** | **k=20, 50 eps** | **0.829** |
| **Ours (MAML)** | **BraTS 2020** | **4-class, 2ch** | **k=20, 50 eps** | **0.823** |
| **Ours (Baseline)** | **BraTS 2020** | **4-class, 2ch** | **Full supervision** | **0.737** |
| Balasundaram et al. 2023 | BraTS 2021 | Binary, 1ch (T2) | 1-shot | 0.834 |
| Kinagi et al. 2026 | BraTS 2021 | Binary, 2-way | 10-shot | 0.818 |
| Alsaleh et al. 2024 | TotalSeg | Organs, 3D | 5-shot MAML | 0.777–0.903 |
| MM-MSCA-AF 2025 | BraTS 2020 | 4-class, 4ch | Full supervision | 0.859 |
| Adv. nnU-Net 2024 | BraTS-Glioma | 3-region, 4ch | Full supervision | 0.830 |

Our prototypical network at k=20 (0.829) approaches full-supervision SOTA with 4 modalities (0.83–0.86) — using only 20 examples and 2 channels.

---

## Project Structure

```
UNet-FewShot/
├── configs/
│   ├── config.py              # Centralized hyperparameters and paths
│   ├── metrics.py             # Dice score, HD95, full evaluation loop
│   ├── model_utils.py         # Shared checkpoint loading
│   └── results_utils.py       # Shared JSON save/load/print utilities
├── data/
│   ├── dataset.py             # BraTSDataset — loads FLAIR + T1CE slices
│   ├── splits.py              # Consistent 70/20/10 train/val/test splits
│   ├── few_shot_sampler.py    # Episode sampling + k-shot fine-tune eval
│   ├── augmented_finetune.py  # Support set augmentation for all methods
│   └── synthetic_tumor_generator.py  # Synthetic tumor generation (experimental)
├── models/
│   ├── bu_net.py              # BUNet — ResNet34 U-Net baseline
│   ├── prototypical_segmentation.py  # Gated prototype-attention network
│   └── maml_segmentation.py   # FOMAML wrapper + trainer
├── training/
│   ├── trainer.py             # Supervised training loop (Dice loss)
│   └── prototypical_trainer.py # Episodic training with BN freeze
├── notebooks/
│   ├── 1_baseline_training.ipynb      # Train supervised U-Net
│   ├── 2_baseline_evaluation.ipynb    # K-shot fine-tuning evaluation
│   ├── 3_prototypical_training.ipynb  # Train + evaluate prototypical net
│   ├── 4_maml_training.ipynb          # Train + evaluate MAML
│   ├── 5_interpretability.ipynb       # Visualizations + method comparison
│   └── 6_augmentation_ablation.ipynb  # Support set augmentation experiment
├── results/                   # Saved JSON metrics + PNG figures
├── checkpoints/               # Model checkpoints (gitignored)
└── .gitignore
```

---

## Methods

### Baseline: Fine-Tuning
Pretrained ResNet34 U-Net (24.4M params). For each episode, deep-copy the model, fine-tune on k support slices for 10 gradient steps, then evaluate on query slices.

### Prototypical Networks (Gated Attention)
Compute per-class prototypes by masked averaging of encoder features from the support set. Generate cosine similarity attention maps and fuse with U-Net output:

```
output = UNet(query) × (1 + σ(gate) × attention)
```

The learnable gate starts at 0 (pure U-Net) and learns when to incorporate prototype guidance. BatchNorm layers are frozen during episodic training to preserve pretrained running statistics.

### MAML (First-Order Approximation)
First-order MAML (FOMAML) with 5 inner-loop SGD steps on the support set. Due to architectural constraints with SMP's U-Net, the outer loop trains on concatenated support+query data rather than backpropagating through the inner loop. This limitation is documented as a finding.

---

## Key Findings

1. **Preserving pretrained features > adaptation mechanism.** Both prototypical and MAML achieve ~0.80 Dice by avoiding destructive fine-tuning, while the baseline degrades to 0.595 at k=1 due to overfitting.

2. **Gated attention is critical.** Ungated prototype attention destroys the U-Net's calibration — Dice dropped to 0.248 without the gate. The gate learned to start conservative and rely primarily on U-Net features.

3. **Support set augmentation only helps when overfitting is the bottleneck.** Augmentation improved baseline k=1 by +0.098 but provided no benefit to prototypical/MAML, which already preserve pretrained features.

4. **Prototype resolution bottleneck.** Small tumor subregions (edema, enhancing) vanish when masks are downsampled to the encoder's 4×4 spatial resolution, producing empty prototypes for those classes.

5. **FOMAML limitation.** The first-order approximation collapses to standard training. Despite this, results are strong because the pretrained ImageNet backbone already provides a good initialization.

---

## Setup

### Requirements
```
torch >= 2.0
segmentation-models-pytorch
nibabel
opencv-python
scikit-learn
scipy
kagglehub
tqdm
matplotlib
```

### Installation
```bash
git clone https://github.com/sherwinvahidi/UNet-FewShot.git
cd UNet-FewShot
pip install -r requirements.txt
```

### Dataset
The BraTS 2020 dataset is downloaded automatically via KaggleHub on first run. You need a Kaggle account with the dataset accepted:
1. Go to [BraTS2020 on Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
2. Accept the dataset terms
3. Set up Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Running
Execute notebooks in order:
```
1_baseline_training.ipynb      → trains the U-Net baseline (~15 hours on MPS)
2_baseline_evaluation.ipynb    → k-shot fine-tuning evaluation (~2 hours)
3_prototypical_training.ipynb  → episodic training + evaluation (~1 hour)
4_maml_training.ipynb          → MAML training + evaluation (~1 hour)
5_interpretability.ipynb       → generates all comparison figures
6_augmentation_ablation.ipynb  → augmentation experiment (~3 hours)
```

All notebooks use `sys.path.append('..')` to import from the project root. Run them from inside the `notebooks/` directory.

---

## Results Gallery

### Method Comparison
![Method Comparison](results/method_comparison.png)

### Prediction Visualization
![Predictions](results/prediction_comparison.png)

### Per-Class Error Analysis
![Error Analysis](results/error_analysis.png)

### Augmentation Ablation
![Augmentation](results/augmentation_ablation_all.png)

---

## References

### Segmentation Architecture
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Rehman, M. U., et al. (2020). *BU-Net: Brain Tumor Segmentation Using Modified U-Net Architecture.* Electronics, 9(12), 2203.

### Meta-Learning
- Finn, C., Abbeel, P., & Levine, S. (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)
- Snell, J., Swersky, K., & Zemel, R. (2017). *Prototypical Networks for Few-Shot Learning.* NeurIPS. [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)

### Few-Shot Medical Image Segmentation
- Balasundaram, A., et al. (2023). *A Foreground Prototype-Based One-Shot Segmentation of Brain Tumors.* Diagnostics, 13(7), 1282. [DOI:10.3390/diagnostics13071282](https://doi.org/10.3390/diagnostics13071282)
- Pavithra, L. K., et al. (2023). *Brain Tumor Segmentation Using UNet-Few Shot Schematic Segmentation.* ITM Web of Conferences, 56, 04006.
- Kinagi, S., et al. (2026). *Brain Tumor Segmentation Using Few-Shot Learning.* Proceedings of CRM 2025, Springer. [DOI:10.1007/978-981-96-8126-6_28](https://doi.org/10.1007/978-981-96-8126-6_28)
- Alsaleh, A. M., et al. (2024). *Few-Shot Learning for Medical Image Segmentation Using 3D U-Net and MAML.* Diagnostics, 14(12), 1213. [DOI:10.3390/diagnostics14121213](https://doi.org/10.3390/diagnostics14121213)
- Ali, S., et al. (2022). *Meta-learning with implicit gradients in a few-shot setting for medical image segmentation.* Computers in Biology and Medicine, 143, 105227.

### Synthetic Data & Augmentation
- Hu, Q., et al. (2022). *Synthetic Tumors Make AI Segment Tumors Better.* NeurIPS. [arXiv:2210.14845](https://arxiv.org/abs/2210.14845)

### Full Supervision Benchmarks
- Enhanced MM-MSCA-AF (2025). *Enhanced Brain Tumor Segmentation Using Multi-Modal Multi-Scale Contextual Aggregation and Attention Fusion.* Scientific Reports. [DOI:10.1038/s41598-025-21255-4](https://doi.org/10.1038/s41598-025-21255-4)
- Ferreira, A., et al. (2024). *Brain Tumor Segmentation with Advanced nnU-Net: Pediatrics and Adults Tumors.* Brain Informatics.

### Dataset
- Menze, B. H., et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE TMI, 34(10), 1993–2024.
- Bakas, S., et al. (2017). *Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features.* Scientific Data, 4, 170117.