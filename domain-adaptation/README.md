# Task 2 - Unsupervised Domain Adaptation

This directory implements the PACS Photo/Art Painting/Cartoon to Sketch
transductive UDA protocol. The implementation is modular; the recommended
Kaggle presentation entry point is
`notebooks/02_task2_pacs_domain_adaptation.ipynb`.

## PACS setup

Attach PACS to the Kaggle notebook or set `PACS_ROOT` to a directory containing:

```text
photo/{dog,elephant,giraffe,guitar,horse,house,person}/
art_painting/{...}/
cartoon/{...}/
sketch/{...}/
```

The loader also recognizes layouts with an intermediate `images/`, `PACS/`, or
`PACS_Original/` directory. PACS contains four visual domains and seven shared
classes, as described by Li et al., *Deeper, Broader and Artier Domain
Generalization* (ICCV 2017).

## Running

The notebook is preferred for Kaggle and report presentation. Direct execution
uses two deliberately separate commands:

```bash
python train.py --pacs-root /path/to/PACS
# Lock all configurations and checkpoints before continuing.
python evaluate_final.py --pacs-root /path/to/PACS
```

`train.py` never constructs a labeled Sketch dataset. `evaluate_final.py` loads
Sketch labels only after verifying that all main and controlled-study
checkpoints exist. The Source-only checkpoint is written to
`../shared/checkpoints/source_only_pacs_sketch_seed6304.pt` for unchanged reuse
as the Task 3 ERM baseline. Source splits are persisted at
`../shared/splits/pacs_sketch_seed6304.json`.

## Structure

- `configs/`: locked common and method-specific settings.
- `models/`: ResNet-18 classifier and domain discriminator.
- `methods/`: MMD, gradient reversal, and CDAN conditioning.
- `training/`: common domain-balanced loaders and training loop.
- `evaluation/`: recognition metrics, domain probe, and class-level analysis.
- `train.py`: source-only and adaptation training without target labels.
- `evaluate_final.py`: final target evaluation and report-ready artifacts.

## External libraries and implementations

- ResNet-18, pretrained ImageNet weights, transforms, and data utilities use
  [torchvision](https://pytorch.org/vision/stable/).
- Metrics and the balanced logistic-regression domain probe use scikit-learn.
- Plots use Matplotlib and seaborn.
- Gradient reversal, multi-kernel MMD, and CDAN's undetached feature-probability
  outer product are implemented locally from the objectives specified in the
  assignment, following Ganin et al. (2016), Long et al. (2015), and Long et al.
  (2018).
