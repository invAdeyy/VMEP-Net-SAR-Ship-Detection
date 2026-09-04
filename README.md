# VSSM-RetinaNet for SAR Ship Detection on SSDD

This repository contains the official implementation of **VSSM-RetinaNet** (VMamba-based Backbone + HFSF-FPN) for SAR ship detection on the SSDD dataset, built upon the [MMDetection](https://github.com/open-mmlab/mmdetection) framework.

---
Status: Code refactoring in progress. Complete implementation and configurations will be updated synchronously soon.

## 📌 Repository Structure

```text
VSSM-RetinaNet-SSDD/
├── mmdet/
│   └── models/
│       ├── backbones/
│       │   └── vmamba_backbone.py    # Custom VMamba/VSSM backbone implementation
│       └── necks/
│           └── hfsf_fpn.py           # High-Frequency Feature Fusion FPN (HFSF-FPN)
├── configs/
│   └── vmamba_ssdd.py                # MMDetection config file for SSDD dataset
├── tools/                            # Visualization and log parsing utilities
│   ├── draw_plots.py                 # Accuracy vs. Efficiency bubble chart & convergence generator
│   └── generate_gt.py                # Batch Ground Truth (GT) visualization script
└── README.md
