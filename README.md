# VSSM-RetinaNet (VMEP-Net) for SAR Ship Detection on SSDD

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![MMDetection](https://img.shields.io/badge/MMDetection-3.x-blue.svg)](https://github.com/open-mmlab/mmdetection)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

This repository contains the official implementation of **VMEP-Net (VSSM-RetinaNet)** for SAR ship detection on the SSDD dataset, built upon the [MMDetection](https://github.com/open-mmlab/mmdetection) framework.

> 📢 **Status:** Code refactoring in progress. Complete implementation, baseline configurations, and pre-trained weights will be updated synchronously soon.

---

## 🌟 Highlights

- **VMamba Backbone**: Integrates Visual State Space Models (VSSM) to effectively capture long-range dependencies in SAR images with linear computational complexity.
- **HFSF-FPN**: High-Frequency Feature Fusion FPN designed to preserve rich target edge and texture details in complex marine backgrounds.
- **Full Benchmark Configs**: Includes complete training/testing configuration files for our model, ablation studies, and baseline algorithms (Faster R-CNN, DINO, etc.).

---

## 📌 Repository Structure

```text
VSSM-RetinaNet-SSDD/
├── mmdet/
│   └── models/
│       ├── backbones/
│       │   └── vmamba_backbone.py               # Custom VMamba/VSSM backbone implementation
│       └── necks/
│           └── hfsf_fpn.py                      # High-Frequency Feature Fusion FPN (HFSF-FPN)
├── configs/                                     # Experiment configurations
│   ├── ours/
│   │   └── retinanet_vmamba_100e_ssdd.py        # Proposed VMEP-Net configuration
│   ├── ablations/
│   │   └── retinanet_vmamba_panet_60e_ssdd.py   # Ablation study (with PANet neck)
│   └── baselines/                               # Baseline models for fair comparison
│       ├── retinanet_r50_fpn_60e_ssdd.py
│       ├── faster-rcnn_r50_fpn_60e_ssdd.py
│       └── dino_r50_60e_ssdd.py
└── README.md
