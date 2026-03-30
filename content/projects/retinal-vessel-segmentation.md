---
title: "Retinal Vessel Segmentation with U-Net"
date: 2026-03-30
summary: "A medical image segmentation project using a custom U-Net on DRIVE retinal images, with Dice and IoU based evaluation and Streamlit inference UI."
weight: 3
tags: ["Computer Vision", "Medical Imaging", "PyTorch", "Segmentation"]
author: "Suyog Ghimire"
showToc: false
hidemeta: false
draft: false
---

## Overview

This project performs blood vessel segmentation from retinal fundus images using a custom U-Net model. It is trained on the DRIVE dataset and focuses on robust vessel extraction that can support early diabetic retinopathy analysis workflows.

### Key Highlights
*   Custom U-Net implementation in PyTorch with Conv + BatchNorm + ReLU blocks.
*   Combined BCE + Dice loss for stable training on a small medical dataset.
*   Includes both CLI inference and a Streamlit app for interactive testing.
*   Achieves test performance around Dice 0.76 and IoU 0.62.

### Technologies
*   **Core**: Python, PyTorch
*   **Data Pipeline**: Albumentations, NumPy
*   **App Layer**: Streamlit
*   **Dataset**: DRIVE

[View Code on GitHub](https://github.com/Suyog-16/retinal-vessel-segmentation)
