---
title: "RSICD Remote Sensing Image Captioning"
date: 2026-03-30
summary: "An encoder-decoder image captioning study on RSICD comparing CNN+LSTM and CNN+Transformer approaches for satellite imagery understanding."
weight: 4
tags: ["Multimodal AI", "Remote Sensing", "PyTorch", "Image Captioning"]
author: "Suyog Ghimire"
showToc: false
hidemeta: false
draft: false
---

## Overview

This project generates natural language captions for satellite and aerial images using the RSICD dataset. It explores encoder-decoder approaches for converting visual scene content into meaningful textual descriptions.

### Key Highlights
*   Compares two modeling directions: CNN + LSTM and CNN + Transformer.
*   Uses RSICD data with around 10,000 remote sensing images and 5 captions per image.
*   Evaluates caption quality with BLEU and CIDEr style metrics.
*   CNN + LSTM setup reaches around BLEU 0.34 and CIDEr 0.81.

### Technologies
*   **Core**: Python, PyTorch
*   **Modeling**: Encoder-decoder caption generation
*   **Evaluation**: BLEU, CIDEr
*   **Dataset**: RSICD

[View Code on GitHub](https://github.com/Suyog-16/rsicd-image-captioning)
