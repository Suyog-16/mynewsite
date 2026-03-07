---
title: "Vision-Language Model Alignment Research"
date: 2024-01-15
summary: "Investigating novel techniques for aligning visual encoders with large language models using minimal paired data. This research focuses on improving zero-shot performance in low-resource specialized domains."
weight: 1
tags: ["Deep Learning", "PyTorch", "Multimodal AI", "Research"]
author: "Suyog Ghimire"
showToc: false
hidemeta: false
draft: false
---

## Overview

In this project, I explore the challenges of grounding language models in visual perception. Current State-of-the-Art (SOTA) models often hallucinate objects or attributes when dealing with out-of-distribution images.

### Key Contributions
*   **Data Efficiency**: Developed a pipeline that requires 40% less annotated data for fine-tuning.
*   **Architecture**: Utilized a frozen CLIP visual encoder paired with a LLaMA-2 backbone, introducing a lightweight adapter layer.
*   **Evaluation**: Benchmarked on VQA v2 and GQA, achieving comparable performance to fully fine-tuned baselines.

### Technologies
*   **Frameworks**: PyTorch, HuggingFace Transformers
*   **Models**: CLIP, LLaMA-2, Vicuna
*   **Tools**: Weights & Biases for tracking experiments

[View Code on GitHub](https://github.com/Suyog-16)
