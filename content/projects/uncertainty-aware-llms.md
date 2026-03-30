---
title: "Uncertainty-Aware Fine-Tuning for LLMs"
date: 2026-03-30
summary: "A fine-tuning experiment on Llama 3.2 1B that reduces hallucinations by training the model to abstain with 'I don't know' when uncertain."
weight: 5
tags: ["LLM", "Fine-tuning", "Hallucination Reduction", "Hugging Face"]
author: "Suyog Ghimire"
showToc: false
hidemeta: false
draft: false
---

## Overview

This project studies whether a small LLM can be trained to express uncertainty instead of confidently hallucinating. The model is fine-tuned to prefer safe abstention when it lacks reliable knowledge.

### Key Highlights
*   Fine-tunes Llama 3.2 1B Instruct on an uncertainty-focused instruction dataset.
*   Demonstrates a qualitative reduction in hallucinated answers on unknown questions.
*   Shows an important trade-off: over-abstention, where the model may say "I don't know" even for answerable prompts.
*   Includes links to both the model and dataset on Hugging Face.

### Technologies
*   **Core**: Python, PyTorch, Hugging Face ecosystem
*   **Training Setup**: Google Colab (T4 GPU)
*   **Approach**: Instruction tuning focused on uncertainty-aware responses

[View Code on GitHub](https://github.com/Suyog-16/uncertainty-aware-llms)
