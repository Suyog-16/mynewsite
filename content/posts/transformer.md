---
date : '2025-04-28T21:06:20+05:45'
draft : True
title : 'Understanding Transformer Architecture from First Principles: A Detailed Exploration'
math : True
author : "Suyog Ghimire"
---


## Introduction 

Transformer Architecture lies at the heart of modern AI models. Nearly all the state-of-the-art large language models(LLMs) like ChatGPT, LLaMa and  Gemini ,all of them are built upon the transformer architecture.The transformer architecture was introduced in the paper **Attention is all you need** [^1]. Although it is used everywhere these days,it was first introduced for the purpose of language translation but then it was quickly generalized to other task as well.Since then, due to its scalability and self-attention mechanism it has found itself in not just natural language processing(NLP) but also in computer vision,speech and multimodal AI systems.

In this blog, we will explore the internals of the Transformer Architecture ,with mathematical intuition and understand why it became the backbone of generative AI.

## Problems with RNN's and Sequence models

Before transformers, many natural language processing task were done using sequence models like recurrent neural network(RNN's).RNNs had major flaws due to which it was not good enough for most language-based applications.Lets discuss their flaws and how transformer massively improved upon them

### 1) Vanishing/Exploding Gradient problem
One of the major problems with RNNs were the vanishing/exploding gradients.
<img src="/images/rnn.webp" alt="rnn" width="1200">

<center><i>Unfolding of Recurrent Neural Network</i></center>

*Credit: [dennybritz](https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-1/).*

The above diagram shows an RNN being unfolded/unrolled across time steps.We essentially have a full network for all the input sequence.For a sequence of 10 words/tokens the network would be unrolled into a 10 layer network one for each time step.This is fine for smaller sequences but for longer input sequences,the network becomes very deep

As a result during, backpropagation through time(BPTT) the gradient at each step would propagate backwards through many steps, this could cause the gradient to be extremely small causing a **Vanishing Gradient Problem** or the gradient could become very large which would cause the weights to be updated by large amount leading to numerical overflow causing a **Exploding Gradient Problem**.


### 2) Handling Long term dependencies

<center><b> "Shyam grew up in Nepal where he stayed until he was 20 years old therefore he speaks fluent _____." </b></center>
<br>

The correct completion is "Nepali" but for the model to correctly predict it. It needs to store the context word "Nepal" which appears 15-20 tokens before the target, diluted by many intermediate states. without the context of "Nepal" the model has no clue to guess the target

### 3) Sequential Processing

















## Architecture Overview

## Positional Encoding

## Self-Attention Mechanism

## Multi-Head Attention
### Layer Normalization
### Residual Connection
### Linear Layer

## Toy Example


## Refrences
[^1]:Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need*. In Advances in Neural Information Processing Systems (NeurIPS).

