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
<img src="/images/rnn.webp" alt="Description" 
     style="width: 100%; height: 300px; object-fit: cover; border-radius: 8px;">

<center><i>Unfolding of Recurrent Neural Network</i></center>

*Credit: [dennybritz](https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-1/).*

The above diagram shows an RNN being unfolded/unrolled across time steps.We essentially have a full network for all the input sequence.For a sequence of 10 words/tokens the network would be unrolled into a 10 layer network one for each time step.This is fine for smaller sequences but for longer input sequences,the network becomes very deep

As a result during, backpropagation through time(BPTT) the gradient at each step would propagate backwards through many steps, this could cause the gradient to be extremely small causing a **Vanishing Gradient Problem** or the gradient could become very large which would cause the weights to be updated by large amount leading to numerical overflow causing a **Exploding Gradient Problem**.


### 2) Handling Long term dependencies

<center><b> "Shyam grew up in Nepal where he stayed until he was 20 years old therefore he speaks fluent _____." </b></center>
<br>

The correct completion is "Nepali" but for the model to correctly predict it. It needs to store the context word "Nepal" which appears 15-20 tokens before the target, diluted by many intermediate states. without the context of "Nepal" the model has no clue to guess the target

### 3) Sequential Processing

RNN's rely on sequential processing, meaning the <b>next output at each step depends on the previous step.</b> Mathematically, for seqeunce of $x_1, x_2, x_3, \ldots, x_n$

$$ h_{t} = f(h_{t-1},x_{t}) $$
where hidden state at time step $t$ depends upon previous step. we cant calculate $h_t$ without calculating $h_{t-1}$ first. Unlike CNNs or Transformers where many computations could occur simultaneously. RNNs cannot make use of parallel computations, they are stuck in a chain-like process.
<br>
<br>
Because of these limitations it was of no use when it came to handling large seqential data, this makes them slow,ineffective and limited. These problems are completely solved in modern architectures like Transformers

## Architecture Overview
Lets look at the Overall architecture of Transformer model and go into each section and dicuss that in detailed later in the blog

<img src="/images/transformers/transformer.png" alt="Description" 
     style="width: 100%; height: 800px; object-fit: cover; border-radius: 8px;">

[^1]<center><i>Transformer Architecture</i></center>

Transformer architecture consist of two main components: **encoder** and a **decoder**. The encoder takes in an input sequence(tokens) and converts it into numerical vectors that captures its meaning and relationship with each other. The decoder uses these encoded representations along with the tokens it has already generated to provide us with a probabilty distribution of possible next tokens at each time step.

We will get into the details of each and every step later in this blog.

## Positional Encoding

Unlike CNNs and RNNs, Transformers which uses self attention  is **Permutation Invaraint** meaning a it doesnt care about what order the tokens came in. It treats them as a bag of vector. For example a input sequence " **Ram pushed Hari**" and "**Hari pushed Ram**" would look the same to the transformer without positional information.

To address this we inject position information to the input sequence embeddings before feeding them to the transformer layers.

We want the model to identify which tokens are nearer and which are futher distant. **Note**: This 

Each input token is first mapped to a vector embedding of dimension 512(in the paper)
<img src="/images/transformers/vector.png" alt="Description" 
     style="width: 100%; height: 350px; object-fit: cover; border-radius: 8px;">

Above figure shows how each input token is mapped to a vector of a fixed dimension but it lacks positional information.Its solution lies in adding another vector **positional encoding vector** of same dimension(512) to the original vector embedding.

$$ E_{final}(pos,token) = E_{word}(token) + PE(pos)$$

Where:
<br>
- $E_{word}(token)$ is the 512 dimensional token embedding(as shown in the diagram)
<br>
- $PE(pos)$ is the positonal encoding vector for position $pos$

To calculate this we have positonal encoding fourmula as:
$$PE(pos, 2i) = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$$


$$PE(pos, 2i+1) = cos(\frac {pos}{10000^{\frac{2i}{d_{model}}}})$$

Where:
<br>
- $pos$ = token position(0,1,2....)
<br>
- $i$ = dimension index(0 to 255 for 512-dimension embeddings)
<br>
- $d_{model}$ = embedding dimesion(512)

So odd dimensions(0,2,4,...) takes the cosine fourmula whereas even dimension(1,3,5,..) takes sine fourmula

Now the question aries on how does adding the positional encodings generated by these trignometric fourmula help the model gain positional information on input sequence.

<img src="/images/transformers/positional_encoding.webp" alt="Description" 
     style="width: 100%; height: 350px; object-fit: cover; border-radius: 8px;">

*Credit: [kazemnejad's blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/).*

In the above chart were X-axis represents encoding dimensions and Y-axis represents different position in sequence,a particular cell's intensity is the sine/cosine value. 
we can clearly see a visible pattern in this chart the left side **High intensity changes** creates a checkerboard pattern and on the right side **Smooth intensity changes**creates a smooth gradients.

<center><h3>Why does this intensity pattern matters?</h3></center>

**Rapid color changes** means that model can distinguish between nearby positions and **smooth color changes** means model understands broad position relationships, with this each position gets a unique fingerprint.

Different frequencies of sine/cosine functions create the intensity variations you see, giving each position a unqiue pattenr across all dimensions





## Self-Attention Mechanism

## Multi-Head Attention
### Layer Normalization
### Residual Connection
### Linear Layer

## Toy Example


## Refrences
[^1]:Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need*. In Advances in Neural Information Processing Systems (NeurIPS).

