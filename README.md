<p align="center">
  <img src="paper/jacobi_forcing_logo.jpeg" alt="Jacobi Forcing" width="220" align="center">
</p>

<div align="center"><h1>&nbsp;Fast and Accurate Causal Parallel Decoding using Jacobi Forcing</h1></div>

<p align="center">
| <a href="http://arxiv.org/abs/XXXX.XXXXX"><b>Paper</b></a> | <a href="http://44.199.207.98:45109/blogs/jacobi-forcing/"><b>Blog</b></a> | <a href="https://github.com/hao-ai-lab/JacobiForcing"><b>Code</b></a> | <a href="http://huggingface.co/JacobiForcing/xxx"><b>Weights</b></a> |
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://flat.badgen.net/github/license/hao-ai-lab/JacobiForcing?label=License" alt="License">
  </a>
  <a href="https://github.com/hao-ai-lab/JacobiForcing/issues">
    <img src="https://flat.badgen.net/badge/Maintained%3F/yes/green" alt="Maintenance">
  </a>
  <a href="https://github.com/hao-ai-lab/JacobiForcing/pulls">
    <img src="https://flat.badgen.net/badge/Contributions/welcome/brightgreen" alt="Contributions welcome">
  </a>
  <a href="http://huggingface.co/JacobiForcing/xxx">
    <img src="https://flat.badgen.net/badge/Weights/JacobiForcing/yellow" alt="Weights">
  </a>
</p>

##

**Jacobi Forcing** is a training technique that converts standard autoregressive (AR) LLMs into **native causal parallel decoders**. It enables **fast multiblock parallel decoding** (Jacobi-style refinement, left-to-right) while keeping the **AR backbone + KV layout**—so it can often be deployed as a near drop-in replacement for existing AR checkpoints.

In the accompanying blog, we show **up to $\sim4\times$ wall-clock speedup** on coding and math tasks with **near-AR quality**, by decoding **multiple tokens per forward pass** and leveraging **multiblock decoding + rejection recycling**.

<p align="center">
  <picture>
    <!-- fig1: main speedup plot -->
    <img src="assets/img/fig1_speedup.png" width="45%">
  </picture>
  <br/>
  <i>fig1: Throughput / speedup of Jacobi Forcing vs AR and strong baselines (placeholder).</i>
</p>

<p align="center">
  <picture>
    <!-- fig2: demo gif (HumanEval / GSM8K) -->
    <img src="assets/img/fig2_demo.gif" width="90%">
  </picture>
  <br/>
  <i>fig2: Demo of faster decoding with comparable outputs (placeholder).</i>
</p>

## Contents
- [News](#news-)
- [Introduction](#introduction)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Citation](#citation)

## News 🔥
- **[2025/12/04]** Blog released: *Jacobi Forcing: Enabling Fast Multiblock Parallel Decoding with AR-Level Quality*.
- **[TBD]** Paper on arXiv + checkpoints on Hugging Face (links above are placeholders—update when finalized).

## Introduction

**Why faster decoding?**  
AR decoding is high-quality but serial: one forward pass per token. Diffusion language models can decode many tokens in parallel, but typically require **non-causal objectives** and often **break KV-cache-friendly serving**.

**Jacobi Forcing** bridges this gap by training an AR model to behave like a diffusion-style decoder **while staying causal**:
- **Causal, left-to-right** generation with **KV-cache reuse**
- **Parallel token updates** via **Jacobi-style refinement**
- **Multiblock decoding** for higher GPU utilization
- **Rejection recycling** to avoid wasting stable n-grams

<p align="center">
  <picture>
    <!-- fig3: multiblock + rejection recycling schematic -->
    <img src="assets/img/fig3_multiblock_recycling.png" width="90%">
  </picture>
  <br/>
  <i>fig3: Multiblock decoding + rejection recycling (placeholder).</i>
</p>

## Installation

> This section is intentionally minimal and conventional—adjust to match your repo structure.

1. Environment setup:
```bash
conda create -n jacobi_forcing python=3.10 -y
conda activate jacobi_forcing
