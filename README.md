<p align="center">
  <img src="paper/jacobi_forcing_logo.jpeg" alt="Jacobi Forcing" width="180" align="center">
</p>

<div align="center"><h1>&nbsp;Jacobi Forcing: Fast and Accurate Native Parallel Decoders</h1></div>

<!-- =========================
     Badges + Links (ALL upfront)
     ========================= -->
<p align="center">
  <a href="http://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://flat.badgen.net/badge/Paper/arXiv/red" alt="Paper">
  </a>
  <a href="http://44.199.207.98:45109/blogs/jacobi-forcing/">
    <img src="https://flat.badgen.net/badge/Blog/Jacobi%20Forcing/blue" alt="Blog">
  </a>
  <a href="https://github.com/hao-ai-lab/JacobiForcing">
    <img src="https://flat.badgen.net/badge/Code/GitHub/black" alt="Code">
  </a>
  <a href="http://huggingface.co/JacobiForcing/xxx">
    <img src="https://flat.badgen.net/badge/Weights/HuggingFace/yellow" alt="Weights">
  </a>
  <br/>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://flat.badgen.net/badge/License/Apache--2.0/blue" alt="License">
  </a>
  <a href="https://github.com/hao-ai-lab/JacobiForcing/issues">
    <img src="https://flat.badgen.net/badge/Maintained%3F/yes/green" alt="Maintenance">
  </a>
  <a href="https://github.com/hao-ai-lab/JacobiForcing/pulls">
    <img src="https://flat.badgen.net/badge/Contributions/welcome/brightgreen" alt="Contributions welcome">
  </a>
</p>

<p align="center">
  <img src="paper/jacobi_forcing_logo.jpeg" alt="Jacobi Forcing" width="180" align="center">
</p>

<div align="center"><h1>&nbsp;Jacobi Forcing: Fast and Accurate Native Parallel Decoders</h1></div>

<div align="center">

##

<p align="center">
  <b>Jacobi Forcing</b> is a training technique that converts standard autoregressive (AR) LLMs into <b>native causal parallel decoders</b>.<br/>
  It enables <b>fast multiblock parallel decoding</b> (Jacobi-style refinement, left-to-right) while keeping the <b>AR backbone + KV layout</b>—so it can often be deployed as a near drop-in replacement for existing AR checkpoints.
</p>

<p align="center">
  In the accompanying blog, we show <b>up to &sim;4&times; wall-clock speedup</b> on coding and math tasks with <b>near-AR quality</b>,<br/>
  by decoding <b>multiple tokens per forward pass</b> and leveraging <b>multiblock decoding + rejection recycling</b>.
</p>

<p align="center">
  <picture>
    <img src="paper/ar_example_demo.gif" width="45%" alt="AR example demo (left)" />
    &nbsp;&nbsp;&nbsp;&nbsp;
    <img src="paper/jacobi_forcing_example_demo.gif" width="45%" alt="Jacobi Forcing example demo (right)" />
  </picture>
  <br/>
  <i>fig1: Demo of on average more than 4x speedup (181.8 TPS vs. 39.81 TPS) by Jacobi Forcing Model in comparison with the AR baseline (Qwen2.5-Coder-7B-Instruct) on coding sessions.</i>
</p>

</div>

<!-- =========================
     Centered Contents
     ========================= -->
<div align="center">

## Contents

<a href="#introduction">Introduction</a> •
<a href="#installation">Installation</a> •
<a href="#model-weights">Model Weights</a> •
<a href="#usage">Usage</a> •
<a href="#citation">Citation</a>

</div>

<div align="center">

## Introduction

<p align="center">
  <b>Why faster decoding?</b><br/>
  AR decoding is high-quality but serial: one forward pass per token. Diffusion language models can decode many tokens in parallel,<br/>
  but typically require <b>non-causal objectives</b> and often <b>break KV-cache-friendly serving</b>.
</p>

<p align="center">
  <b>Jacobi Forcing</b> bridges this gap by training an AR model to behave like a diffusion-style decoder <b>while staying causal</b>:
</p>

<p align="center">
  • <b>Causal, left-to-right</b> generation with <b>KV-cache reuse</b><br/>
  • <b>Parallel token updates</b> via <b>Jacobi-style refinement</b><br/>
  • <b>Multiblock decoding</b> for higher GPU utilization<br/>
  • <b>Rejection recycling</b> to avoid wasting stable n-grams
</p>

<p align="center">
  <picture>
    <img src="paper/multiblock_rejection_recycling.gif" width="90%" alt="Multiblock + rejection recycling" />
  </picture>
  <br/>
  <i>fig2: Illustration of multiblock Jacobi decoding with rejection recycling.</i>
</p>

</div>

<div align="center">

## Installation

</div>

<p align="center">
  <i>This section is intentionally minimal and conventional—adjust to match your repo structure.</i>
</p>

1. Environment setup:
```bash
conda create -n jacobi_forcing python=3.12 -y
conda activate jacobi_forcing
```