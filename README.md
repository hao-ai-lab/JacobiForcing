<p align="center">
  <img src="paper/jacobi_forcing_logo.jpeg" alt="Jacobi Forcing" width="180" align="center">
</p>

<div align="center"><h1>&nbsp;Jacobi Forcing: Fast and Accurate Native Parallel Decoders</h1></div>

<!-- =========================
     Badges + Links
     ========================= -->
<p align="center">
  <a href="http://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://flat.badgen.net/badge/Paper/arXiv/red" alt="Paper">
  </a>
  <a href="http://44.199.207.98:45109/blogs/jacobi-forcing/">
    <img src="https://flat.badgen.net/badge/Blog/Jacobi%20Forcing/blue" alt="Blog">
  </a>
  <a href="http://huggingface.co/JacobiForcing/xxx">
    <img src="https://flat.badgen.net/badge/Weights/HuggingFace/yellow" alt="Weights">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://flat.badgen.net/badge/License/Apache--2.0/blue" alt="License">
  </a>
</p>


##

<p align="justify">
  <b>Jacobi Forcing</b> is a training technique that converts standard autoregressive (AR) LLMs into <b>native causal parallel decoders</b>.
  
  It enables <b>fast multiblock parallel decoding</b> (Jacobi-style refinement, left-to-right) while keeping the <b>AR backbone + KV layout</b>—so it can often be deployed as a near drop-in replacement for existing AR checkpoints.
</p>

<p align="justify">
  In the accompanying blog, we show <b>up to &sim;4&times; wall-clock speedup</b> on coding and math tasks with <b>near-AR quality</b>, by decoding <b>multiple tokens per forward pass</b> and leveraging <b>multiblock decoding + rejection recycling</b>.
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


## Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Citation](#citation)

## Introduction

### Why faster decoding?

<p align="justify">
  AR decoding is high-quality but serial: one forward pass per token. Diffusion language models can decode many tokens in parallel,<br/>
  but typically require <b>non-causal objectives</b> and often <b>break KV-cache-friendly serving</b>.
</p>

<p align="justify">
  <b>Jacobi Forcing</b> bridges this gap by training an AR model to behave like a diffusion-style decoder <b>while staying causal</b>:
</p>

<p align="justify">
  - <b>Causal, left-to-right</b> generation with <b>KV-cache reuse</b><br/>
  - <b>Parallel token updates</b> via <b>Jacobi-style refinement</b><br/>
  - <b>Multiblock decoding</b> for higher GPU utilization<br/>
  - <b>Rejection recycling</b> to avoid wasting stable n-grams
</p>

<p align="center">
  <picture>
    <img src="paper/multiblock_rejection_recycling.gif" width="90%" alt="Multiblock + rejection recycling" />
  </picture>
  <br/>
  <i>fig2: Illustration of multiblock Jacobi decoding with rejection recycling.</i>
</p>



## Installation



<p align="justify">
  <i>This section is demonstrative with path placeholders: adjust to match your repo structure.</i>
</p>

1. Environment setup:
```bash
conda create -n jacobi_forcing python=3.12 -y
conda activate jacobi_forcing
```

2. Clone this repository and build from source:
```bash
git clone https://github.com/hao-ai-lab/JacobiForcing.git
cd JacobiForcing
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Model Weights

### Base Models
| Size | Domain | HuggingFace Repo                 |
| ---- | ------ | -------------------------------- |
| 7B   | Code   | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| 7B   | Math   | `Qwen/Qwen2.5-Math-7B-Instruct`  |


### Jacobi Forcing Models
| Size | Domain | Data | HuggingFace Repo                    |
| ---- | ------ | ------ | ------------------------ |
| 7B   | Code | [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)  | [JacobiForcing_Coder_7B_v1](https://huggingface.co/JacobiForcing/JacobiForcing_Coder_7B_v1) |
| 7B   | Math | [OpenThoughts2](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M)  | [JacobiForcing_Coder_7B_v1](https://huggingface.co/JacobiForcing/JacobiForcing_Math_7B_v1)  |

## Usage

### Training

Jacobi Forcing training involves the following steps:

1. Collect Jacobi trajectories from a base AR model (intermediate drafts + fixed points).


2. Noise-conditioned training over long horizons to make drafts stable under Jacobi refinement.


3. Mix with a small AR loss to anchor generation quality.



### Inference

Jacobi Forcing decoding typically exposes knobs like:

- block size `n` (tokens updated in parallel)

- rejection recycling verification budget `pool_size`

- block count `K` (maximum blocks “in flight”)

- activation ratio `r`

Recommended starting point:

`n=64, K=2, pool_size=4, r=0.85`


### Evaluation

We evaluate baseline models' and Jacobi Forcing models' performance on HumanEval, MBPP, GSM8K and MATH following the settings in [evalchemy](https://github.com/mlfoundations/evalchemy).

## Citation

This is the official project repository for the following paper. If you find this repository helpful, Please kindly cite:

```
@article{hu2025jacobi-forcing,
    title = {Fast and Accurate Causal Parallel Decoding using Jacobi Forcing},
    author = {Hu, Lanxiang and Kou, Siqi and Fu, Yichao and Rajbhandari, Samyam and Rosing, Tajana and He, Yuxiong and Deng, Zhijie and Zhang, Hao},
    year = {2025},
    archivePrefix= {arXiv},
    primaryClass = {cs.CL},
}
```
