<p align="center">
  <img src="assets/jacobi_forcing_logo.jpeg" alt="Jacobi Forcing" width="180" align="center">
</p>

<div align="center"><h1>&nbsp;Jacobi Forcing: Fast and Accurate Causal Parallel Decoding</h1></div>

<!-- =========================
     Badges + Links
     ========================= -->
<p align="center">
  <a href="http://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://flat.badgen.net/badge/assets/arXiv/red" alt="Paper">
  </a>
  <a href="https://hao-ai-lab.github.io/blogs/jacobi-forcing/">
    <img src="https://flat.badgen.net/badge/Blog/Jacobi%20Forcing/blue" alt="Blog">
  </a>
  <a href="http://huggingface.co/JacobiForcing">
    <img src="https://flat.badgen.net/badge/Weights/HuggingFace/yellow" alt="Weights">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://flat.badgen.net/badge/License/Apache--2.0/blue" alt="License">
  </a>
</p>


##

*Jacobi Forcing* is a new training technique that converts LLMs into native casual parallel decoders. Jacobi forcing keeps the casual AR backbone and fixes the AR-to-diffusion mismatch by training the model to handle noisy future blocks along its own Jacobi decoding trajectories. 

*Jacobi Forcing* yields an AR model which behaves like a diffusion-style decoder—decoding multiple tokens per pass, but still from left to right—with up to $4.5\times$ higher tokens-per-forward and $4\times$ wall-clock speedup on coding and math tasks, while retraining near-AR generation quality. 

<p align="center">
  <picture>
    <img src="assets/ar_example_demo.gif" width="45%" alt="AR example demo (left)" />
    &nbsp;&nbsp;&nbsp;&nbsp;
    <img src="assets/jacobi_forcing_example_demo.gif" width="45%" alt="Jacobi Forcing example demo (right)" />
  </picture>
  <br/>
  <i>Demo of on average more than 4x speedup (181.8 TPS vs. 39.81 TPS) by Jacobi Forcing Model in comparison with the AR baseline (Qwen2.5-Coder-7B-Instruct) on coding sessions.</i>
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

AR decoding is high-quality but serial: one forward pass per token. Diffusion language models can decode many tokens in parallel, but typically require **non-causal objectives** and often **break KV-cache-friendly serving**.

Jacobi Forcing bridges this gap by training an AR model to behave like a diffusion-style decoder while staying causal:
  - **Causal, left-to-right generation** with KV-cache reuse
  - Parallel token updates within a block of size $n$ (via Jacobi decoding) and training makes such convergence faster
  - Multiblock decoding and Rejection recycling to exploit higher-quality draft with higher GPU utilization

<p align="center">
    <img src="assets/trajectory.jpeg" width="90%" alt="higher-quality draft" />
  <br/>
  <i>fig1: Illustration of higher quality drafts that emerge from Jacobi Forcing model.</i>
</p>


| Method        | Attention      | Parallelism                      | Training Cost | Single-model Decoding (no draft–verifier)   | Efficient KV Reuse        | Real Speedup | Generation Quality         |
|----------------------|------------------------|-----------------------------------------------|------------------------------------|-------------------------------------------|----------------------|-------------------------------|---------------------------|
| **AR**        | Causal | None                          | None                                            | No   |  Yes               |No            | Lossless            |
| **SD** | Causal                 | Yes             | No to Small: Draft model FT        | $\textcolor{red}{\text{No}}$ | Yes     | $<3.5\times$ | Lossless            | 
| **dLLMs**            | Non-causal    | Yes   | High: from scratch or heavy diffusion FT      | Yes | $\textcolor{red}{\text{No}}$ | $< 3\times$ | Low to near-AR quality |
| **Jacobi Forcing**   | Causal                 | Yes   | Small: noise-conditioned FT on trajectories | $\textcolor{green}{\text{Yes}}$ |  $\textcolor{green}{\text{Yes}}$   |  $\sim3-4\times$    | near-AR quality   |



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
| 7B   | Code   | [`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| 7B   | Math   | [`Qwen/Qwen2.5-Math-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)  |


### Jacobi Forcing Models
| Size | Domain | Data | HuggingFace Repo                    |
| ---- | ------ | ------ | ------------------------ |
| 7B   | Code | [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)  | [`JacobiForcing_Coder_7B_v1`](https://huggingface.co/JacobiForcing/JacobiForcing_Coder_7B_v1) |
| 7B   | Math | [OpenThoughts2](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M) (math split)  | [`JacobiForcing_Math_7B_v1`](https://huggingface.co/JacobiForcing/JacobiForcing_Math_7B_v1)  |

## Usage

### Training

Jacobi Forcing training involves the following steps:

#### Prepare training data

##### Choice A: download existing data from Huggingface.

```
git lfs clone https://huggingface.co/datasets/JacobiForcing/OpenCodeInstruct_training_data_n32w16
```

##### Choice B

- step 1: Collect Jacobi trajectories from a base AR model (intermediate states + fixed-point state for all $n-$token blocks).

```
# generate trajctories using customized models
bash generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.sh
```

If the target model is not Qwen2.5, first modify `generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.sh` to customize model path, trajectory data destimation, and input data path (you can download our length-bucketed input data from [this link for code](https://huggingface.co/datasets/JacobiForcing/OpenCodeInstruct_length_sorted) and [this link for math](https://huggingface.co/datasets/JacobiForcing/OpenThought2_length_bucketed)).

Then adapt from the script `generate_trajectory/generation/qwen2_modeling_jacobi_forcing_greedy.py` to make your target model compatible.

- step 2: training sequence packing and mapping noise schedule to training sequence.

```
python3 generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py \
    --input_path {trajectory_data_path} \
    --output_path {output_training_seq_path} \
    --n_token_seq_length {block_size} \
    --window_size {window_size} \
    --min_noisy_ratio 0 \
    --max_noisy_ratio 1.0 \
    --strategy "progressive"
```

<p align="center">
  <picture>
    <img src="assets/noise_schedule_and_sequence_packing.gif" width="60%" alt="noise schedule mapping" />
  </picture>
  <br/>
  <i>fig2: Illustration of the training sequence packing process with an example (linear progressive) noise schedule mapping.</i>
</p>


#### Noise-conditioned training over long horizons

```
cd /home/lah003/workspace/CLLM2/JacobiForcing
bash scripts/train/train_jacobi_forcing_coder_n32.sh
```

<p align="center">
    <img src="assets/noisy_context_attention_mask.jpeg" width="50%" alt="noise context training" />
  <br/>
  <i>fig3: Jacobi Forcing uses the attention implementation shown above. It allows logits from clean blocks and noisy blocks to be generated with single forward pass to calculate the progressive consistency loss and AR loss.</i>
</p>

### Inference

#### Multiblock Decoding

Jacobi Forcing decoding typically exposes knobs like:

- block size `n` (tokens updated in parallel)

- rejection recycling verification budget `pool_size`

- block count `K` (maximum blocks “in flight”)

- activation ratio `r`

<p align="center">
  <picture>
    <img src="assets/multiblock_rejection_recycling.gif" width="90%" alt="MR decoding" />
  </picture>
  <br/>
  <i>fig4: Illustration of multiblock Jacobi decoding with rejection recycling. High-quality n-grams from earlier iterations are reused as drafts.</i>
</p>

Recommended starting point (from our grid search):

`n=64, K=2, pool_size=4, r=0.85`

To run comprehensive grid search profiling for TPS speedup and TPF across different settings, run:

```
cd JacobiForcing
bash scripts/inference/scanning_hyperparameter_jacobi_decoding_mr.sh
```

To run a specific decoding setting with multiblock decoding and rejection recycling, run:
```
# vanilla Jacobi decoding
python3 JacobiForcing/jacobi_forcing_inference_humaneval.py

# with multiblock decoding and rejection recycling
python3 JacobiForcing/jacobi_forcing_inference_MR_humaneval.py
```


### Evaluation

#### Generation Quality Evaluation
We evaluate baseline models' and Jacobi Forcing models' performance on HumanEval, MBPP, GSM8K and MATH following the settings in [evalchemy](https://github.com/mlfoundations/evalchemy).


#### Performance Comparison

| Task      | Method           | Family      | Speedup | TPF | TPS | Acc / Solve $\uparrow$ |
|----------|--------------|------------|-----------|-------|-------|---------------|
| HumanEval| AR           | AR           | $1.00\times$    | 1.0  | 41.3  | 87.8%       |
|          | D2F          | dLLM         | $1.8\times$    |  2.5   | 73.2    | 54.3%   |
|          | Fast-dLLM    | dLLM         | $1.5\times$    |  1.8   | 60.0    | 53.0%   |
|          | dParallel    | dLLM-distilled  | $2.1\times$  |  2.9   | 88.5    | 54.3%   |
|          | EAGLE-3      | SD           |  $2.9\times$ | 6.4 | 120.7 | 68.9%* |
|          | HASS         | SD           |  $3.4\times$ | 5.5 | 138.7 | 61.6%* |
|          | CLLM*        | causal parallel | $2.5\times$  | 2.7  | 103.3 | 88.0%       |
|          | **Jacobi Forcing model**    | causal parallel | $3.9\times$    | 4.0  | 159.5 | 83.5%  |
|          | **Jacobi Forcing model (MR)** | causal parallel | **$4.0\times$** | 4.1  | 163.9 | 83.5% |
| GSM8K | AR               | AR           | $1.0\times$     | 1.0  | 41.8   | 92.4%      |
|       | D2F                | dLLM       | $2.2\times$   |  2.3  |  91.2   | 77.6%    |
|       | Fast-dLLM      | dLLM           | $1.2\times$       |  2.1 | 49.8   | 75.0%      |
|       | dParallel      | dLLM-distilled | $3.1\times$  |  3.8   |  128.0   | 82.9%   |
|       | EAGLE-3        | SD             | $3.3\times$  | 7.2   | 138.6  |  63.9%*           |
|       | HASS           | SD             | $3.1\times$  | 5.0   | 128.1  |  74.0%*           |
|       | CLLM*          | causal parallel    | $2.1\times$     | 2.3  | 86.8   | 92.2%      |
|       | **Jacobi Forcing model**        | causal parallel | $3.5\times$    | 3.7  | 146.1  | 91.4% |
|       | **Jacobi Forcing model (MR)**   |causal parallel | **$3.7\times$** | 4.0  | 154.9 | 91.4% |

<p align="justify">
  <i>*Here we report the strongest checkpoints released by the authors; in principle EAGLE-3 and HASS are lossless in comparison with greedy AR checkpoints if they were trained with the Qwen2.5-7B backbone. Note that SD has a worse acceptance length (TPF) to TPS conversion ratio due to other overheads in the algorithm like token drafting using draft head, tree-like verification overhead, feature merging from different layers etc.</i>
</p>

#### Performance Summary

Overall, Jacobi Forcing model consistently delivers **up to $3-4\times$ wall-clock speedup** on coding and math tasks with only minor accuracy changes versus greedy AR, while significantly outperforming both dLLMs and prior consistency-based parallel decoders in the accuracy–throughput tradeoff.


On a single B200 GPU with much higher FLOPs, the same Jacobi Forcing model with multiblock + rejection recycling can achieve an even more significant speedup at around 330 tokens/s (vs. around 80 tokens/s using AR), showing that the design continues to scale on newer accelerators.


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
