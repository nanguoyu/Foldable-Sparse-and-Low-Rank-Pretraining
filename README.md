# FOSL: A Foldable Sparse-and-Low-Rank method for Efficient LLM Pretraining

## Abstract

FOSL is a novel method for efficient neural network pre-training that reparameterizes linear layers by combining low-rank adaptation (LoRA) with a folded sparse structure. This approach achieves significant reductions in computational cost and memory usage while maintaining model expressivity through principled variance correction.

## Core Algorithm

### Mathematical Formulation

For a standard linear layer `Linear(d → m)` with input `x ∈ R^{...,d}` and output dimension `m`, fosl replaces it with:

```
y = α · LoRA_branch(x) + β · Sparse_branch(x) + bias
```

Where:
- `α = 0.7, β = 0.3` (mixing coefficients), could also be trainable in our paper by default
- `LoRA_branch(x) = B(φ(Ax)) · s`
- `Sparse_branch(x) = [base_out, foldable_out]`

### Key Components

#### 1. Low-Rank Branch (LoRA)
- **Matrices**: `A ∈ R^{d×r}, B ∈ R^{r×m}` where `r ≪ min(d,m)`
- **Activation**: Optional nonlinearity `φ` (default: SiLU)
- **Scaling**: `s = lora_alpha/r` (fixed) or trainable via `tanh`
- **Initialization**: `A` uses Kaiming uniform, `B` initialized to zeros

#### 2. Folded Sparse Branch
Given folding ratio `ρ ∈ (0,1]`:
- `m_base = m - ⌊ρm⌋` (truly computed channels)
- `m_fold = ⌊ρm⌋` (synthesized channels)

**Base computation**: `base_out_raw = x W_base` where `W_base ∈ R^{d×m_base}`

**Channel folding**: `foldable_out = base_out_raw[..., M]` where `M` is a random mask

**Variance correction**: Apply scaling factors to maintain proper variance when channels are reused


---

## Installation Instructions

You can set up your environment with `Conda` or `Mamaba`.

```bash
mamba create -n fosl python=3.11 -y
mamba install -n fosl pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba install -n fosl -c pytorch -c nvidia -c huggingface -c conda-forge pytorch-cuda=12.1 transformers datasets huggingface_hub accelerate pandas scipy tqdm pkg-config tokenizers scikit-learn -y
mamba install -n fosl -c conda-forge opencv pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba activate fosl
pip3 install torchopt ultralytics-thop wandb timm torchist==0.2.3 matplotlib torchopt "jsonargparse[signatures]>=4.37" psutil loguru bitsandbytes evaluate
pip3 install -U "huggingface_hub[cli]"
pip install --upgrade transformers
```

## Prepare dataset

You can run `Python download_full_c4.py --data_dir=~/data` to download full C4 dataset to `--data_dir`. In the following scripts, please use the same path to `data_dir`.

## FOSL on efficient pre-training

Assume you have a DGX A100 with 8XA100 GPUS, you can change the number of GPUS and GPU ids in each script.

To get the results shown in the fig.2 of our paper, please run the following. For other ablation study experiments, please change the `--mix_trainable`, `--mix_per_channel` `--init_lost`

```Bash
scripts/dgx/60m_fosl.sh
```

```Bash
scripts/dgx/130m_fosl.sh
```


```Bash
scripts/dgx/350m_fosl.sh
```

```Bash
scripts/dgx/1b_fosl.sh
```

## FOSL on efficient fine-tuning

Please run all the following scripts to get results of fine-tuning `FOSL` model on GLUE benchmark dataset.
```Bash
scripts/dgx/glue_easy.sh
```

```Bash
scripts/dgx/glue_medium.sh
```


```Bash
scripts/dgx/glue_hard.sh
```