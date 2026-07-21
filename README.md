# [SHIFT: Steering Hidden Intermediates in Flow Transformers]()

**Nina Konovalova, Andrey Kuznetsov, Aibek Alanov**

<a href=""><img src="https://img.shields.io/badge/PDF-Paper-red" height=22.5></a>
<a href=""><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b.svg" height=22.5></a>
<a href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>
<a href=""><img src="https://img.shields.io/badge/Project-Website-blue" height=22.5></a>
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE.txt)

We propose SHIFT, a simple but effective and lightweight framework for concept removal in DiT diffusion models via targeted manipulation of intermediate activations at inference time, inspired by activation steering in large language models. SHIFT learns steering vectors that are dynamically applied to selected layers and timesteps to suppress unwanted visual concepts while preserving the prompt's remaining content and overall image quality. Beyond suppression, the same mechanism can shift generations into a desired style domain or bias samples toward adding or changing target objects.

Supported backbones: **FLUX.1-schnell / FLUX.1-dev** and **Stable Diffusion 3.5 Medium**.

## Nudity erase

![Nudity removal results](figures/nudity.png)

## Environment

```bash
git clone <your-repo-url>
cd SHIFT
pip3 install -r requirements.txt
pip3 install clean-fid torchmetrics
```

## Quickstart

![Method diagram](figures/main_scheme.png)

The pipeline has three steps: extract activations, compute steering vectors, apply them at inference. Edit paths and concept names in the scripts before running. All commands below assume the repository root.

### FLUX.1-schnell

#### Step 1 — Extract activations and text embeddings

```bash
bash scripts/get_vector.sh
```

This runs:
1. **`get_vector_1.py`** — generates images and saves intermediate block / attention activations.
2. **`get_encoding_vector.py`** — encodes positive / negative concept prompts into text embeddings (optional; enable the corresponding block in the script).

#### Step 2 — Calculate steering vectors

```bash
bash scripts/steering_calculate.sh
```

Supports activation-based vectors (`svm` / `diff`) and text-embedding-based vectors (`--method text` via `scripts/steering_calculate_txt.sh`).

#### Step 3 — Apply steering

```bash
bash scripts/remove_flux_schnell.sh
```

Additional example launchers: `scripts/add_flux_schnell.sh`, `scripts/add_flux_schnell_coco.sh`.

### Stable Diffusion 3.5

#### Step 1 — Extract activations and text embeddings

```bash
bash scripts/get_vector_sd3.sh
```

Uses **`get_vector_sd35.py`** (24 MM-DiT blocks) and **`get_encoding_vector.py`**.

#### Step 2 — Calculate steering vectors

```bash
bash scripts/steering_calculate_sd3.sh
```

#### Step 3 — Apply steering

```bash
bash scripts/apply_steering_sd3.sh
```

COCO evaluation launch: `scripts/apply_steering_sd3_coco.sh`.

## Repository layout

```text
src/
  models/          # FLUX and SD3.5 pipelines with steering hooks
  steering/        # extract → calculate → apply
  utils/           # shared helpers
scripts/           # example bash launchers
prompts_collection/
metrics/           # CLIP / DINO / FID / GroundingDINO eval scripts
figures/
```

## Citation

If our work assists your research, feel free to cite:

```bibtex
@article{konovalova2025shift,
  title     = {SHIFT: Steering Hidden Intermediates in Flow Transformers},
  author    = {Konovalova, Nina and Kuznetsov, Andrey and Alanov, Aibek},
  year      = {2025}
}
```
