# SRPO

SRPO is a research codebase for reinforcement learning of vision-language models with **separate credit assignment for visual perception and reasoning**. The main implementation lives in [`vision_sd/`](vision_sd/), built on the training infrastructure of Vision-SR1, EasyR1, and veRL.

The repository includes Qwen2.5-VL-3B/7B training recipes, GRPO and DAPO-style baselines, token-level credit assignment variants, ablations, checkpoint export, and multimodal evaluation scripts. Internal module names retain the `vision_sd` prefix.

## Method and implementation

The model generates a structured response:

```text
<perception>Visual evidence needed to solve the problem.</perception>
<reasoning>Reasoning grounded in that evidence.</reasoning>
\boxed{FINAL ANSWER}
```

The training pipeline combines:

- **Answer supervision:** [`answer_reward.py`](vision_sd/reward_function/answer_reward.py) scores final-answer correctness and response format. Its default reward is `0.95 * accuracy + 0.05 * format`.
- **Perception credit:** `simple_rlsd` compares token log probabilities under the original image and a weakened image, such as a partially masked image. The proportional variant assigns perception weights from these likelihood differences and normalizes them per sequence.
- **Reasoning credit:** the `rlsvr` path uses clipped, advantage-dependent likelihood-ratio weights. The `rollout_vs_qi` variant compares rollout log probabilities with a question-and-image-only rescore of the same reasoning tail.
- **Policy optimization:** GRPO-based updates, with recipes for DAPO-style asymmetric clipping, online filtering, and an optional entropy penalty.

[`CreditDistillTrainer`](vision_sd/ray_trainer.py) implements the credit assignment. External-teacher and privileged-context variants are also included; their behavior is selected through `VISION_SD_*` environment variables. The `simple_rlsd` mode does not require external Teacher A.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`vision_sd/`](vision_sd/) | SRPO training entry point, trainer, datasets, prompts, rewards, and experiment scripts |
| [`verl/`](verl/) | Ray/FSDP workers, vLLM rollout, policy losses, and shared training utilities |
| [`vision_r1/`](vision_r1/), [`vision_sr1/`](vision_sr1/) | Inherited accuracy-reward and self-reward training implementations |
| [`vision_r1_lora/`](vision_r1_lora/), [`vision_sr1_lora/`](vision_sr1_lora/) | Inherited LoRA variants |
| [`evaluation/`](evaluation/) | Response generation, answer judging, and accuracy reporting |
| [`scripts/model_merger.py`](scripts/model_merger.py) | Export FSDP actor checkpoints to Hugging Face format |
| [`LLaMA-Factory-Cold-Start/`](LLaMA-Factory-Cold-Start/) | Inherited supervised fine-tuning code |
| [`examples/`](examples/) | Additional RL examples |

## Installation

Use Linux with NVIDIA GPUs and a CUDA-compatible PyTorch/vLLM installation. Training recipes generally target multiple GPUs; memory requirements depend on model size, sequence length, rollout count, and batch sizes.

```bash
git clone https://github.com/Messimanda/SRPO.git
cd SRPO
conda create -n srpo python=3.11 -y
conda activate srpo
```

Install a mutually compatible PyTorch, torchvision, vLLM, and FlashAttention stack for your CUDA environment, then install the repository:

```bash
pip install -e .
```

See [`requirements.txt`](requirements.txt) for dependencies and [`setup.sh`](setup.sh) for the original environment recipe. That script contains explicit package pins, system-package installation, plotting dependencies, and an interactive W&B login; review it before running it on your machine. The dependency list alone is not a tested environment lockfile.

## Data preparation

Prepare the model weights and datasets separately. The experiment scripts reference **ViRL39K** for training and **MMK12** for validation; the base YAML files retain upstream Vision-SR1 dataset defaults. Pass your own paths explicitly when reproducing a recipe.

The dataset loader supports Hugging Face dataset identifiers, local dataset directories, and Parquet/JSON/JSONL files. Append `@train` or `@test` to select a split.

| Field | Meaning |
| --- | --- |
| `problem` | Question or prompt |
| `answer` | Ground-truth answer used by the reward function |
| `images` | Image inputs accepted by the dataset loader |

Column names are configurable with `data.prompt_key`, `data.answer_key`, and `data.image_key`. Set `data.image_dir` to the image root when using relative image paths. The prompt template is [`perception_reasoning_answer.jinja`](vision_sd/format_prompt/perception_reasoning_answer.jinja).

## Training

Run commands from the repository root. **The shell recipes contain machine-specific paths.** Before launching, replace model, dataset, image-root, and checkpoint-output paths. In particular, changing `TRAIN_FILES` does not replace a hard-coded `data.image_dir` in a script.

### Single-node proportional perception + reasoning credit

After adjusting `data.image_dir` in the selected script:

```bash
TRAIN_FILES=/path/to/ViRL39K/39Krelease.parquet@train \
VAL_FILES=/path/to/MMK12@test \
TRAIN_N_GPUS=8 \
bash vision_sd/train_simple_rlsd_prop_rlsvr_3b.sh /path/to/Qwen2.5-VL-3B-Instruct
```

The corresponding 7B recipe is [`train_simple_rlsd_prop_rlsvr_7b.sh`](vision_sd/train_simple_rlsd_prop_rlsvr_7b.sh). These scripts use YAML configuration plus command-line overrides and export their credit-assignment settings before starting `python3 -m vision_sd.main`.

### Multi-node rollout-vs-QI recipe

The following recipes combine proportional perception credit, `rollout_vs_qi` reasoning credit, DAPO-style settings, and entropy penalty coefficient `0.03`:

- [3B recipe](vision_sd/train_simple_rlsd_prop_rlsvr_rollout_qi_3b_multinode_dapo_entpen0p03.sh)
- [7B recipe](vision_sd/train_simple_rlsd_prop_rlsvr_rollout_qi_7b_multinode_dapo_entpen0p03.sh)

The committed launchers use Ray with a head node and worker nodes. Set `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`, and `NPROC_PER_NODE` consistently across machines; launch the script on each node with its own rank. All nodes need matching environments and access to the same model, code, and data paths. These launchers stop existing Ray processes, so use nodes allocated to this job.

### Baselines and ablations

| Experiment | Example script |
| --- | --- |
| Pure GRPO | [`train_pure_grpo_3b.sh`](vision_sd/train_pure_grpo_3b.sh) |
| DAPO-style baseline | [`train_dapo_3b.sh`](vision_sd/train_dapo_3b.sh) |
| No perception distillation | [`train_simple_rlsd_prop_rlsvr_rollout_qi_3b_multinode_dapo_entpen0p03_no_percdistill.sh`](vision_sd/train_simple_rlsd_prop_rlsvr_rollout_qi_3b_multinode_dapo_entpen0p03_no_percdistill.sh) |
| Entropy penalty `0.1` | [`train_simple_rlsd_prop_rlsvr_rollout_qi_3b_multinode_dapo_entpen0p1.sh`](vision_sd/train_simple_rlsd_prop_rlsvr_rollout_qi_3b_multinode_dapo_entpen0p1.sh) |
| External Teacher A | [`train_external_teacher_a_virl39k.sh`](vision_sd/train_external_teacher_a_virl39k.sh) |

Check the selected script for its exact settings: recipes differ in credit assignment, KL regularization, filtering, and optimization parameters. Checkpoints are written to `trainer.save_checkpoint_path`; configure logging through `trainer.logger`.

## Checkpoint export

Export a specific actor checkpoint:

```bash
python scripts/model_merger.py \
  --local_dir /path/to/checkpoint/global_step_100/actor
```

Use the resulting `actor/huggingface` directory for inference or evaluation.

## Evaluation

[`eval_vision_sd.sh`](evaluation/full_rl/eval_vision_sd.sh) generates responses, extracts boxed answers, runs an LLM judge, and prints accuracy summaries.

Before running it, update `LOCAL_EVAL_ROOT`, `LOCAL_TRAIN_DATASET`, and the dataset list inside the script. The placeholder training dataset is required by this evaluation entry point even in validation-only mode. Set the GPU count to match your environment.

```bash
# Evaluate an exported model with a local judge.
bash evaluation/full_rl/eval_vision_sd.sh \
  /path/to/exported/huggingface \
  /path/to/Qwen2.5-VL-3B-Instruct \
  /path/to/judge-model
```

The script also accepts a `global_step_*` checkpoint directory as its first argument. Its benchmark list includes MMStar, MM-Vet, VisNumBench, MMMU/MMMU-Pro, HallusionBench, MMSI, MathVerse, MathVision, MathVista, and RealWorldQA, among others. Dataset preparation and judge weights are required separately.

Responses are stored under `evaluation/responses/`, with judgments under `evaluation/judgments/`. Existing nonempty response files are reused; use a fresh output location when changing the evaluated model or decoding settings.

## Acknowledgments and license

This repository builds on [Vision-SR1](https://github.com/zli12321/Vision-SR1), [EasyR1](https://github.com/hiyouga/EasyR1), [veRL](https://github.com/volcengine/verl), and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). The inherited training code and datasets should be credited to their original authors.

See [LICENSE](LICENSE) for the repository's Apache-2.0 license. Retain upstream notices and follow the applicable licenses for models and datasets used with this code.
