# Week 3 — Efficient Fine-Tuning & Quantization

After learning the fundamentals of Supervised Fine-Tuning in Week 2, this week focuses on making fine-tuning *efficient* — so large models can be trained on a single consumer GPU instead of a data center.

You'll begin by understanding quantization (why fewer bits still gives you a usable model), then progress through LoRA, DoRA, QLoRA, speed/memory optimization tricks, and finally model merging.

---

## Learning Objectives

By the end of this week, you'll be able to:

- Explain why quantization enables large models to fit on small GPUs
- Understand and implement LoRA (Low-Rank Adaptation)
- Understand DoRA (Weight-Decomposed LoRA) and how it improves on LoRA
- Combine quantization + LoRA into QLoRA for single-GPU fine-tuning of large models
- Apply memory/speed optimizations: Flash Attention 2, gradient checkpointing, Unsloth
- Merge trained LoRA adapters back into a base model for deployment

---

## Week Overview

| Day | Topic | Key Code |
|-----|-------|----------|
| **Day 1** | Quantization Basics | FP32/FP16/INT8/INT4 comparison, BitsAndBytesConfig, memory profiling |
| **Day 2** | LoRA | Low-rank adapter matrices, rank/alpha, PEFT LoraConfig, trainable param comparison |
| **Day 3** | DoRA | Weight-decomposed LoRA, magnitude + direction decomposition, LoRA vs DoRA comparison |
| **Day 4** | QLoRA | 4-bit base model + LoRA adapters, single-GPU fine-tuning of a larger model |
| **Day 5** | Speed & Memory Tricks | Flash Attention 2, gradient checkpointing, Unsloth speedups |
| **Day 6** | Model Merging | Merging LoRA adapters into base weights, evaluating the merged model |

---

## Technologies & Libraries

- **PyTorch** — tensor operations and model training
- **Hugging Face Transformers** — model loading and tokenization
- **PEFT** — LoRA / DoRA adapter implementation
- **BitsAndBytes** — 4-bit and 8-bit quantization
- **TRL** — SFTTrainer for adapter training
- **Unsloth** — optimized single-GPU fine-tuning
- **Accelerate** — device management
- **Matplotlib / Pandas** — data visualization and logging

---

## Folder Structure

```text
week-03-efficient-fine-tuning/
│
├── README.md
├── requirements.txt
│
├── day-01-quantization-basics/
│   ├── quantization_basics.py
│   ├── quantization_basics.ipynb
│   └── README.md
│
├── day-02-lora/
│   ├── lora.py
│   ├── lora.ipynb
│   └── README.md
│
├── day-03-dora/
│   ├── dora.py
│   ├── dora.ipynb
│   └── README.md
│
├── day-04-qlora/
│   ├── qlora.py
│   ├── qlora.ipynb
│   └── README.md
│
├── day-05-speed-memory-tricks/
│   ├── speed_memory_tricks.py
│   ├── speed_memory_tricks.ipynb
│   └── README.md
│
└── day-06-model-merging/
    ├── model_merging.py
    ├── model_merging.ipynb
    └── README.md
```

The **capstone project** is located at:

```
projects/efficient_assistant/
├── train.py           # Complete pipeline covering all 6 days
├── inference.py        # Interactive chat with the merged model
└── README.md
```

---

## Environment Setup

### Step 1 — Install the base requirements

```powershell
# From repository root
venv\Scripts\activate

cd weeks\week-03-efficient-fine-tuning

pip install -r requirements.txt
```

### Step 2 — Check whether you have an NVIDIA GPU

```bash
nvidia-smi
```

This prints your GPU model, driver version, and available VRAM. If this command isn't recognized at all, you don't have an NVIDIA GPU set up on this machine — see the CPU vs. GPU table below to know which days that affects.

### Step 3 — Install a CUDA-enabled build of PyTorch (GPU users only)

By default, `pip install torch` often installs the **CPU-only** build, even on a machine with a perfectly good GPU. This is the single most common setup mistake — everything appears to install correctly, then every GPU-dependent day fails or silently runs on CPU. Fix it explicitly:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> Check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the current recommended `cu###` version — it changes periodically. Pick the highest version listed there; PyTorch's CUDA wheels are backward-compatible with newer GPU drivers, so you don't need an exact match to your driver's reported CUDA version.

### Step 4 — Verify PyTorch can see your GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output: `True` followed by your GPU's name (e.g. `NVIDIA GeForce RTX 4050 Laptop GPU`). If it prints `False`, Step 3 installed the wrong build — reinstall using the command above.

### Step 5 — Verify `bitsandbytes` (needed for Days 1 and 4)

```bash
python -c "import bitsandbytes as bnb; print('bitsandbytes version:', bnb.__version__)"
```

This should print a version number with no errors. `bitsandbytes` depends on a working CUDA-enabled PyTorch install, so run this **after** Step 4 passes, not before.

### Which days actually need a GPU?

| Day | Needs GPU? | Notes |
|---|---|---|
| Day 1 — Quantization Basics | Partial | FP32/FP16 run on CPU; INT8/INT4 sections need a GPU and skip automatically otherwise |
| Day 2 — LoRA | No | Runs fine on CPU (small model, short toy training) |
| Day 3 — DoRA | No | Same as Day 2 |
| Day 4 — QLoRA | **Yes** | `bitsandbytes` 4-bit loading has no CPU fallback |
| Day 5 — Speed & Memory Tricks | Partial | Gradient checkpointing works on CPU (savings only visible on GPU); Flash Attention 2 and Unsloth both require a GPU and skip automatically otherwise |
| Day 6 — Model Merging | No | Runs fine on CPU |
| Capstone project | **Yes** | Same reason as Day 4 — loads a model in 4-bit |

No local GPU at all? A free Google Colab account with a T4 GPU runtime covers every day in this week except one specific part of Day 5 (Flash Attention 2 needs Ampere-or-newer hardware, which the free T4 doesn't have — that section will skip cleanly).

### Notes for Windows users specifically

- Steps 1-5 above work the same way on native Windows, with one exception: `flash-attn` (used in one part of Day 5) does not ship prebuilt wheels for Windows and must be compiled from source, requiring the CUDA Toolkit and Visual Studio Build Tools. If that install fails, run just that section inside WSL2 (Windows Subsystem for Linux) or on Google Colab instead — everything else installs normally on native Windows, including `bitsandbytes` and `unsloth`.
- **Running on a laptop GPU with 6GB VRAM (e.g. RTX 4050 Laptop)?** Days 1-3 and 6 have no VRAM concerns. Day 4 and the capstone are pre-configured with a conservative `MAX_SEQ_LENGTH` to fit comfortably in 6GB — see those days' READMEs for the exact values. If you still hit an out-of-memory error, close other GPU-using applications first (browser tabs with hardware acceleration are a common culprit), since laptop GPUs often share memory with the display.

---

## What You'll Learn

- The memory math behind FP32, FP16, INT8, and INT4 precision
- How LoRA freezes the base model and trains small low-rank adapters instead
- How DoRA decomposes weights into magnitude and direction for better adaptation
- How QLoRA combines 4-bit quantization with LoRA to fine-tune large models cheaply
- How Flash Attention 2, gradient checkpointing, and Unsloth cut training time and memory further
- How to merge trained adapters back into the base model for simpler deployment

---

## Capstone Project

### Efficient Fine-Tuning Assistant

Located at `projects/efficient_assistant/`, the capstone project demonstrates all six days' concepts in a single pipeline — training a small cybersecurity assistant efficiently enough to run on a free-tier GPU, then merging it into a standalone deployable model.

| Day | Concept | In Project |
|-----|---------|------------|
| 1 | Quantization | Base model loaded in 4-bit (NF4) |
| 2 | LoRA | Low-rank adapters attached to the quantized base |
| 3 | DoRA | Toggle `USE_DORA = True` to switch adapter types |
| 4 | QLoRA | Days 1 + 2/3 combined — the full training step |
| 5 | Speed & Memory Tricks | Gradient checkpointing enabled by default |
| 6 | Model Merging | `merge_and_unload()` for a standalone deployable model |

To run:

```bash
cd projects/efficient_assistant
python train.py          # Runs the full pipeline
python inference.py      # Chat with your trained, merged model
```

Domain-agnostic — swap the toy dataset and `DOMAIN` name to reuse it for anything.

---

## Ready to Begin?

Start with Day 1 to understand quantization — it's the foundation that QLoRA (Day 4) builds directly on top of.
