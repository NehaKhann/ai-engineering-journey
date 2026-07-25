# Day 1 — Quantization Basics

> Week 3 — Efficient Fine-Tuning & Quantization

Learn **quantization** — storing model weights with fewer bits to save memory, with minimal quality loss. This is the foundation that QLoRA (Day 4) builds directly on top of.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- Why quantization matters for fine-tuning and inference on limited hardware
- The difference between FP32, FP16, INT8, and INT4 precision
- How to load a model in different precisions using `BitsAndBytesConfig`
- How to measure GPU memory footprint, load time, and inference time
- The quality trade-offs between precision levels
- Why INT4 (NF4) quantization is the format used in QLoRA

---

## Project Structure

```
day-01-quantization-basics/
├── README.md
├── quantization_basics.py
└── quantization_basics.ipynb
```

| File | Description |
|------|-------------|
| `quantization_basics.py` | Loads the same model in FP32/FP16/INT8/INT4 and compares memory, speed, and output quality. |
| `quantization_basics.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

Running either the script or the notebook generates two output files: `experiment_log.json` (raw numbers) and `quantization_comparison.png` (a 3-panel bar chart — memory, load time, inference time — built with `matplotlib`).

---

## Requirements

Install once from the week-level requirements file:

```bash
cd weeks/week-03-efficient-fine-tuning
pip install -r requirements.txt
```

**If you have an NVIDIA GPU and want to run the INT8/INT4 sections**, verify your setup before running the script — this catches the most common setup mistake (PyTorch silently installing as CPU-only) before it wastes your time mid-script:

```bash
# Confirm PyTorch can see your GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

This should print `True` followed by your GPU's name. If it prints `False`, reinstall PyTorch with a CUDA-enabled build:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then confirm `bitsandbytes` (the library that performs the actual 4-bit/8-bit compression) is working:

```bash
python -c "import bitsandbytes as bnb; print('bitsandbytes version:', bnb.__version__)"
```

If this errors instead of printing a version number, reinstall it with `pip install bitsandbytes` after the PyTorch check above passes.

---

## Run

```bash
python quantization_basics.py
```

Or open `quantization_basics.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

> **Note:** `bitsandbytes` (used for 8-bit/4-bit loading) requires a CUDA GPU. A free Google Colab T4 runtime works well. On CPU-only machines, the FP32/FP16 runs still work, but INT8/INT4 are skipped automatically. The first run downloads `Qwen2.5-0.5B-Instruct` (~1GB) from Hugging Face; subsequent runs load from cache.

---

## Concepts Covered

### 1. Why Quantization Matters

Every weight in a neural network is stored using some number of bits. Fewer bits means less memory, but also less precision. Quantization stores weights in a lower-precision format without losing much model quality — which is why a 7B parameter model can shrink from ~28GB (FP32) down to ~3.5GB (INT4).

### 2. The Four Precision Levels

| Precision | Bits per weight | Relative size (vs FP32) | Typical use |
|---|---|---|---|
| FP32 | 32 | 1x | Training from scratch, research baselines |
| FP16 / BF16 | 16 | 0.5x | Standard for fine-tuning & inference |
| INT8 | 8 | 0.25x | Inference on memory-constrained GPUs |
| INT4 (NF4) | 4 | 0.125x | QLoRA fine-tuning, running big models on small GPUs |

### 3. BitsAndBytesConfig

`BitsAndBytesConfig` is the Hugging Face interface for loading models in 8-bit or 4-bit. Key options used in this lesson:

- `load_in_8bit` / `load_in_4bit` — turns on quantized loading
- `bnb_4bit_quant_type="nf4"` — NormalFloat4, a quantization scheme tuned for the distribution of LLM weights
- `bnb_4bit_compute_dtype=torch.float16` — computations still happen in FP16 even though weights are stored in 4-bit
- `bnb_4bit_use_double_quant=True` — quantizes the quantization constants themselves for extra savings

### 4. Memory vs Load Time vs Quality

Quantized models are smaller in memory but not necessarily faster to *load* — the quantization step itself takes time. The real payoff shows up in memory footprint and the ability to fit larger models on the same GPU.

### 5. When Precision Loss Becomes Noticeable

For small models on simple prompts, FP16/INT8/INT4 outputs are usually all coherent. Quality loss becomes more visible on harder prompts, longer generations, or lower-quality quantization schemes — which is why NF4 (not naive INT4) is the standard for fine-tuning.

---

## Comparison: Precision Trade-offs

| Aspect | FP32 | FP16 | INT8 | INT4 (NF4) |
|--------|------|------|------|------------|
| Memory | Highest | Medium | Low | Lowest |
| Quality | Reference | Near-identical | Small loss | Small-moderate loss |
| GPU requirement | Highest VRAM | Standard | Reduced VRAM | Minimal VRAM |
| Used in | Research baselines | Standard fine-tuning | Constrained inference | QLoRA |

---

## Sample Output

The program loads the same model four times, prints memory/timing for each, saves a summary to `experiment_log.json`, and generates `quantization_comparison.png` — a 3-panel bar chart showing memory, load time, and inference time side by side across all four modes. The memory panel is the clearest one: the bars visibly shrink from FP32 down to INT4.

---

## Key Takeaways

- Quantization trades a small amount of precision for large memory savings
- FP16 is nearly free — always prefer it over FP32 unless you specifically need full precision
- INT8/INT4 quantization (via `bitsandbytes`) is what makes it possible to fine-tune multi-billion parameter models on a single consumer GPU
- NF4 with double quantization is the standard configuration used in QLoRA (Day 4)

---

## Related Resources

- Day 2 — LoRA
- Day 4 — QLoRA (combines this day's quantization with Day 2's LoRA)
- Week 3 Overview — `weeks/week-03-efficient-fine-tuning`
