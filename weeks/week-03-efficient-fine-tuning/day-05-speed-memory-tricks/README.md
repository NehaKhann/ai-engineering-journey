# Day 5 — Speed & Memory Tricks

> Week 3 — Efficient Fine-Tuning & Quantization

Learn three optimizations that stack on top of everything from Days 1-4: **gradient checkpointing**, **Flash Attention 2**, and **Unsloth** — each measured for real, not just taken on faith.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- How gradient checkpointing trades compute for memory, and when that trade is worth it
- How Flash Attention 2 avoids materializing the full attention matrix, and its hardware requirements
- How Unsloth patches training with fused kernels for speedups across a wider range of GPUs
- How to measure real memory and time impact for each optimization
- How these tricks stack with quantization and LoRA/DoRA/QLoRA from earlier days

---

## Project Structure

```
day-05-speed-memory-tricks/
├── README.md
├── speed_memory_tricks.py
└── speed_memory_tricks.ipynb
```

| File | Description |
|------|-------------|
| `speed_memory_tricks.py` | Trains the same LoRA setup four ways (baseline, +gradient checkpointing, +Flash Attention 2, Unsloth) and compares memory/speed. |
| `speed_memory_tricks.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

---

## Requirements

Install the base requirements from the week-level file:

```bash
cd weeks/week-03-efficient-fine-tuning
pip install -r requirements.txt
```

Two optional packages are needed for the full comparison (both are skipped gracefully if missing):

```bash
pip install flash-attn --no-build-isolation   # Flash Attention 2 — needs Ampere+ GPU
pip install unsloth                            # Unsloth — works on T4 and newer
```

---

## Run

```bash
python speed_memory_tricks.py
```

Or open `speed_memory_tricks.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

> **GPU notes:**
> - Gradient checkpointing works on CPU or GPU, but memory savings are only meaningful on GPU.
> - **Flash Attention 2 requires an Ampere-or-newer GPU** (A100, RTX 30xx/40xx, L4). Colab's free **T4 is NOT supported** — that section will skip automatically with a clear message.
> - **Unsloth works on T4** and newer, so it's the more accessible speedup to test on a free Colab runtime.
> - If a package is missing or the GPU doesn't support a technique, that configuration is skipped rather than crashing the script.
>
> **Running on an RTX 4050 Laptop GPU (or similar RTX 40-series laptop card)?** These are Ada Lovelace architecture, which meets Flash Attention 2's hardware requirement — so all four configurations in this lesson can run on your machine, not just three. The one snag: `flash-attn` doesn't ship prebuilt wheels for native Windows, and building it from source needs the CUDA Toolkit and Visual Studio Build Tools installed, which can be a fiddly one-off setup. Two options if `pip install flash-attn --no-build-isolation` fails on Windows: run this section inside WSL2 (Windows Subsystem for Linux, which behaves like a normal Linux install for these purposes), or just run it on Colab instead — the *result* you're comparing is the technique's impact, not where it runs. Unsloth installs normally on native Windows either way.

---

## Concepts Covered

### 1. Gradient Checkpointing

Normally, every intermediate activation from the forward pass is kept in memory so it's available for the backward pass. Gradient checkpointing discards most of them and recomputes them on demand during backward — trading extra compute for significantly less memory.

```python
model.gradient_checkpointing_enable()
model.enable_input_require_grads()  # needed when the base model is frozen (LoRA/DoRA/QLoRA)
```

### 2. Flash Attention 2

Standard attention computes and stores a full `(seq_len, seq_len)` matrix of attention scores. Flash Attention 2 is a fused kernel that computes attention in blocks, without ever materializing that full matrix — both faster and lower memory, especially for longer sequences.

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
```

Requires an Ampere-or-newer GPU and the `flash-attn` package — this is a hardware-dependent optimization, not just a software toggle.

### 3. Unsloth

Unsloth replaces key attention and MLP operations with hand-optimized Triton kernels, giving training speedups with almost no code change, plus a simplified API for 4-bit loading (`FastLanguageModel.from_pretrained(..., load_in_4bit=True)`). Unlike Flash Attention 2, it runs on a broader range of GPUs, including the free Colab T4.

### 4. Measuring Rather Than Assuming

Each configuration is timed and memory-profiled identically — same model, same toy dataset, same number of training steps — so the numbers reflect the actual effect of each optimization in this environment, not marketing claims.

### 5. Stacking Optimizations

In real-world fine-tuning setups, these tricks combine: QLoRA (Day 4) + gradient checkpointing + Unsloth or Flash Attention 2 is a common recipe for training larger models efficiently on a single GPU.

---

## Comparison: The Three Tricks

| Trick | Effect | Hardware requirement | Trade-off |
|---|---|---|---|
| Gradient checkpointing | Lower memory | Any GPU (or CPU) | Slightly slower per step |
| Flash Attention 2 | Faster + lower memory | Ampere+ GPU only | None, if hardware supports it |
| Unsloth | Faster training, easier 4-bit loading | Wider GPU support (incl. T4) | Library dependency, patches internals |

---

## Sample Output

The program runs baseline LoRA training, then the same training with gradient checkpointing enabled, then with Flash Attention 2 (if supported), then with Unsloth (if installed) — printing load time, train time, peak memory, and final loss for each, with a side-by-side summary table saved to `experiment_log.json`. Unsupported configurations print a clear skip reason instead of failing.

---

## Key Takeaways

- Gradient checkpointing trades compute for memory — use it when memory, not speed, is the bottleneck
- Flash Attention 2 is faster *and* lower memory, but only on Ampere-or-newer GPUs
- Unsloth gives training speedups on a wider range of hardware, including free-tier Colab GPUs
- All three stack with quantization and LoRA/DoRA/QLoRA from Days 1-4
- Always measure in your own environment — GPU generation changes which tricks are even available

---

## Related Resources

- Day 2 — LoRA (the training setup all four configurations build on)
- Day 4 — QLoRA (commonly combined with these tricks in practice)
- Day 6 — Model Merging
- Week 3 Overview — `weeks/week-03-efficient-fine-tuning`
