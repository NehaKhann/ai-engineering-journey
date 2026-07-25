# Day 4 — QLoRA (Quantized LoRA)

> Week 3 — Efficient Fine-Tuning & Quantization

Learn **QLoRA** — the technique that combines Day 1's quantization with Day 2/3's LoRA-style adapters to fine-tune large models on a single consumer GPU. This is the method most real-world "fine-tune a 7B/13B model at home" tutorials are actually using.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- How QLoRA combines a frozen 4-bit base model with trainable LoRA adapters
- What `prepare_model_for_kbit_training()` does and why it's needed
- How to load a model with `BitsAndBytesConfig` and immediately attach `LoraConfig` on top
- How to measure real GPU memory usage during a QLoRA training run
- Why QLoRA's memory advantage over full fine-tuning grows with model size

---

## Project Structure

```
day-04-qlora/
├── README.md
├── qlora.py
└── qlora.ipynb
```

| File | Description |
|------|-------------|
| `qlora.py` | Loads a model in 4-bit, attaches LoRA adapters, trains, and measures memory vs an estimated full-fine-tuning cost. |
| `qlora.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

---

## Requirements

Install once from the week-level requirements file:

```bash
cd weeks/week-03-efficient-fine-tuning
pip install -r requirements.txt
```

---

## Run

```bash
python qlora.py
```

Or open `qlora.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

> **This day REQUIRES a CUDA GPU.** `bitsandbytes` 4-bit loading is not available on CPU. Use a free Google Colab T4 runtime if you don't have local GPU access — go to Runtime → Change runtime type → T4 GPU before running.
>
> This lesson uses a slightly larger model (`Qwen2.5-1.5B-Instruct`) than Days 1-3 (`Qwen2.5-0.5B-Instruct`) specifically to make QLoRA's memory savings more visible — the benefit scales with model size. The first run downloads the model (~3GB) from Hugging Face; subsequent runs load from cache.
>
> **Running on a 6GB laptop GPU (e.g. RTX 4050 Laptop)?** This fits comfortably — a 1.5B model in 4-bit uses roughly 1-1.5GB for the weights alone, leaving headroom for training. `MAX_SEQ_LENGTH` at the top of the script defaults to 128 for exactly this kind of setup; lower it further if you still hit an out-of-memory error. Laptop GPUs often share VRAM with the display, so closing other GPU-heavy apps (browser tabs with hardware acceleration, other AI tools) before running helps too.

---

## Concepts Covered

### 1. The QLoRA Recipe

QLoRA is Day 1 (quantization) and Day 2 (LoRA) combined:

1. Load the base model in 4-bit (NF4), frozen
2. Call `prepare_model_for_kbit_training()` to prepare it for stable training
3. Attach LoRA adapters on top via `get_peft_model()` — these train in higher precision (bfloat16)
4. Train only the adapters; the 4-bit base does the forward-pass work without ever updating

### 2. Why `prepare_model_for_kbit_training()` Is Needed

Training directly on top of a quantized model has stability pitfalls — certain layers (like layer norms) need to stay in higher precision, and gradient checkpointing needs to be wired up correctly. This PEFT helper handles those details automatically so the training loop itself looks just like plain LoRA.

### 3. Loading Quantized + Attaching Adapters

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)
```

From here, the training loop is identical to Day 2's LoRA loop.

### 4. Measuring the Real Memory Payoff

The script measures actual GPU memory after loading the 4-bit base and at peak during training, then compares it against an estimate of what full FP32 fine-tuning would require (roughly 16 bytes per parameter — weights, gradients, and two AdamW optimizer states).

### 5. Why This Scales With Model Size

QLoRA's advantage isn't very visible on a 0.5B model — it becomes dramatic on 7B, 13B, or 70B models, where full fine-tuning would need 100+ GB of GPU memory but QLoRA fits in a fraction of that. This lesson uses a 1.5B model as a middle ground to make the gap visible without requiring a huge download or a data-center GPU.

---

## Comparison: Full Fine-Tuning vs LoRA vs QLoRA

| Aspect | Full Fine-Tuning | LoRA | QLoRA |
|--------|------------------|------|-------|
| Base model precision | FP32/FP16 | FP32/FP16 | 4-bit (NF4) |
| Base model trainable | Yes | No (frozen) | No (frozen) |
| Adapter parameters | N/A | Small, trained | Small, trained |
| Memory footprint | Highest | Lower | Lowest |
| Largest model feasible on 1 consumer GPU | Small models only | Medium models | Large models (7B+) |

---

## Sample Output

The program loads the 1.5B model in 4-bit, reports memory usage, attaches LoRA adapters, trains for 20 steps on toy data, and prints a before/after comparison plus a memory comparison table (measured QLoRA memory vs estimated full-FP32 fine-tuning memory) — saved to `experiment_log.json`.

---

## Key Takeaways

- QLoRA = 4-bit quantized frozen base model + trainable LoRA adapters on top
- `prepare_model_for_kbit_training()` bridges quantization and trainability
- Only the small adapters update; the base model stays frozen and quantized throughout
- Memory savings scale with model size — this is what makes fine-tuning 7B+ models feasible on a single consumer GPU
- Requires a CUDA GPU; use Colab's free T4 runtime if you don't have local GPU access

---

## Related Resources

- Day 1 — Quantization Basics (the "Q" in QLoRA)
- Day 2 — LoRA (the "LoRA" in QLoRA)
- Day 3 — DoRA (target_modules and adapter config can swap to `use_dora=True` for QDoRA)
- Week 3 Overview — `weeks/week-03-efficient-fine-tuning`
