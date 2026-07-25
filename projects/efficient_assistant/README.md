# Capstone Project — Efficient Fine-Tuning Pipeline

> Week 3 — Efficient Fine-Tuning & Quantization

A complete, reusable pipeline combining all six days of Week 3 into one working system: a small cybersecurity assistant, fine-tuned efficiently enough to train on a single free-tier GPU, then merged into a standalone deployable model.

---

## How Each Day Fits In

| Day | Concept | In This Project |
|-----|---------|------------------|
| 1 | Quantization | Base model loaded in 4-bit (NF4) via `BitsAndBytesConfig` |
| 2 | LoRA | Low-rank adapters attached via `LoraConfig` + `get_peft_model()` |
| 3 | DoRA | Toggle `USE_DORA = True` to switch to weight-decomposed adapters |
| 4 | QLoRA | Days 1 + 2/3 combined — adapters trained directly on the 4-bit base |
| 5 | Speed & Memory Tricks | Gradient checkpointing enabled by default via `USE_GRADIENT_CHECKPOINTING` |
| 6 | Model Merging | `merge_and_unload()` produces a standalone deployable model at the end |

---

## Project Structure

```
projects/efficient_assistant/
├── README.md
├── train.py          # Full pipeline: Days 1-6 in one script
└── inference.py       # Interactive chat with the merged model
```

---

## The Domain: Cybersecurity Assistant

The pipeline ships with a small (8-example) cybersecurity instruction dataset — teaching the model to explain security concepts (phishing, MFA, zero-days, SQL injection, etc.) clearly and briefly. This ties naturally into security-focused portfolio work, but the pipeline is domain-agnostic:

```python
CYBERSECURITY_DATA = [
    {"prompt": "...", "response": "..."},
    ...
]
```

Swap this list for your own `(prompt, response)` pairs — banking compliance, a coding assistant, a support bot, anything — and the rest of the pipeline works unchanged. For anything beyond a toy demo, prepare a real dataset the way Week 2 Day 3 covered (JSONL, token length stats, train/val split) rather than a hardcoded list.

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
cd projects/efficient_assistant
python train.py          # Runs the full 6-day pipeline
python inference.py      # Chat with your trained, merged model
```

> **REQUIRES a CUDA GPU** — 4-bit loading via `bitsandbytes` doesn't work on CPU. Use a free Google Colab T4 runtime if you don't have local GPU access (Runtime → Change runtime type → T4 GPU). The first run downloads `Qwen2.5-1.5B-Instruct` (~3GB) from Hugging Face.
>
> **Running on a 6GB laptop GPU (e.g. RTX 4050 Laptop)?** This fits — 4-bit weights for a 1.5B model are roughly 1-1.5GB, leaving headroom for adapters and training on a 6GB card. `MAX_SEQ_LENGTH` at the top of `train.py` defaults to 256; lower it if you hit an out-of-memory error. Closing other GPU-heavy apps (browser hardware acceleration, other AI tools) before running helps too, since laptop GPUs typically share VRAM with the display.

---

## What `train.py` Does, Step by Step

1. **Load quantized base** (Day 1 + Day 4) — `Qwen2.5-1.5B-Instruct` in 4-bit NF4, measuring GPU memory immediately after load.
2. **Evaluate before training** — asks the untrained base model two held-out cybersecurity questions, to have a genuine "before" baseline.
3. **Attach adapters** (Day 2 / Day 3) — `prepare_model_for_kbit_training()` + `LoraConfig` (or DoRA, if `USE_DORA = True`), with gradient checkpointing enabled (Day 5).
4. **Train** (Day 4) — runs several epochs over the toy dataset, printing loss every few steps, tracking peak GPU memory.
5. **Save the adapter** — the small trained A/B matrices, saved separately from the base model.
6. **Evaluate after training (adapter-attached)** — same two held-out questions, now answered by the fine-tuned adapter.
7. **Merge** (Day 6) — reloads the base model in full precision (not 4-bit — merging can't target quantized weights directly), loads the saved adapter on top, and calls `merge_and_unload()`.
8. **Evaluate after merging** — confirms the merged, standalone model still gives the fine-tuned answers.
9. **Save results** — all parameters, memory, timing, and evaluation outputs to `capstone_log.json`.

---

## What `inference.py` Does

Loads the merged model from `merged_cybersecurity_assistant/` with plain `AutoModelForCausalLM` — no PEFT import needed — and opens an interactive terminal chat loop. Type `exit` or `quit` to stop.

---

## Customizing the Pipeline

| Want to... | Change this |
|---|---|
| Use DoRA instead of LoRA | `USE_DORA = True` |
| Change adapter capacity | `LORA_RANK`, `LORA_ALPHA` |
| Target more layers | `TARGET_MODULES` (e.g. add `"k_proj"`, `"o_proj"`) |
| Use your own domain data | Replace `CYBERSECURITY_DATA` and `DOMAIN` |
| Train longer | `TRAINING_EPOCHS`, `LEARNING_RATE` |
| Disable gradient checkpointing | `USE_GRADIENT_CHECKPOINTING = False` (slightly faster, more memory) |
| Add Flash Attention 2 / Unsloth | See Day 5's README for the exact code to layer in — not included by default here since they require specific GPU hardware |

---

## Sample Output

`train.py` prints each stage as it runs — quantized load memory, before/after training answers to two evaluation questions, training loss curve, peak memory, merge confirmation, and a final round of answers from the merged model — all saved to `capstone_log.json` for reference when writing up results.

---

## Key Takeaways

- This project is Week 3 end-to-end: quantize → adapt → train → optimize → merge → deploy
- The same script trains LoRA or DoRA with a single flag flip
- Peak GPU memory and training loss are measured directly, not estimated
- The final artifact is a standalone merged model — deployable with zero PEFT dependency
- The pipeline is domain-agnostic — swap the dataset and `DOMAIN` name to reuse it for anything

---

## Related Resources

- Day 1 — Quantization Basics
- Day 2 — LoRA
- Day 3 — DoRA
- Day 4 — QLoRA
- Day 5 — Speed & Memory Tricks
- Day 6 — Model Merging
- Week 3 Overview — `weeks/week-03-efficient-fine-tuning`
