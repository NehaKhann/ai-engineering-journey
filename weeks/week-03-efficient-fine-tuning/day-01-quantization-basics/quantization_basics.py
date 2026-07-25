"""
Day 01 — Quantization Basics
=============================
Goal: Understand WHY quantization matters for fine-tuning and inference,
and see it in action — loading the same model in different precisions
and comparing memory footprint + output quality.

We compare four precision modes:
  1. FP32  (full precision)       -> baseline, most memory
  2. FP16  (half precision)       -> ~2x smaller, minimal quality loss
  3. INT8  (8-bit quantization)   -> ~4x smaller vs FP32
  4. INT4  (4-bit quantization)   -> ~8x smaller vs FP32, used in QLoRA

Run this on a machine/Colab with a GPU for the memory numbers to be meaningful.
On CPU-only machines, 8-bit/4-bit loading via bitsandbytes may not work —
see the README for a Colab link.
"""

import gc
import json
import time
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # small model so this runs on modest GPUs
PROMPT = "Explain what a neural network is in one sentence."
MAX_NEW_TOKENS = 40
LOG_FILE = "experiment_log.json"
PLOT_FILE = "quantization_comparison.png"


@dataclass
class RunResult:
    mode: str
    load_time_sec: float
    memory_mb: float
    output_text: str
    inference_time_sec: float


def get_gpu_memory_mb() -> float:
    """Returns current GPU memory allocated, in MB. 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_model(mode: str):
    """
    Loads the model in one of: 'fp32', 'fp16', 'int8', 'int4'
    """
    common_kwargs = dict(device_map="auto" if torch.cuda.is_available() else None)

    if mode == "fp32":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, **common_kwargs
        )

    elif mode == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, **common_kwargs
        )

    elif mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, **common_kwargs
        )

    elif mode == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",       # normal-float 4bit, best for LLM weights
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # quantizes the quantization constants too
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, **common_kwargs
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model


def run_mode(mode: str, tokenizer) -> RunResult:
    print(f"\n{'=' * 50}\nRunning mode: {mode.upper()}\n{'=' * 50}")
    clear_memory()

    t0 = time.time()
    model = load_model(mode)
    load_time = time.time() - t0

    memory_mb = get_gpu_memory_mb()

    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    inference_time = time.time() - t0

    output_text = tokenizer.decode(
        output_ids[0][inputs.shape[-1]:], skip_special_tokens=True
    )

    print(f"Load time:      {load_time:.2f}s")
    print(f"GPU memory:     {memory_mb:.1f} MB")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Output:         {output_text.strip()}")

    result = RunResult(
        mode=mode,
        load_time_sec=round(load_time, 2),
        memory_mb=round(memory_mb, 1),
        output_text=output_text.strip(),
        inference_time_sec=round(inference_time, 2),
    )

    del model
    clear_memory()
    return result


def plot_results(results: list):
    """
    Builds a 3-panel bar chart comparing memory, load time, and inference
    time across all four precision modes, and saves it as a PNG.

    This turns the printed summary table into something you can actually
    look at — the memory panel in particular is the clearest visual proof
    that quantization works: watch the bars shrink from FP32 to INT4.
    """
    import matplotlib.pyplot as plt

    modes = [r.mode.upper() for r in results]
    memory = [r.memory_mb for r in results]
    load_times = [r.load_time_sec for r in results]
    inference_times = [r.inference_time_sec for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(modes, memory, color="#2a78d6")
    axes[0].set_title("GPU memory used")
    axes[0].set_ylabel("MB")

    axes[1].bar(modes, load_times, color="#eb6834")
    axes[1].set_title("Model load time")
    axes[1].set_ylabel("seconds")

    axes[2].bar(modes, inference_times, color="#1baf7a")
    axes[2].set_title("Inference time")
    axes[2].set_ylabel("seconds")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Quantization comparison — {MODEL_NAME}", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison chart to {PLOT_FILE}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    modes = ["fp32", "fp16", "int8", "int4"]
    results = []

    for mode in modes:
        try:
            results.append(run_mode(mode, tokenizer))
        except Exception as e:
            print(f"[{mode}] skipped — {e}")

    # Summary table
    print(f"\n{'=' * 65}")
    print(f"{'Mode':<8}{'Memory (MB)':<15}{'Load (s)':<12}{'Inference (s)':<15}")
    print(f"{'=' * 65}")
    for r in results:
        print(f"{r.mode:<8}{r.memory_mb:<15}{r.load_time_sec:<12}{r.inference_time_sec:<15}")

    if results:
        try:
            plot_results(results)
        except ImportError:
            print("matplotlib not installed — skipping chart. Run: pip install matplotlib")

    with open(LOG_FILE, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved full results to {LOG_FILE}")


if __name__ == "__main__":
    main()
