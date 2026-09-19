import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import pipeline

print("=" * 80)
print("Module 01 — The Generative AI Landscape")
print("=" * 80 + "\n")

# =============================================
# Part 1 — Discriminative vs Generative
# =============================================
# Same input text, two kinds of model:
#   discriminative -> learns P(label | text)  -> "what is this?"
#   generative     -> learns P(text)          -> "what comes next / what would this look like?"

text = "The new firewall update went smoothly and everything works"

print("PART 1 — Discriminative vs Generative on the same input")
print(f'Input: "{text}"\n')

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
verdict = classifier(text)[0]
print(f"Discriminative (DistilBERT): {verdict['label']} (confidence {verdict['score']:.2f})")
print("  -> Output is a LABEL. The model can only choose from classes it was trained on.\n")

generator = pipeline("text-generation", model="gpt2")
continuation = generator(
    text,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.8,
    pad_token_id=50256,
)[0]["generated_text"]
print(f"Generative (GPT-2): {continuation}")
print("  -> Output is NEW TEXT. The model samples from a learned distribution.\n")

# =============================================
# Part 2 — A generative model from scratch
# =============================================
# A bigram model learns P(next word | current word) by counting.
# GPT-2 does the same job with a Transformer and a far longer context.
# The idea — learn a distribution, then sample from it — is identical.

corpus = """
the model reads the prompt. the model predicts the next word. the model samples the next word from a distribution.
a generative model learns the distribution of its training data. a discriminative model learns a decision boundary.
the model generates text one token at a time. the model generates images by removing noise step by step.
a large model learns patterns from a large dataset. a small model learns patterns from a small dataset.
the prompt guides the model. the data shapes the model. the sampling settings shape the output.
"""

tokens = re.findall(r"[a-z']+|\.", corpus.lower())

bigrams = defaultdict(Counter)
previous = "<s>"
for token in tokens:
    bigrams[previous][token] += 1
    previous = "<s>" if token == "." else token


def next_word_probabilities(word):
    counts = bigrams[word]
    total = sum(counts.values())
    return {w: c / total for w, c in counts.items()}


def sample_sentence(rng, max_words=15):
    word, sentence = "<s>", []
    for _ in range(max_words):
        probs = next_word_probabilities(word)
        word = rng.choices(list(probs), weights=list(probs.values()))[0]
        if word == ".":
            break
        sentence.append(word)
    return " ".join(sentence).capitalize() + "."


print("PART 2 — A tiny generative model built from scratch (bigram counts)")
print(f"Trained on {len(tokens)} tokens, vocabulary of {len(set(tokens))} words.\n")

rng = random.Random(7)
for i in range(5):
    print(f"Sample {i + 1}: {sample_sentence(rng)}")

context = "model"
probs = next_word_probabilities(context)
print(f'\nP(next word | "{context}"):')
for word, p in sorted(probs.items(), key=lambda item: -item[1]):
    print(f"  {word:<12} {p:.2f}")

# =============================================
# Visualization
# =============================================
ranked = sorted(probs.items(), key=lambda item: -item[1])
plt.figure(figsize=(9, 4.5))
plt.bar([w for w, _ in ranked], [p for _, p in ranked], color="#4C72B0")
plt.title(f'Learned distribution: P(next word | "{context}")', fontsize=13)
plt.ylabel("Probability")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "assets" / "bigram_distribution.png", dpi=150)
plt.show()

# =============================================
# Part 3 — The map of generative model families
# =============================================
print("\nPART 3 — Generative model families at a glance\n")

families = [
    ("Autoregressive (GPT, Claude, Llama)", "Next token given previous tokens", "One token at a time", "Text, code"),
    ("Diffusion (Stable Diffusion, DALL-E)", "How to remove noise from data", "Denoise over many steps", "Images, video, audio"),
    ("GAN", "Generator vs discriminator game", "One forward pass", "Images"),
    ("VAE", "Compressed latent space of the data", "Decode a sampled latent", "Images, embeddings"),
]

print(f"{'Family':<40}{'Learns':<40}{'Generates by':<28}{'Typical output'}")
print("-" * 125)
for name, learns, generates, output in families:
    print(f"{name:<40}{learns:<40}{generates:<28}{output}")

print("\n🎯 Key Takeaways from Module 01:")
print("• Discriminative models predict a label; generative models produce new data.")
print("• Every generative model learns a probability distribution and samples from it.")
print("• An LLM is a bigram model scaled up: better context, better architecture, same idea.")
print("• Different data types favor different families: text -> autoregressive, images -> diffusion.")
