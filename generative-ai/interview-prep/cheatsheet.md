# 📝 Generative AI Cheatsheet

One-line answers for quick revision. Each links to the module with the full explanation.

---

## Module 01 — The Generative AI Landscape

| Concept | One-liner |
|---|---|
| **Discriminative model** | Learns `P(label \| input)`. Predicts a label, cannot create data. |
| **Generative model** | Learns the distribution of the data. Can sample new data from it. |
| **Autoregressive generation** | `P(text) = ∏ P(tokenₜ \| earlier tokens)`. Predict, sample, append, repeat. |
| **Bigram model** | A language model that conditions on one previous word. An LLM with the context and learning removed. |
| **Autoregressive family** | Text and code. Predicts the next token. |
| **Diffusion family** | Images, video, audio. Learns to remove noise; slow to sample, stable to train. |
| **GAN** | Generator vs discriminator. Fast to sample, unstable to train, prone to mode collapse. |
| **VAE** | Encoder and decoder around a latent space. Stable, but samples are blurrier. |
| **Foundation model** | A large model pretrained on broad data and adapted to many tasks. |
| **Why LLMs hallucinate** | They are trained for plausible continuations, not verified facts. |

[Full explanations →](../01-beginner/01-genai-landscape/README.md)

---

## Module 02 — Prompt Engineering

| Concept | One-liner |
|---|---|
| **Prompt engineering** | Designing the input so the model does what you want. Cheapest lever, try it first. |
| **System / user / assistant** | Rules and role / the request / the model's earlier replies. The model only sees flattened text. |
| **Zero-shot** | Task description only. |
| **Few-shot** | Add worked examples. Best for teaching a format or label style. |
| **Chain-of-thought** | Ask for step-by-step reasoning first. Helps multi-step problems, costs tokens. |
| **Structured output** | Ask, parse, validate, retry. Never trust the format. |
| **Prompt injection** | Untrusted text containing instructions that hijack the model. Layered defenses, no guarantee. |
| **Direct vs indirect injection** | Typed by the user vs hidden in content the model reads (emails, web pages). Indirect is worse. |
| **Test prompts** | Run variants on a labeled set and compare a metric. Re-run when the model changes. |

[Full explanations →](../01-beginner/02-prompt-engineering/README.md)
