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
