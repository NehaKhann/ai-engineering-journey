# Neha's AI/LLM Engineering Bootcamp — 8 Week Roadmap

**Format per day:** Theory (30 min) → Hands-on code (45 min) → Mini practical task (30 min)
**Pace assumption:** ~1.5-2 hrs/day, 5 days/week.
**LinkedIn rule:** minimum 3 posts per week (marked 📝 below) — mix of "today I learned X," a code screenshot/output, and a week-end mini-project reveal.

---

## WEEK 1 — LLM & Transformer Foundations
- **Day 1:** What is an LLM really doing? Next-token prediction, tokenization, embeddings. Analogy: autocomplete on steroids. 📝 *Post: "How does ChatGPT actually predict words? Started digging in today."*
- **Day 2:** Transformer architecture — attention, Q/K/V, why "attention is all you need" mattered.
- **Day 3:** PyTorch essentials as a tool — tensors, `nn.Module`, forward pass, autograd. 📝 *Post: a code snippet/screenshot of your first tensor ops + one-line takeaway.*
- **Day 4:** Load a small open model (GPT-2/Qwen-0.5B) with `transformers`, run inference, inspect tokens/logits.
- **Day 5:** Mini project: "explain this token by token" notebook. 📝 *Post: screenshot of your notebook output — "Built my first token-by-token LLM explainer."*

## WEEK 2 — Fine-Tuning Fundamentals & SFT
- **Day 1:** What is fine-tuning, why not always prompt-engineer instead, cost/benefit. 📝 *Post: "Fine-tuning vs prompting — when do you actually need one over the other?"*
- **Day 2:** SFT theory — instruction datasets, loss masking, chat templates.
- **Day 3:** Hands-on: SFT a small model on a tiny custom dataset using `trl`'s `SFTTrainer`. 📝 *Post: training loss curve screenshot — "Watching my first fine-tune converge."*
- **Day 4:** Evaluate before/after — see the behavior change.
- **Day 5:** Practical: fine-tune a model into a custom persona/style. 📝 *Post: before/after output comparison — "Fine-tuned my first LLM today."*

## WEEK 3 — LoRA, QLoRA, Quantization & Unsloth
- **Day 1:** Why full fine-tuning is expensive — LoRA (low-rank adapters), rank/alpha explained simply. 📝 *Post: "Why train billions of parameters when you can train millions? LoRA explained."*
- **Day 2:** Hands-on LoRA fine-tuning the "manual" way with `peft` — see exactly what's happening under the hood.
- **Day 3:** Quantization theory — FP32 → FP16 → INT8 → INT4, trade-offs. 📝 *Post: a simple visual/table you make of the precision trade-off.*
- **Day 4:** QLoRA = quantized base + LoRA adapters. Hands-on 4-bit QLoRA fine-tune with `bitsandbytes` (manual approach).
- **Day 5:** Compare full fine-tune vs LoRA vs QLoRA — memory/time/quality. 📝 *Post: comparison table screenshot — "Trained a model on my laptop instead of a GPU cluster."*
- **Day 6:** Redo the same QLoRA fine-tune using **Unsloth** instead — compare speed and VRAM usage directly against Day 4's manual run. 📝 *Post: real before/after numbers — "Same fine-tune, half the time, less than half the memory — here's why."*

## WEEK 4 — RLHF & PPO
- **Day 1:** RLHF theory — why SFT alone isn't enough, reward models, human preference data. 📝 *Post: "How do models learn from human feedback? RLHF, explained simply."*
- **Day 2:** Reward model training — hands-on with a tiny preference dataset.
- **Day 3:** PPO theory (policy, reward, KL penalty) via a coaching-an-employee analogy. 📝 *Post: your own analogy explanation of PPO — good engagement post.*
- **Day 4:** Hands-on: simple PPO fine-tuning loop with `trl`'s `PPOTrainer`.
- **Day 5:** Practical: see how PPO shifts outputs toward preferred responses. 📝 *Post: before/after PPO output comparison.*

## WEEK 5 — DPO & GRPO
- **Day 1:** DPO theory — direct preference optimization, why it skips the reward model. 📝 *Post: "DPO — the simpler alternative to PPO that's replacing it almost everywhere."*
- **Day 2:** Hands-on DPO fine-tuning with `trl`'s `DPOTrainer`.
- **Day 3:** GRPO theory (used in DeepSeek-R1) — group-relative rewards, why it's cheaper than PPO. 📝 *Post: "The training trick behind DeepSeek-R1 — GRPO explained."*
- **Day 4:** Simplified GRPO walkthrough (conceptual + code) — including a look at Unsloth's GRPO support, which cuts VRAM usage significantly for RL-based fine-tuning.
- **Day 5:** Practical: PPO vs DPO vs GRPO comparison table — what Gulf AI teams are adopting. 📝 *Post: your comparison table — strong "thought leadership" style post.*

## WEEK 6 — RAG Systems
- **Day 1:** RAG theory — why retrieval beats fine-tuning for fresh/private knowledge. 📝 *Post: "Why RAG might matter more than fine-tuning for most real businesses."*
- **Day 2:** Embeddings + vector databases (FAISS/Chroma) hands-on.
- **Day 3:** Build a basic RAG pipeline: chunk → embed → retrieve → generate. 📝 *Post: architecture diagram of your RAG pipeline.*
- **Day 4:** Add re-ranking + prompt templates to improve RAG quality.
- **Day 5:** Practical: RAG chatbot over real documents (e.g. GDPR project docs). 📝 *Post: demo screenshot — "Built my first RAG chatbot that answers from my own documents."*

## WEEK 7 — OCR & Document AI
- **Day 1:** OCR theory — classical (Tesseract) vs modern vision-language OCR models. 📝 *Post: "OCR isn't just 'reading text from images' anymore — here's what changed."*
- **Day 2:** Hands-on Tesseract/EasyOCR on scanned docs.
- **Day 3:** OCR + LLM pipelines (extract text → structure with an LLM) — relevant to your GDPR tool. 📝 *Post: before/after of a scanned doc → structured JSON.*
- **Day 4:** Hands-on: "scan a form → extract structured JSON" mini pipeline.
- **Day 5:** Practical: combine OCR + RAG (documents in, searchable Q&A out). 📝 *Post: demo of OCR+RAG combo — "Turning scanned documents into a searchable knowledge base."*

## WEEK 8 — Capstone: Saudi Market Research + Project Build
- **Day 1:** Research what Saudi/Gulf companies are hiring/building for (NEOM, STC, Aramco Digital, Vision 2030 AI initiatives, local startups) — done live together. 📝 *Post: "What Saudi Arabia's AI market is actually looking for right now — my research."*
- **Day 2:** Shortlist 3 project ideas combining your new skills + market demand (e.g. Arabic RAG + OCR for gov docs, fine-tuned Arabic assistant, compliance-doc RAG tool tying into your GDPR project). 📝 *Post: your shortlist + reasoning — good visibility post.*
- **Day 3-4:** Build a portfolio-ready mini version of the chosen project.
- **Day 5:** Document it, prep README + demo. 📝 *Post: full project launch post with a demo screenshot/video.*

---

### How we'll run each day
Tell me "Day X" (or "next day") and I'll teach that full session: theory with an analogy, a runnable code example, a small practical task, and a ready-to-post LinkedIn caption for each 📝 marked day — same conversational, plain-language, SecureNet-style analogies we've used before.
