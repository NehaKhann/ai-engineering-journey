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

---

## Module 03 — LLM APIs

| Concept | One-liner |
|---|---|
| **Token** | The chunk of text a model reads and writes. Cost, speed, and context limits are all in tokens. |
| **Stateless API** | The model remembers nothing between calls. You re-send the whole conversation each turn. |
| **Cost formula** | `input × input price + output × output price`, quoted per million tokens. Output costs more. |
| **Why long chats are costly** | History is re-sent every turn, so total input grows roughly with the square of the turns. |
| **Cost levers** | Shorter prompts, trim or summarize history, cap `max_tokens`, cache prefixes, batch, cheaper model. |
| **Streaming** | Send tokens as generated. Same total time, far lower time to first token. |
| **Retry policy** | Retry 429, 5xx, timeouts with exponential backoff plus jitter. Never retry 400, 401, 404. |
| **`stop_reason`** | Check it. `max_tokens` means the reply was cut off. |
| **Hosted vs self-hosted** | Hosted: best quality, easy start. Self-hosted: privacy, fixed cost at high volume, full control. |

[Full explanations →](../01-beginner/03-llm-apis/README.md)

---

## Module 04 — Embeddings & Vector Search

| Concept | One-liner |
|---|---|
| **Embedding** | A vector of numbers where similar meaning means nearby vectors. |
| **Cosine similarity** | Compares direction, ignores length. For unit vectors it equals the dot product. |
| **Semantic search** | Embed the query and documents, rank by similarity. Survives paraphrasing. |
| **Keyword search wins on** | Exact terms: IDs, error codes, names, rare jargon. Use **hybrid** search to get both. |
| **Brute force search** | Compare against every vector. Linear time and memory. Fine for small and medium collections. |
| **ANN index (HNSW, IVF)** | Approximate search that trades a little recall for big speed gains at scale. |
| **Metadata filter** | Restrict a search by category, tenant, date, or permissions. |
| **Similarity threshold** | Reject weak matches so the app can say "not found". Tune on your own data. |
| **Embedding weaknesses** | Negation, opposites, numbers, exact identifiers. Finds related text, not true text. |
| **Switching models** | Re-embed everything. Vectors from different models are not comparable. |

[Full explanations →](../01-beginner/04-embeddings-vector-search/README.md)

---

## Module 05 — RAG

| Concept | One-liner |
|---|---|
| **RAG** | Retrieve relevant chunks, put them in the prompt, generate a grounded answer with citations. |
| **RAG vs fine-tuning** | RAG adds facts (fresh, citable, permissioned). Fine-tuning changes behavior and style. |
| **Pipeline** | Offline: chunk, embed, index. Online: embed query, retrieve top-k, build prompt, generate. |
| **Debugging** | Right chunk retrieved? No: retrieval failure. Yes but bad answer: generation failure. Measure separately. |
| **Chunk size** | Smallest chunks that keep a fact intact. Big chunks cost tokens and dilute matches. Add overlap. |
| **Improve retrieval** | Hybrid search, reranker, query rewriting, metadata filters, better embeddings, tune `k`. |
| **Hallucination controls** | Good retrieval, "answer only from sources", citations, similarity threshold, groundedness checks. |
| **"I don't know" trade-off** | Stops invented answers but can cause wrongful refusals. Measure both. |
| **RAG vs long context** | Long context costs per question and buries facts. RAG is cheaper, fresher, and can filter by permission. |

[Full explanations →](../02-intermediate/05-rag/README.md)

---

## Module 06 — Evaluation & LLM-as-Judge

| Concept | One-liner |
|---|---|
| **Golden set** | Questions paired with known-correct answers. Phrase them like real users, include unanswerable ones. |
| **Evaluate separately** | Retrieval (Hit@k, recall@k, MRR, no LLM) and generation (correctness, faithfulness). |
| **Hit@k / recall@k / MRR / nDCG** | Any relevant in top k / share of all relevant in top k / mean of 1/rank of first hit / graded ranking quality. |
| **Faithfulness vs correctness** | Supported by the retrieved context vs true. An answer can be faithful to a wrong document. |
| **Bootstrap interval** | Resample the questions to see if a gap could be noise. If the interval includes 0, no proven difference. |
| **String metrics** | Cheap, but exact match and F1 punish correct paraphrases. |
| **LLM judge** | Handles open-ended answers. Must be validated against human labels. |
| **Judge biases** | Position, verbosity, self-preference, inconsistency. Test for each. |
| **Offline vs online** | Fixed test set before release vs real usage signals after. Failures found online become new offline tests. |

[Full explanations →](../02-intermediate/06-evaluation/README.md)

---

## Module 07 — Tool Use & Agents

| Concept | One-liner |
|---|---|
| **Function calling** | Model outputs a structured request to call a function. Your code runs it and returns the result. The model never executes anything. |
| **Agent** | An LLM that picks its next step at runtime, in a loop, until done. |
| **Workflow vs agent** | Code fixes the steps vs the model decides them. Use the simplest structure that works. |
| **ReAct** | Alternate reasoning and acting: think, call a tool, observe, repeat. |
| **Loop limits** | Max steps, timeout, spend budget, repeat-call detection, graceful failure. |
| **Tool design** | Few tools, precise descriptions, tight schemas, concise results, helpful errors. Tool schemas cost tokens on every call. |
| **Untrusted arguments** | The model picks them and can be tricked. Validate and whitelist. Never `eval`. |
| **Indirect injection** | Instructions hidden in content the agent reads. Least privilege, human approval, treat tool output as data. |
| **Compounding errors** | 90% per step over 5 steps is about 59% overall. Keep tasks short, add checks. |
| **Testing agents** | Scripted fake model for the loop, then a scored task set for answer, tool choice, steps, and cost. |

[Full explanations →](../02-intermediate/07-tool-use-agents/README.md)

---

## Module 08 — Guardrails, Safety & Hallucination

| Concept | One-liner |
|---|---|
| **Guardrail** | A check around the model. Every one is a classifier with false positives and false negatives. |
| **Layers** | Input (PII, injection), model/tools (least privilege), output (groundedness, format, safety filters). |
| **PII redaction** | Regex plus checksums for well-formed data. Misses obfuscated and free-text data, so add NER. Redact before the call and before logging. |
| **Direct vs indirect injection** | Typed by the user vs hidden in content the model reads. Indirect is the dangerous one. |
| **Injection detectors** | Rules: precise but brittle. Embeddings: catch paraphrases but need a tuned threshold. Score long documents per sentence. |
| **Hallucination check** | NLI (does the source entail the claim?) beats word overlap and embeddings, which miss wrong numbers. Only checks against the sources. |
| **Threshold choice** | Set by the cost of each error type in your application. Compare detectors with AUC. |
| **Baseline first** | Measure the attack success rate without the defense. A defense with cost and no measured benefit isn't free. |
| **Red-teaming** | Attack your own system, record results, keep every finding as a regression test. |

[Full explanations →](../02-intermediate/08-guardrails-safety/README.md)
