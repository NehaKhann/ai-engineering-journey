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
