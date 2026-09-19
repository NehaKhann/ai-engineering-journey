# 🎤 Interview Prep

Every question from the Generative AI track, indexed by topic and linked to the module that answers it.

Questions live inside their modules so you learn and practice together. Use this page to revise.

For one-line answers, see the [Cheatsheet](cheatsheet.md).

---

## 🟢 Beginner

### Module 01 — The Generative AI Landscape

| Question | Answer in |
|---|---|
| What is the difference between a generative and a discriminative model? | [Q1](../01-beginner/01-genai-landscape/README.md#-interview-qa) |
| Name the main families of generative models and what each is used for. | [Q2](../01-beginner/01-genai-landscape/README.md#-interview-qa) |
| How does an LLM actually generate text? | [Q3](../01-beginner/01-genai-landscape/README.md#-interview-qa) |
| Is a bigram model a language model? How does it differ from GPT? | [Q4](../01-beginner/01-genai-landscape/README.md#-interview-qa) |
| Compare GANs, VAEs, and diffusion models. | [Q5](../01-beginner/01-genai-landscape/README.md#-interview-qa) |
| What is a foundation model? | [Q6](../01-beginner/01-genai-landscape/README.md#-interview-qa) |

### Module 02 — Prompt Engineering

| Question | Answer in |
|---|---|
| What is prompt engineering, and why try it before fine-tuning? | [Q1](../01-beginner/02-prompt-engineering/README.md#-interview-qa) |
| Explain zero-shot, few-shot, and chain-of-thought prompting. | [Q2](../01-beginner/02-prompt-engineering/README.md#-interview-qa) |
| What are system, user, and assistant messages? | [Q3](../01-beginner/02-prompt-engineering/README.md#-interview-qa) |
| How do you get reliable structured output such as JSON from an LLM? | [Q4](../01-beginner/02-prompt-engineering/README.md#-interview-qa) |
| What is prompt injection and how do you defend against it? | [Q5](../01-beginner/02-prompt-engineering/README.md#-interview-qa) |
| How do you know whether one prompt is better than another? | [Q6](../01-beginner/02-prompt-engineering/README.md#-interview-qa) |

### Module 03 — LLM APIs

| Question | Answer in |
|---|---|
| What is a token, and why does it matter to an application developer? | [Q1](../01-beginner/03-llm-apis/README.md#-interview-qa) |
| Why does a chatbot get more expensive as the conversation gets longer? | [Q2](../01-beginner/03-llm-apis/README.md#-interview-qa) |
| How would you estimate and reduce the cost of an LLM feature? | [Q3](../01-beginner/03-llm-apis/README.md#-interview-qa) |
| Why stream responses, and what does it change? | [Q4](../01-beginner/03-llm-apis/README.md#-interview-qa) |
| How do you handle rate limits and transient failures? | [Q5](../01-beginner/03-llm-apis/README.md#-interview-qa) |
| When would you use a hosted API instead of a self-hosted open model? | [Q6](../01-beginner/03-llm-apis/README.md#-interview-qa) |

### Module 04 — Embeddings & Vector Search

| Question | Answer in |
|---|---|
| What is an embedding, and what is it used for? | [Q1](../01-beginner/04-embeddings-vector-search/README.md#-interview-qa) |
| Cosine similarity, dot product, Euclidean distance: what is the difference, and why normalize? | [Q2](../01-beginner/04-embeddings-vector-search/README.md#-interview-qa) |
| When is semantic search better than keyword search, and when is it worse? | [Q3](../01-beginner/04-embeddings-vector-search/README.md#-interview-qa) |
| How do vector databases search millions of vectors quickly? | [Q4](../01-beginner/04-embeddings-vector-search/README.md#-interview-qa) |
| What are the limitations of embedding-based retrieval? | [Q5](../01-beginner/04-embeddings-vector-search/README.md#-interview-qa) |
| How would you choose and evaluate an embedding model? | [Q6](../01-beginner/04-embeddings-vector-search/README.md#-interview-qa) |

### 🏆 Beginner Project — Support Ticket Assistant

How to present it in an interview, with the questions it prepares you for, is in the [project README](../projects/01-support-ticket-assistant/README.md#-how-to-talk-about-this-project-in-an-interview).

---

## 🟡 Intermediate

### Module 05 — Retrieval-Augmented Generation (RAG)

| Question | Answer in |
|---|---|
| What is RAG, and why use it instead of fine-tuning? | [Q1](../02-intermediate/05-rag/README.md#-interview-qa) |
| Walk me through a RAG pipeline, and tell me how you would debug a wrong answer. | [Q2](../02-intermediate/05-rag/README.md#-interview-qa) |
| How do you choose chunk size and overlap? | [Q3](../02-intermediate/05-rag/README.md#-interview-qa) |
| How do you reduce hallucination in a RAG system? | [Q4](../02-intermediate/05-rag/README.md#-interview-qa) |
| Retrieval is missing relevant chunks. How do you improve it? | [Q5](../02-intermediate/05-rag/README.md#-interview-qa) |
| Why not just paste all the documents into a long context window? | [Q6](../02-intermediate/05-rag/README.md#-interview-qa) |

### Module 06 — Evaluation & LLM-as-Judge

| Question | Answer in |
|---|---|
| How would you evaluate a RAG system? | [Q1](../02-intermediate/06-evaluation/README.md#-interview-qa) |
| Explain Hit@k, recall@k, MRR, and nDCG. When do you use each? | [Q2](../02-intermediate/06-evaluation/README.md#-interview-qa) |
| What is LLM-as-a-judge, and what are its pitfalls? | [Q3](../02-intermediate/06-evaluation/README.md#-interview-qa) |
| Your new prompt scores 3 points higher on your eval set. How do you know it's better? | [Q4](../02-intermediate/06-evaluation/README.md#-interview-qa) |
| How do you build a good golden set? | [Q5](../02-intermediate/06-evaluation/README.md#-interview-qa) |
| What is the difference between offline and online evaluation? | [Q6](../02-intermediate/06-evaluation/README.md#-interview-qa) |

### Module 07 — Tool Use & Agents

| Question | Answer in |
|---|---|
| What is function calling, and how does it work? | [Q1](../02-intermediate/07-tool-use-agents/README.md#-interview-qa) |
| What is an agent, and how is it different from a simple LLM chain or workflow? | [Q2](../02-intermediate/07-tool-use-agents/README.md#-interview-qa) |
| Explain the ReAct pattern and how you stop an agent from looping forever. | [Q3](../02-intermediate/07-tool-use-agents/README.md#-interview-qa) |
| How do you design good tools for an LLM? | [Q4](../02-intermediate/07-tool-use-agents/README.md#-interview-qa) |
| What are the security risks of agents, and how do you mitigate them? | [Q5](../02-intermediate/07-tool-use-agents/README.md#-interview-qa) |
| How do you evaluate an agent, and why are agents unreliable? | [Q6](../02-intermediate/07-tool-use-agents/README.md#-interview-qa) |

### Module 08 — Guardrails, Safety & Hallucination

| Question | Answer in |
|---|---|
| What guardrails would you put around an LLM application? | [Q1](../02-intermediate/08-guardrails-safety/README.md#-interview-qa) |
| What is prompt injection, and how do you defend against it? | [Q2](../02-intermediate/08-guardrails-safety/README.md#-interview-qa) |
| How do you detect hallucinations in a RAG system? | [Q3](../02-intermediate/08-guardrails-safety/README.md#-interview-qa) |
| Every guardrail has false positives and false negatives. How do you choose the balance? | [Q4](../02-intermediate/08-guardrails-safety/README.md#-interview-qa) |
| How do you handle sensitive data (PII) in an LLM application? | [Q5](../02-intermediate/08-guardrails-safety/README.md#-interview-qa) |
| What is red-teaming, and how would you do it for an LLM feature? | [Q6](../02-intermediate/08-guardrails-safety/README.md#-interview-qa) |

### Module 09 — Prompting vs RAG vs Fine-Tuning

| Question | Answer in |
|---|---|
| When would you use prompting, RAG, or fine-tuning? | [Q1](../02-intermediate/09-prompting-vs-rag-vs-finetuning/README.md#-interview-qa) |
| Why is fine-tuning a poor way to add knowledge? | [Q2](../02-intermediate/09-prompting-vs-rag-vs-finetuning/README.md#-interview-qa) |
| When is fine-tuning the right choice? | [Q3](../02-intermediate/09-prompting-vs-rag-vs-finetuning/README.md#-interview-qa) |
| A policy changes tomorrow. How does each approach cope? | [Q4](../02-intermediate/09-prompting-vs-rag-vs-finetuning/README.md#-interview-qa) |
| What are the cost trade-offs between RAG and fine-tuning? | [Q5](../02-intermediate/09-prompting-vs-rag-vs-finetuning/README.md#-interview-qa) |
| How do you decide with data instead of opinion? | [Q6](../02-intermediate/09-prompting-vs-rag-vs-finetuning/README.md#-interview-qa) |

### Module 10 — Multimodal & Diffusion Models

| Question | Answer in |
|---|---|
| How does a diffusion model generate an image? | [Q1](../02-intermediate/10-multimodal-diffusion/README.md#-interview-qa) |
| Why is diffusion slow at generation, and how is it sped up? | [Q2](../02-intermediate/10-multimodal-diffusion/README.md#-interview-qa) |
| What is CLIP, and what is zero-shot classification? | [Q3](../02-intermediate/10-multimodal-diffusion/README.md#-interview-qa) |
| What are the known weaknesses of multimodal models, and how would you test for them? | [Q4](../02-intermediate/10-multimodal-diffusion/README.md#-interview-qa) |
| How does a text prompt control the image in a text-to-image model? | [Q5](../02-intermediate/10-multimodal-diffusion/README.md#-interview-qa) |
| How do you evaluate a generative image model? | [Q6](../02-intermediate/10-multimodal-diffusion/README.md#-interview-qa) |

### Module 11 — Deployment & LLMOps

| Question | Answer in |
|---|---|
| What are prefill and decode, and why does it matter? | [Q1](../02-intermediate/11-deployment-llmops/README.md#-interview-qa) |
| What is batching, and what is the trade-off? | [Q2](../02-intermediate/11-deployment-llmops/README.md#-interview-qa) |
| How would you cache LLM responses, and what can go wrong? | [Q3](../02-intermediate/11-deployment-llmops/README.md#-interview-qa) |
| What is quantization, and what are its trade-offs? | [Q4](../02-intermediate/11-deployment-llmops/README.md#-interview-qa) |
| What would you monitor for an LLM service in production? | [Q5](../02-intermediate/11-deployment-llmops/README.md#-interview-qa) |
| How would you decide between an API and hosting a model yourself? | [Q6](../02-intermediate/11-deployment-llmops/README.md#-interview-qa) |

---

## 🧭 System Design Questions

Open-ended design questions such as *"Design a chatbot over 10,000 internal documents"* combine the modules. Use the [Ask Your Documents](../projects/02-ask-your-documents/README.md) and [Production LLM API](../projects/04-production-llm-api/README.md) projects as worked examples: their READMEs list the design decisions and the interview questions they prepare you for.

