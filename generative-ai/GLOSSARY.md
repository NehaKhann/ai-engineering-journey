# 📖 Glossary

Plain-English definitions of the terms used in the Generative AI track, in alphabetical order. Each entry says where you'll meet the idea in practice.

**How to use this:** if a word in a module trips you up, look it up here, then go back. You don't need to memorize any of it. The modules explain each idea with a working example.

---

**Agent.** A model in a loop that decides which tool to use next, uses it, looks at the result, and repeats until it has an answer. *(Module 07)*

**API.** A way for your program to ask another program to do something, usually over the internet. A "hosted model API" lets your code send text to a model running on someone else's computers and get a reply. *(Module 03)*

**Attention.** The mechanism inside a Transformer that lets each word look at the other words in the text to work out what it means in context. *(Module 01, and Week 1 of the weekly course)*

**Autoregressive.** Generating one piece at a time, where each new piece depends on everything before it. Language models like GPT work this way: they predict the next token, add it, and repeat. *(Module 01)*

**Batching.** Processing several requests together in one pass so the hardware does more useful work at once. It raises total throughput but makes each request wait a bit longer. *(Module 11)*

**BM25.** A classic keyword-search formula that ranks documents by how many of the query's words they contain, giving more weight to rare words. *(Module 06)*

**Chunk.** A small piece of a document, such as 60 words, that is stored and retrieved on its own. Documents are split into chunks because you can't paste whole libraries into a prompt. *(Module 05)*

**Context window.** The most text (measured in tokens) a model can consider at one time: your prompt, the conversation so far, and its reply, all together. *(Module 03, and Week 1)*

**Cosine similarity.** A number from about 0 to 1 saying how close in meaning two embeddings are. 1 means nearly identical, 0 means unrelated. *(Module 04)*

**Decode.** The phase where a model produces its reply one token at a time, after it has read the prompt (see *Prefill*). *(Module 11)*

**Diffusion model.** A model that generates images (or other data) by starting from random noise and removing a little noise at a time until a clean result appears. *(Module 10)*

**Embedding.** A list of numbers that represents the meaning of a piece of text (or an image). Texts with similar meaning get similar lists, which makes "search by meaning" possible. *(Module 04)*

**Evaluation ("eval").** Measuring how well a system works on a fixed set of test cases, so you can tell whether a change helped. *(Module 06)*

**Few-shot prompting.** Putting a few worked examples in the prompt so the model can see the pattern you want. Zero-shot means no examples. *(Module 02)*

**Fine-tuning.** Training an existing model further on your own examples, which changes the model itself. Good for style and format, poor for storing facts that change. *(Module 09, and Weeks 2 and 3)*

**Foundation model.** A large model trained on broad data that can be adapted to many tasks through prompting, retrieval, or fine-tuning. *(Module 01)*

**Function calling / tool use.** Letting a model ask for one of your functions to be run (for example a calculator or a search). The model only *asks*; your code runs it and returns the result. *(Module 07)*

**Golden set.** A collection of test questions paired with known-correct answers, used to measure a system. *(Module 06)*

**Guardrail.** A check placed around a model to catch problems, such as hiding personal data, blocking prompt injection, or verifying an answer against its sources. *(Module 08)*

**Hallucination.** When a model states something false with confidence. It happens because models are trained to produce *plausible* text, not to verify facts. *(Modules 01, 05, and 08)*

**Hit@k.** Of all the test questions, the share where the right document appears in the top k search results. Hit@3 means "in the top three." *(Module 06)*

**Inference.** Using a trained model to produce an output, as opposed to training it. *(Throughout)*

**LLM (large language model).** A model trained on huge amounts of text to predict the next token, which lets it write, summarize, answer questions, and more. *(Module 01)*

**LLM-as-judge.** Using a language model to grade another model's answers. It handles paraphrases that simple string matching can't, but it has biases and must be checked against human labels. *(Module 06)*

**LoRA.** A way to fine-tune by training small add-on matrices instead of every weight in the model, so it needs far less memory and time. *(Module 09, and Week 3)*

**MRR (mean reciprocal rank).** A search score that rewards putting the right result *higher* in the list. If the right document is ranked 1st you get 1, 2nd gets 0.5, 3rd gets 0.33, averaged over all questions. *(Module 06)*

**Multimodal.** Handling more than one kind of data, such as text and images together. *(Module 10)*

**NLI (natural language inference).** A model that judges whether one text *supports*, *contradicts*, or is *unrelated to* a claim. Useful for checking whether an answer is backed by its sources. *(Module 08)*

**Parameter.** One of the numbers a model learns during training. "A 0.5-billion-parameter model" has about 500 million of them. Bigger models usually know and do more but need more memory and time. *(Throughout)*

**PII.** Personally identifiable information: emails, phone numbers, card numbers, and similar data that should not be sent around or logged carelessly. *(Module 08)*

**Prefill.** The phase where a model reads your whole prompt in one pass, before it starts writing. Its length decides how long you wait for the *first* word. *(Module 11)*

**Prompt.** The text you give a model. Prompt engineering is writing that text so the model does what you want. *(Module 02)*

**Prompt injection.** Hidden instructions in text the model reads (an email, a web page, a file) that try to take over its behavior. *Indirect* injection is the kind hidden in content, and it is the dangerous one. *(Modules 02, 07, and 08)*

**Prompt caching / response caching.** Reusing earlier work instead of repeating it. A response cache returns a stored answer for a repeated question. *(Modules 03 and 11)*

**Quantization.** Storing a model's numbers with fewer bits so it is smaller and often faster, usually at some cost in quality. *(Module 11, and Week 3)*

**RAG (retrieval-augmented generation).** Looking up relevant text from your own documents and pasting it into the prompt so the model answers from that evidence instead of guessing. *(Module 05)*

**Rate limit.** A cap on how many requests a caller may make in a period. Going over it usually returns an HTTP 429 error. *(Modules 03 and 11)*

**Reranker.** A second, more careful model that re-scores the top few search results to put the best ones first. *(Module 05)*

**Red-teaming.** Attacking your own system on purpose to find weaknesses before someone else does. *(Module 08)*

**Sampling / temperature.** How a model picks the next token. Low temperature is predictable, high temperature is more varied. *(Module 01, and Week 1)*

**Streaming.** Sending a reply piece by piece as it is generated, so users see text almost immediately instead of waiting for the whole answer. *(Modules 03 and 11)*

**System prompt.** The instructions that set a model's role and rules, written by the developer and separate from the user's message. *(Module 02)*

**Token.** The chunk of text a model reads and writes, often a word or part of a word. Cost, speed, and limits are all counted in tokens. *(Module 03)*

**Tokenizer.** The component that splits text into tokens. Each model family has its own, so token counts differ between models. *(Module 03)*

**Vector database.** A database built to store embeddings and quickly find the ones closest to a query. *(Module 04)*

**Zero-shot.** Doing a task with no examples, just a description. CLIP's zero-shot image classification, for instance, classifies images into categories it was never trained on. *(Modules 02 and 10)*
