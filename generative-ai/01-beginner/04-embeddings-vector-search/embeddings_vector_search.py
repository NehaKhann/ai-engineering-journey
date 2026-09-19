# pip: transformers>=4.56 torch scikit-learn matplotlib chromadb
# %% [markdown]
# # 📘 Module 04 — Embeddings & Vector Search
#
# **Generative AI Track • Beginner**
#
# How do you search for "my card got charged twice" and find a document titled "Duplicate payment
# refunds", when the two share almost no words? You turn text into **numbers that capture meaning**.
# These numbers are called **embeddings**, and they are the foundation of retrieval-augmented
# generation (RAG), semantic search, recommendations, and duplicate detection.
#
# In this notebook you will:
#
# - Turn sentences into vectors and compare them with **cosine similarity**
# - Build search **from scratch** in a few lines of NumPy
# - Compare keyword search against semantic search on the same queries
# - See how brute-force search scales, and why vector databases exist
# - Use a vector database (Chroma) with metadata filters
# - Learn what embeddings are **bad** at

# %%
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

OUT = Path(__file__).parent / "assets" if "__file__" in globals() else Path("assets")
OUT.mkdir(exist_ok=True)

# A small, fast embedding model (about 90 MB) that runs on a laptop CPU.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL).eval()


def embed(texts):
    """Turn a list of strings into an array of unit-length vectors, one row per string.

    This is what the sentence-transformers library does for this model:
      1. run the text through the model to get one vector per token
      2. average the token vectors (ignoring padding), called "mean pooling"
      3. scale the result to length 1
    """
    batch = embed_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        token_vectors = embed_model(**batch).last_hidden_state  # (texts, tokens, 384)
    mask = batch["attention_mask"].unsqueeze(-1).float()  # 1 for real tokens, 0 for padding
    pooled = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1)
    return torch.nn.functional.normalize(pooled, dim=1).numpy()


# %% [markdown]
# ## 1. What is an embedding?
#
# An embedding model reads text and outputs a **list of numbers** (a vector). The model is trained
# so that texts with similar meaning get vectors that point in similar directions.

# %%
vector = embed(["How do I reset my password?"])[0]
print("Shape:", vector.shape, "(one number per dimension)")
print("First 8 numbers:", np.round(vector[:8], 3))
print("Length of the vector:", round(float(np.linalg.norm(vector)), 3), "(we normalized it to 1)")

# %% [markdown]
# (Many tutorials use the `sentence-transformers` library, which wraps exactly the steps in `embed`
# above in one call: `SentenceTransformer("all-MiniLM-L6-v2").encode(texts)`.)
#
# The individual numbers mean nothing on their own. What matters is the **distance between
# vectors**. We measure it with **cosine similarity**, the cosine of the angle between two vectors:
#
# - `1.0` means the same direction (very similar meaning)
# - `0.0` means unrelated
# - negative means opposite (rare in practice for text)
#
# When vectors are normalized to length 1, cosine similarity is just a **dot product**.

# %%
def cosine_similarity(a, b):
    """Cosine similarity written out in full, so nothing is hidden."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What is the best pizza topping?",
]
vectors = embed(sentences)

print(f'"{sentences[0]}"')
print(f'  vs "{sentences[1]}": {cosine_similarity(vectors[0], vectors[1]):.2f}   <- same meaning, different words')
print(f'  vs "{sentences[2]}": {cosine_similarity(vectors[0], vectors[2]):.2f}   <- unrelated')

# %% [markdown]
# ## 2. Keyword search vs semantic search
#
# We have a tiny knowledge base of 15 support articles. Real users do not phrase questions the way
# the articles are written, so we test with **paraphrased** queries that share few words with the
# right answer.

# %%
documents = [
    # billing
    "Refunds are issued to the original payment method within 5 to 7 business days after approval.",
    "If you were charged twice for the same subscription, contact billing and we will reverse the duplicate payment.",
    "Invoices are emailed on the first day of each month and can be downloaded from the billing page.",
    "You can change your plan at any time and the price difference is prorated on your next invoice.",
    "We accept Visa, Mastercard, and PayPal. Bank transfers are available for annual plans.",
    # technical
    "If the app crashes on launch, clear the cache and reinstall the latest version.",
    "Slow dashboards are usually caused by large date ranges. Narrow the range or enable data caching.",
    "API requests return 429 when you exceed the rate limit. Wait a few seconds and retry with backoff.",
    "Exports to CSV fail when a report has more than one million rows. Split the report into smaller parts.",
    "Webhooks that fail three times in a row are disabled. Re-enable them from the developer settings.",
    # account
    "To reset your password, choose Forgot password on the sign in page and follow the emailed link.",
    "You can change the email on your profile from Settings, then confirm it using the code we send.",
    "Two-factor authentication can be turned on from Security settings using an authenticator app.",
    "To delete your account permanently, open Privacy settings and choose Delete account. This cannot be undone.",
    "Team owners can invite new members by email and assign them a viewer, editor, or admin role.",
]
categories = ["billing"] * 5 + ["technical"] * 5 + ["account"] * 5

# (query, index of the document that answers it)
queries = [
    ("my card got hit two times for the same month", 1),
    ("I forgot my login credentials", 10),
    ("the page takes forever to load charts", 6),
    ("how do I get my money back", 0),
    ("app keeps closing as soon as I open it", 5),
    ("too many requests error from your API", 7),
    ("I want to leave and remove all my data", 13),
    ("add my colleague to the workspace", 14),
    ("spreadsheet download comes out empty for huge reports", 8),
    ("can I pay by wire", 4),
]

STOPWORDS = set("a an and are as at be by can do for from how i if in is it me my of on or the to we you your".split())


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def keyword_search(query, top_k=1):
    """Rank documents by how many (non-stopword) words they share with the query."""
    query_words = words(query)
    scores = [len(query_words & words(doc)) for doc in documents]
    return list(np.argsort(scores)[::-1][:top_k]), scores


# %% [markdown]
# ### Semantic search from scratch
#
# 1. Embed every document once and keep the matrix.
# 2. Embed the query.
# 3. Score = dot product of the query with every document.
# 4. Return the highest scores.
#
# That is the whole algorithm. Vector databases are this, made fast and durable.

# %%
doc_vectors = embed(documents)  # shape: (15, 384)


def semantic_search(query, top_k=1):
    query_vector = embed([query])[0]
    scores = doc_vectors @ query_vector  # one dot product per document
    return list(np.argsort(scores)[::-1][:top_k]), scores


print(f"{'Query':<55}{'Keyword':<10}{'Semantic'}")
print("-" * 75)
keyword_hits = semantic_hits = 0
for query, answer in queries:
    keyword_top, _ = keyword_search(query)
    semantic_top, _ = semantic_search(query)
    keyword_ok, semantic_ok = keyword_top[0] == answer, semantic_top[0] == answer
    keyword_hits += keyword_ok
    semantic_hits += semantic_ok
    print(f"{query:<55}{'✅' if keyword_ok else '❌':<10}{'✅' if semantic_ok else '❌'}")

print("-" * 75)
print(f"{'Top-1 accuracy':<55}{keyword_hits}/{len(queries):<8}{semantic_hits}/{len(queries)}")

# %% [markdown]
# Keyword search needs the same words, so it only got the queries that happened to reuse them.
# Semantic search matches **meaning** and survived the paraphrasing.
#
# Don't over-trust 10 out of 10: this is a tiny, clean collection where every topic is distinct.
# With thousands of similar articles it gets harder (Module 06 shows how to measure that). Two
# things are worth checking now: how **confident** each match is, and what happens when **no
# article answers** the question.

# %%
print(f"{'Query':<55}{'Best':<8}{'Runner-up'}")
print("-" * 72)
best_scores = []
for query, answer in queries:
    top, scores = semantic_search(query, top_k=2)
    best_scores.append(scores[top[0]])
    print(f"{query:<55}{scores[top[0]]:<8.2f}{scores[top[1]]:.2f}")
print(f"\nLowest best-match score on real questions: {min(best_scores):.2f}")

print("\nA question the knowledge base cannot answer:")
for off_topic in ["how do I bake a chocolate cake", "who won the football match yesterday"]:
    top, scores = semantic_search(off_topic)
    print(f'  "{off_topic}" -> best match scores {scores[top[0]]:.2f}: {documents[top[0]][:50]}...')

# %% [markdown]
# Search **always** returns something, even when nothing is relevant. The off-topic questions
# score clearly lower than the real ones, so a **similarity threshold** lets an app say "I could
# not find an answer" instead of showing an unrelated article. Choose the threshold by testing on
# your own data, because scores depend on the model.

# %% [markdown]
# ## 3. A picture of meaning
#
# Embeddings have 384 dimensions, which we cannot draw. **PCA** squashes them to 2 while keeping as
# much structure as it can. Colored by category, the articles should form groups.

# %%
points = PCA(n_components=2).fit_transform(doc_vectors)
colors = {"billing": "#4C72B0", "technical": "#DD8452", "account": "#55A868"}

plt.figure(figsize=(8, 6))
for category, color in colors.items():
    mask = [c == category for c in categories]
    plt.scatter(points[mask, 0], points[mask, 1], s=140, color=color, label=category)
for (x, y), index in zip(points, range(len(documents))):
    plt.annotate(str(index), (x, y), ha="center", va="center", color="white", fontsize=8)
plt.title("15 support articles in embedding space (2D projection)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "embedding_map.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Brute force, and why vector databases exist
#
# Our search compares the query against **every** document. That is fine for 15 documents. How
# does it behave as the collection grows? We measure with random 384-dimensional vectors.

# %%
rng = np.random.default_rng(0)
query_vector = rng.standard_normal(384).astype(np.float32)

print(f"{'Documents':<14}{'Search time':<14}{'Memory'}")
print("-" * 40)
for n in [1_000, 10_000, 100_000, 500_000]:
    matrix = rng.standard_normal((n, 384)).astype(np.float32)
    start = time.perf_counter()
    for _ in range(5):
        top = np.argsort(matrix @ query_vector)[::-1][:5]
    per_search_ms = (time.perf_counter() - start) / 5 * 1000
    print(f"{n:<14,}{per_search_ms:<14.1f}{matrix.nbytes / 1e6:.0f} MB")
    del matrix

# %% [markdown]
# Time and memory grow **linearly** with the number of documents. On this laptop, brute force is
# still quick at half a million vectors, so for small and medium collections it is a perfectly good
# choice. The trouble starts later: at 500,000 vectors it already uses about 770 MB, so ten times
# that would need about 7.7 GB of memory and roughly ten times the search time (an extrapolation,
# not a measurement).
#
# Vector databases exist for that scale, and for what brute force lacks: **approximate nearest
# neighbor (ANN)** indexes such as HNSW that skip most comparisons in exchange for a little
# accuracy, plus durable storage, updates and deletes, and **metadata filtering**.

# %% [markdown]
# ## 5. A vector database in practice (Chroma)
#
# We give Chroma our own embeddings plus **metadata**, then search with and without a filter.

# %%
import chromadb

client = chromadb.EphemeralClient()  # in memory; use PersistentClient(path=...) to save to disk
try:
    client.delete_collection("support_articles")
except Exception:
    pass
collection = client.create_collection("support_articles", metadata={"hnsw:space": "cosine"})

collection.add(
    ids=[f"doc{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=doc_vectors.tolist(),
    metadatas=[{"category": c} for c in categories],
)
print("Documents stored:", collection.count())

query = "I was billed twice"
query_embedding = embed([query]).tolist()

everything = collection.query(query_embeddings=query_embedding, n_results=2)
print(f'\nQuery: "{query}"  (no filter)')
for text, distance in zip(everything["documents"][0], everything["distances"][0]):
    print(f"  distance {distance:.2f}: {text[:70]}...")

filtered = collection.query(query_embeddings=query_embedding, n_results=2, where={"category": "technical"})
print(f'\nQuery: "{query}"  (only category = technical)')
for text, distance in zip(filtered["documents"][0], filtered["distances"][0]):
    print(f"  distance {distance:.2f}: {text[:70]}...")

# %% [markdown]
# Chroma reports **distance** (lower is closer), where cosine distance = `1 - cosine similarity`.
# Metadata filters let you restrict a search by category, customer, date, or permissions, which
# real applications need constantly.

# %% [markdown]
# ## 6. What embeddings are bad at
#
# Embeddings capture **topical similarity**, not logic or truth. Knowing this is what separates
# someone who has used them from someone who has only read about them.

# %%
pairs = [
    ("Negation", "The refund was approved.", "The refund was not approved."),
    ("Opposites", "The server is up.", "The server is down."),
    ("Different numbers", "The plan costs $10 per month.", "The plan costs $1000 per month."),
    ("Word sense", "I deposited money at the bank.", "We had a picnic on the river bank."),
    ("Truly unrelated", "The refund was approved.", "Penguins live in Antarctica."),
]

print(f"{'Case':<20}{'Similarity':<12}Sentences")
print("-" * 80)
for case, first, second in pairs:
    a, b = embed([first, second])
    print(f"{case:<20}{cosine_similarity(a, b):<12.2f}{first!r} vs {second!r}")

# %% [markdown]
# Sentences that mean **opposite** things can score almost as high as paraphrases, because they
# talk about the same topic. So similarity is a good way to **find candidates**, but a poor way to
# decide what is *true*. Real systems retrieve with embeddings and then let a stronger step
# (a reranker or the LLM itself) judge relevance.
#
# ## 🎯 Key Takeaways
#
# - An **embedding** is a vector of numbers where similar meaning means nearby vectors.
# - **Cosine similarity** (a dot product for normalized vectors) measures closeness.
# - Semantic search matches meaning, so it survives paraphrasing where keyword search fails.
# - Brute-force search scales **linearly**. Vector databases use approximate indexes to go faster.
# - Metadata filters let you combine meaning with rules such as category or permissions.
# - Embeddings find **related** text, not necessarily **correct** text. Negation and numbers fool them.
#
# ## 🚀 What's Next?
#
# **Beginner Project — Support Ticket Assistant**: combine prompting (Module 02), token-aware
# design (Module 03), and semantic search (this module) into one working tool.
