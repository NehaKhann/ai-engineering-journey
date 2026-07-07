# 📘 Day 2 — Understanding Attention in Transformers

Learn how the **Attention mechanism** enables Transformers to understand relationships between words by deciding which parts of a sentence are most relevant when processing each token.

Attention is the key innovation that made modern Large Language Models possible.

---

## 🎯 Objective

In this project, you'll explore how GPT-2 uses self-attention by:

* Running a sentence through a pretrained GPT-2 model
* Selecting a target word
* Visualizing which previous tokens receive the most attention
* Understanding how Query, Key, and Value work together
* Seeing how attention helps models capture context

---

## 📂 Project Files

| File                    | Description                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `attention_analysis.py` | Loads GPT-2, analyzes self-attention for a selected word, and prints the attention weights assigned to previous tokens. |

---

## ⚙️ Setup

Activate your virtual environment and install the required packages.

```powershell
# From repository root
venv\Scripts\activate

cd week01-llm-foundations\day02-attention

pip install transformers torch --break-system-packages
```

---

## ▶️ Run

```powershell
python attention_analysis.py
```

---

## 📦 Model Download

On the first run, Hugging Face downloads the pretrained GPT-2 model (approximately **548 MB**).

This only happens once.

Afterward, the model is loaded from the local cache, so future runs start almost instantly.

---

## 🔧 Experiment

Modify these variables in the script to analyze different words.

```python
text = "The firewall blocked the traffic because it was suspicious"

target_word = "it"      # Try: "firewall", "blocked", "traffic", etc.
```

Try changing both the sentence and the target word to observe how attention patterns change.

---

## 🧠 Key Takeaways

### 1. Attention connects related words

Instead of processing words independently, the model learns which earlier words are most relevant to the current one.

For example, when processing **"it"**, the model may assign higher attention to **"firewall"** because they are contextually related.

---

### 2. Query, Key, and Value form the attention mechanism

Every token is transformed into three vectors:

* **Query** → *What information am I looking for?*
* **Key** → *What information does each token represent?*
* **Value** → *What information should be passed forward?*

The model compares Queries with Keys to determine which Values deserve the most attention.

---

### 3. Multiple attention heads learn different relationships

Transformers don't rely on a single attention pattern.

Instead, they use **multiple attention heads**, allowing different relationships to be learned simultaneously.

For example, one head may focus on grammar while another captures long-range context.

---

### 4. Averaging attention heads can be misleading

When averaging all attention heads together, the first token often receives a disproportionately high attention score.

This is a common positional bias and doesn't necessarily indicate that the first token is the most meaningful.

Examining individual attention heads often provides more insight.

---

### 5. Later layers capture richer context

Earlier Transformer layers generally learn simpler patterns.

As information flows through the network, later layers develop more meaningful contextual relationships.

This is why attention from the final layers often appears more intuitive.

---

## 💻 Sample Output

```text
Attention from 'Ġit' (position 7) to previous tokens:

The            → 0.6787
Ġfirewall      → 0.0700
Ġblocked       → 0.0568
Ġbecause       → 0.0601
Ġit            → 0.0551
```

---

## 🚀 What's Next?

**Week 1 • Day 3 — PyTorch Essentials**

Learn the PyTorch fundamentals behind deep learning models, including:

* Tensors
* `nn.Module`
* Forward pass
* Automatic differentiation (`autograd`)
* Building simple neural networks

These concepts will provide the foundation needed to understand how Transformers are implemented under the hood.
