# LLM Explainer Dashboard

> Week 1 Capstone Project

An interactive educational application that visualizes how a Large Language Model (LLM) processes text through tokenization, self-attention, and next-token prediction.

This project combines the concepts covered throughout **Week 1** into a single application, making it easier to understand how Transformer-based language models generate text.

---

## 🚀 Features

- 🔤 Tokenization Breakdown
  - View how input text is split into tokens.

- 🎯 Attention Visualization
  - Explore which previous tokens the model attends to.
  - Includes a simple attention bar chart.

- 🤖 Next-Token Prediction
  - Display the model's top predicted next tokens and their scores.

- 💬 Interactive Exploration
  - Analyze multiple sentences without restarting the application.

- 📊 Educational Visualizations
  - Combine textual explanations with graphical outputs to better understand model behavior.

---

## 📚 Concepts Demonstrated

This project combines the core concepts introduced throughout Week 1.

| Topic | Description |
|--------|-------------|
| Tokenization | Converting text into model-readable tokens |
| Byte Pair Encoding (BPE) | GPT-2's tokenization strategy |
| Self-Attention | Understanding contextual relationships between tokens |
| Next-Token Prediction | The mechanism behind autoregressive text generation |
| Hugging Face Transformers | Loading and running pretrained language models |
| GPT-2 Inference | Generating predictions from a pretrained model |

---

## 📂 Project Structure

```text
llm-explainer-dashboard/
├── README.md
├── llm_explainer.py
└── llm_explainer_dashboard.ipynb
```

| File | Description |
|------|-------------|
| `llm_explainer.py` | Main application combining tokenization, attention visualization, and next-token prediction. |
| `llm_explainer_dashboard.ipynb` | Interactive notebook used for development and experimentation. |
| `README.md` | Project documentation. |

---

## ⚙️ Requirements

From the repository root:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python llm_explainer.py
```

---

## 💡 How It Works

1. Enter any sentence.
2. (Optional) Select a focus word.
3. View the generated tokens.
4. Explore the attention visualization.
5. Inspect the model's next-token predictions.
6. Repeat with different prompts.

---

## 💻 Example

```text
Input

The firewall blocked the traffic because it was suspicious

Focus Word

it
```

The application displays:

- Tokenized input
- Attention scores
- Attention visualization
- Top next-token predictions

---

## 🧠 Concepts Covered

### Tokenization

Every sentence is first converted into tokens before entering the model.

---

### Self-Attention

Attention allows the model to determine which previous tokens are most relevant when processing the current token.

---

### Next-Token Prediction

Text generation is performed by repeatedly predicting the most likely next token.

---

### Hugging Face Transformers

The application uses pretrained GPT-2 models through the Hugging Face Transformers library.

---

### Interactive Learning

Experimenting with different prompts makes abstract LLM concepts much easier to understand.

---

## 🎯 Learning Outcomes

After completing this project you'll understand:

- How LLMs tokenize text
- How attention captures context
- How next-token prediction drives text generation
- How multiple Transformer components work together
- How Hugging Face simplifies working with open-source language models

---

## 📖 Related Lessons

- Day 1 — Tokenization
- Day 2 — Transformer Attention
- Day 3 — PyTorch Fundamentals
- Day 4 — Running Open Language Models

---

## 📈 Future Improvements

Planned enhancements include:

- Support for multiple open-source LLMs
- Interactive generation parameter controls
- Token probability visualization
- Layer-by-layer attention analysis
- Side-by-side model comparison

---

## 🎉 Week 1 Complete

This project marks the completion of **Week 1 — LLM Foundations**.

The concepts learned throughout the week provide the foundation for future topics including:

- Supervised Fine-Tuning (SFT)
- LoRA & QLoRA
- RLHF
- RAG
- AI Agents
- Production AI Systems

---

## 📄 License

This project is part of the **AI Engineering Journey** repository and is released under the MIT License.