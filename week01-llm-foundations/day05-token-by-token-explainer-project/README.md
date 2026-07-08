# 📘 Day 5 — Mini Project: LLM Explainer Dashboard

Welcome to the **capstone project of Week 1!**

This interactive dashboard brings together everything you've learned throughout the week—from tokenization to attention and next-token prediction—into a single tool that lets you explore how a Large Language Model processes text.

Rather than treating LLMs as "black boxes," this project helps you visualize the individual components that work together to generate language.

---

## 🎯 Objective

In this project, you'll build an interactive dashboard that allows you to:

* Break text into tokens
* Visualize how attention connects words
* Predict the most likely next tokens
* Experiment with different sentences
* Explore how an LLM processes language step by step

---

## ✨ Features

* 🔤 **Tokenization Breakdown**

  * View how your input sentence is split into individual tokens.

* 🎯 **Attention Visualization**

  * Analyze which previous words the model focuses on for a selected token.
  * Includes an easy-to-read bar chart.

* 🤖 **Next-Token Prediction**

  * Display the model's top predicted next tokens along with their scores.

* 💬 **Interactive Mode**

  * Enter multiple sentences without restarting the program.

* 📊 **Readable Visual Output**

  * Combines text output with visualizations to make model behavior easier to understand.

---

## 📂 Project Files

| File               | Description                                                                                                                |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `llm_explainer.py` | Interactive dashboard that combines tokenization, attention analysis, and next-token prediction into a single application. |

---

## ⚙️ Setup

Activate your virtual environment and install the required packages.

```powershell
# From repository root
venv\Scripts\activate

cd week01-llm-foundations\day05-token-by-token-explainer-project

pip install transformers matplotlib
```

---

## ▶️ Run

```powershell
python llm_explainer.py
```

---

## 💡 How to Use

1. Run the program.
2. Enter any sentence you'd like the model to analyze.
3. (Optional) Enter a **focus word** to visualize its attention.
4. Explore the tokenization, attention scores, and next-token predictions.
5. Repeat with new sentences or type **`exit`** to close the application.

---

## 💻 Example

```text
Enter a sentence:
The firewall blocked the traffic because it was suspicious

Focus word:
it
```

The dashboard will display:

* The tokenized sentence
* Attention weights for the selected word
* A visualization of attention scores
* The model's top next-token predictions

---

## 🧠 Key Takeaways

### 1. Multiple Hugging Face components work together

This project combines several pretrained components:

* Tokenizer
* Transformer model with attention
* Causal Language Model (for next-token prediction)

Each performs a different role within the LLM pipeline.

---

### 2. Tokenization is the first step

Before a model understands text, it converts the sentence into tokens.

Everything else—attention, embeddings, and prediction—builds upon these tokens.

---

### 3. Attention helps the model understand context

Attention determines which earlier words are most relevant when processing the current token.

Visualizing these connections makes the model's reasoning process much easier to understand.

---

### 4. Next-token prediction drives text generation

Every generated response is created by repeatedly predicting the most likely next token.

Even long conversations are built one token at a time.

---

### 5. Interactive tools improve understanding

Being able to experiment with different sentences and focus words makes abstract concepts much easier to grasp than static examples.

---

## 📚 Week 1 Concepts Combined

This dashboard brings together everything you've learned so far:

| Day       | Concept                               |
| --------- | ------------------------------------- |
| **Day 1** | Tokenization & Next-Token Prediction  |
| **Day 2** | Self-Attention                        |
| **Day 3** | PyTorch Fundamentals                  |
| **Day 4** | Loading Open Models with Hugging Face |
| **Day 5** | Interactive LLM Exploration Dashboard |

---

## 💡 Key Insight

Large Language Models aren't magic.

They generate text through a sequence of understandable steps:

1. Split text into tokens.
2. Convert tokens into numerical representations.
3. Use self-attention to understand context.
4. Predict the next token.
5. Repeat until the response is complete.

This project brings all of these concepts together in one interactive application, making it easier to see how modern LLMs work under the hood.

---

## 🎉 Week 1 Complete!

Congratulations! 🎉

You've built a strong foundation in the core mechanics behind Large Language Models, including:

* Tokenization
* Embeddings
* Self-attention
* PyTorch fundamentals
* Hugging Face Transformers
* Next-token prediction
* Interactive model exploration

You're now ready to move beyond using pretrained models and begin teaching them new behaviors.

---

## 🚀 What's Next?

**Week 2 — Fine-Tuning Begins**

In the next week, you'll learn how to adapt pretrained language models to specific tasks through fine-tuning, enabling them to generate more accurate, specialized, and task-oriented responses.
