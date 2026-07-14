# 📘 Prerequisite — Neural Networks Fundamentals

Before exploring Transformers and Large Language Models, it's important to understand the foundation they are built upon: **Neural Networks**.

In this prerequisite lesson, you'll build and train a simple neural network using PyTorch while learning how models make predictions, measure errors, and improve through optimization.

---

## 🎯 Objective

In this project, you'll learn how a neural network works by:

- Building a simple neural network with PyTorch
- Understanding inputs, weights, and biases
- Running a forward pass to make predictions
- Measuring error using a loss function
- Computing gradients with backpropagation
- Updating model parameters using gradient descent
- Observing how the model improves during training

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `neural_network_basics.py` | Demonstrates a simple neural network, forward pass, loss calculation, backpropagation, and the training loop. |
| `neural_network_basics.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation and explanation for the project. |

---

## ⚙️ Setup

Activate your virtual environment and install the required packages.

```powershell
# From repository root
venv\Scripts\activate

cd prerequisites\neural-networks

pip install torch matplotlib
```

---

## ▶️ Run

```powershell
python neural_network_basics.py
```

---

## 🧠 Key Concepts

### 1. What is a Neural Network?

A neural network is a computational model inspired by the human brain. It learns patterns from data by adjusting numerical parameters called **weights** and **biases**.

Modern Large Language Models such as **GPT**, **Llama**, and **Qwen** are all built upon neural network architectures.

---

### 2. Forward Pass

During the forward pass, input data flows through the network to produce a prediction.

A simplified representation is:

```text
Prediction = (Input × Weight) + Bias
```

At this stage, the model is simply making its best guess using its current parameters.

---

### 3. Loss Function

The loss function measures how far the model's prediction is from the correct answer.

- Lower loss → Better predictions
- Higher loss → Larger errors

Training aims to minimize this value over time.

---

### 4. Backpropagation

After computing the loss, the model performs **backpropagation**.

PyTorch automatically calculates gradients for every learnable parameter, indicating:

- Which parameters contributed to the error
- How much each parameter should change
- Which direction reduces the loss

---

### 5. Optimizer

An optimizer updates the model's parameters using the calculated gradients.

This project uses **Stochastic Gradient Descent (SGD)** to gradually improve the model's predictions.

---

### 6. Training Loop

A typical training loop consists of four steps:

```text
Forward Pass
      ↓
Calculate Loss
      ↓
Backward Pass
      ↓
Update Parameters
```

Repeating this cycle allows the model to learn from data.

---

## 📚 Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Neural Network** | A computational model that learns patterns from data by adjusting weights and biases. |
| **Forward Pass** | Produces predictions from the current model parameters. |
| **Loss Function** | Measures prediction error. |
| **Backpropagation** | Computes gradients used to improve the model. |
| **Gradient** | Indicates how each parameter should change to reduce the loss. |
| **Optimizer (SGD)** | Updates model parameters using gradients. |
| **Training Loop** | Repeats prediction, error calculation, and parameter updates until the model learns. |

---

## 💻 Sample Output

```text
=== Neural Networks Fundamentals ===

Simple Neural Network created!

Input data:
tensor([[1., 2.],
        [3., 4.]])

Output (prediction):
tensor([[0.1234],
        [0.5678]])

Training in progress...

Epoch  0 | Loss: 23.4567 | Prediction: 1.2345
Epoch  2 | Loss: 12.3456 | Prediction: 2.3456
...
Training finished!
```

---

## 🎯 Key Takeaways

After completing this lesson, you'll understand:

- How neural networks transform inputs into predictions
- How weights and biases are updated during training
- Why loss functions measure learning progress
- How backpropagation computes gradients automatically
- How optimizers improve model performance over time
- Why neural networks form the foundation of modern AI systems, including Large Language Models

---

## 🚀 What's Next?

**Week 1 • Day 1 — Tokenization & Next-Token Prediction**

Learn how Large Language Models convert text into tokens, transform them into numerical representations, and predict the next token one step at a time using GPT-2.