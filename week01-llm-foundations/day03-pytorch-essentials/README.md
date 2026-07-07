# 📘 Day 3 — PyTorch Essentials

Learn the core building blocks of **PyTorch**, the deep learning framework behind most modern AI models, including GPT, Llama, and many other Large Language Models.

Today focuses on understanding how neural networks learn through tensors, automatic differentiation, and gradient descent.

---

## 🎯 Objective

In this project, you'll explore how a simple neural network learns by:

* Working with PyTorch tensors
* Building a model using `nn.Module`
* Running a forward pass
* Computing loss
* Using Autograd to calculate gradients
* Updating model parameters with gradient descent
* Visualizing learning through a loss curve

---

## 📂 Project Files

| File                | Description                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `pytorch_basics.py` | Demonstrates tensors, a simple neural network, automatic differentiation, a training loop, and plots the loss curve over time. |

---

## ⚙️ Setup

Activate your virtual environment and install the required package.

```powershell
# From repository root
venv\Scripts\activate

cd week01-llm-foundations\day03-pytorch-essentials

pip install matplotlib
```

> **Note:** PyTorch is already installed from previous days. Only `matplotlib` is needed to display the training loss graph.

---

## ▶️ Run

```powershell
python pytorch_basics.py
```

After training completes, you'll see:

* Training progress printed for each epoch
* The model's prediction improving over time
* A loss curve showing how the error decreases during training

---

## 🧠 Key Takeaways

### 1. Tensors are the foundation of PyTorch

A **Tensor** is PyTorch's primary data structure.

Everything in a neural network—including inputs, outputs, model weights, and gradients—is represented as tensors.

---

### 2. `nn.Module` defines a neural network

Every PyTorch model inherits from `nn.Module`.

It provides a clean way to organize layers, parameters, and the forward computation of your network.

Nearly every modern deep learning model—including Transformers—is built using this class.

---

### 3. The forward pass makes predictions

During the **forward pass**, the model receives input data and produces a prediction.

At this stage, no learning has happened yet—the model is simply making its best guess using its current parameters.

---

### 4. Loss measures prediction error

A **loss function** compares the model's prediction with the correct answer.

* Lower loss = better prediction
* Higher loss = larger error

Training aims to minimize this loss over time.

---

### 5. Autograd performs automatic differentiation

PyTorch's **Autograd** engine automatically computes gradients during the backward pass.

These gradients indicate:

* Which parameters contributed to the error
* How much each parameter should change
* Which direction reduces the loss

This removes the need to manually calculate derivatives.

---

### 6. Gradient descent is how learning happens

Once gradients are computed, the optimizer updates the model's parameters.

The update follows the basic rule:

```text
parameter = parameter − learning_rate × gradient
```

Repeating this process gradually improves the model's predictions.

---

### 7. One epoch equals one full pass through the data

An **epoch** represents one complete pass over the entire training dataset.

Training typically involves many epochs, allowing the model to continually refine its parameters.

---

### 8. The loss curve reveals learning progress

Plotting the loss after each epoch helps visualize whether the model is learning effectively.

A steadily decreasing curve generally indicates that training is working as expected.

---

## 📚 Key Concepts Summary

| Concept              | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| **Tensor**           | The fundamental data structure used throughout PyTorch.                |
| **Weights & Biases** | Learnable parameters initialized randomly and updated during training. |
| **Forward Pass**     | Produces predictions from the current model.                           |
| **Loss Function**    | Measures how far predictions are from the target values.               |
| **Autograd**         | Automatically computes gradients for every learnable parameter.        |
| **Gradient**         | Indicates how each parameter should change to reduce the loss.         |
| **Learning Rate**    | Controls the size of each parameter update.                            |
| **Gradient Descent** | The optimization algorithm that updates parameters to minimize loss.   |
| **Epoch**            | One complete pass through the training dataset.                        |

---

## 💻 Sample Output

```text
Epoch  0 | Loss: 71.8357 | Prediction: 1.5244
Epoch  5 | Loss:  2.0575 | Prediction: 8.5657
...
Final Prediction: 9.7606 (Target was 10.0)
```

You'll also see a **Loss Curve** graph illustrating how the error decreases as the model learns.

---

## 🚀 What's Next?

**Week 1 • Day 4 — Running Your First Language Model**

Load a pretrained GPT-2 model using the `transformers` library, perform inference on custom prompts, and inspect the generated logits and token predictions to understand how an LLM produces text.
