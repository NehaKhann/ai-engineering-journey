# Day 3 — PyTorch Fundamentals

> Week 1 • LLM Foundations

Learn the core building blocks of **PyTorch**, the deep learning framework behind modern AI models such as GPT, Llama, and many other Large Language Models.

This exercise introduces the essential concepts behind neural network training, including tensors, automatic differentiation, gradient descent, and the training loop.

---

## 📚 Learning Objectives

By the end of this exercise, you'll understand:

- How tensors are used in PyTorch
- How to build a neural network using `nn.Module`
- How the forward pass generates predictions
- How loss functions measure prediction error
- How Autograd computes gradients automatically
- How gradient descent updates model parameters
- How to visualize learning using a loss curve

---

## 📂 Project Structure

```text
day-03-pytorch/
├── README.md
├── pytorch_fundamentals.ipynb
└── pytorch_basics.py
```

| File | Description |
|------|-------------|
| `pytorch_fundamentals.ipynb` | Interactive notebook introducing tensors, neural networks, and the PyTorch training workflow. |
| `pytorch_basics.py` | Standalone implementation demonstrating tensors, `nn.Module`, Autograd, gradient descent, and loss visualization. |
| `README.md` | Documentation for this exercise. |

---

## ⚙️ Requirements

Install the required dependencies from the repository root.

```bash
pip install -r requirements.txt
```

or

```bash
pip install matplotlib
```

> **Note:** PyTorch is already installed from previous exercises. Only `matplotlib` is required for plotting the training loss curve.

---

## ▶️ Run

```bash
python pytorch_basics.py
```

After execution you'll see:

- Training progress for each epoch
- Model predictions improving over time
- A loss curve showing how the error decreases during training

You can also explore the concepts interactively using:

```
pytorch_fundamentals.ipynb
```

---

## 🧠 Concepts Covered

### Tensors

A **Tensor** is PyTorch's primary data structure.

Everything in a neural network—including inputs, outputs, model weights, and gradients—is represented as tensors.

---

### `nn.Module`

Every PyTorch model inherits from `nn.Module`.

It provides a structured way to define layers, parameters, and the forward computation of a neural network.

Modern architectures—including Transformers—are built using this class.

---

### Forward Pass

During the **forward pass**, the model receives input data and generates predictions using its current parameters.

At this stage, no learning has occurred yet.

---

### Loss Function

A **loss function** measures how far the model's predictions are from the expected output.

- Lower loss → Better predictions
- Higher loss → Larger prediction errors

The objective of training is to minimize this loss.

---

### Autograd

PyTorch's **Autograd** engine automatically computes gradients during backpropagation.

These gradients determine:

- Which parameters contributed to the error
- How much each parameter should change
- Which direction reduces the loss

This eliminates the need for manually computing derivatives.

---

### Gradient Descent

Once gradients are calculated, the optimizer updates the model parameters.

The update follows the basic rule:

```text
parameter = parameter − learning_rate × gradient
```

Repeating this process gradually improves the model's predictions.

---

### Epochs

An **epoch** represents one complete pass through the training dataset.

Training typically consists of multiple epochs, allowing the model to progressively refine its parameters.

---

### Loss Curve

Plotting the loss after each epoch provides a visual indication of learning progress.

A steadily decreasing curve generally indicates that training is converging successfully.

---

## 📚 Key Concepts Summary

| Concept | Description |
|----------|-------------|
| **Tensor** | Fundamental data structure used throughout PyTorch. |
| **Weights & Biases** | Learnable parameters updated during training. |
| **Forward Pass** | Generates predictions using the current model. |
| **Loss Function** | Measures prediction error. |
| **Autograd** | Automatically computes gradients. |
| **Gradient** | Indicates how parameters should change to reduce loss. |
| **Learning Rate** | Controls the size of parameter updates. |
| **Gradient Descent** | Optimization algorithm used to minimize loss. |
| **Epoch** | One complete pass through the training dataset. |

---

## 💻 Sample Output

```text
Epoch  0 | Loss: 71.8357 | Prediction: 1.5244
Epoch  5 | Loss:  2.0575 | Prediction: 8.5657
...
Final Prediction: 9.7606 (Target was 10.0)
```

A **Loss Curve** will also be displayed, illustrating how the model's error decreases as training progresses.

---

## 🎯 Key Takeaways

- Tensors are the foundation of every PyTorch model.
- Neural networks are organized using `nn.Module`.
- The forward pass generates predictions.
- Loss functions measure prediction error.
- Autograd computes gradients automatically.
- Gradient descent updates parameters to improve performance.
- Loss curves help visualize the learning process.

---

## 📖 Related Resources

- Week 1 Overview → `weeks/week-01-llm-foundations`
- Repository Roadmap → `ROADMAP.md`

---

## ⏭️ Next Lesson

**Day 4 — Running Open Language Models**

Load a pretrained GPT-2 model using the `transformers` library, perform inference on custom prompts, inspect logits, and understand how an LLM generates text.