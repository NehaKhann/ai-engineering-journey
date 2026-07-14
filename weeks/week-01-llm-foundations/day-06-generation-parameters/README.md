# 📘 Day 6 — Generation Parameters in Large Language Models

Learn how generation parameters influence the responses produced by a Large Language Model.

Using GPT-2 and the Hugging Face Transformers library, you'll experiment with different settings to see how the same prompt can produce dramatically different outputs.

---

## 🎯 Objective

In this project, you'll explore how different generation parameters affect text generation by:

- Controlling creativity using `temperature`
- Limiting response length with `max_new_tokens`
- Enabling probabilistic sampling with `do_sample`
- Reducing repetition using `repetition_penalty`
- Preventing repeated phrases with `no_repeat_ngram_size`
- Comparing multiple outputs side by side

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `generation_parameters.py` | Compares GPT-2 outputs using different generation parameters and visualizes the results. |
| `generation_parameters.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation and explanation for the project. |

---

## ⚙️ Setup

Activate your virtual environment and install the required packages.

```powershell
# From repository root
venv\Scripts\activate

cd weeks\week-01-llm-foundations\day-06-generation-parameters

pip install transformers matplotlib
```

---

## ▶️ Run

```powershell
python generation_parameters.py
```

---

## 🧠 Key Concepts

### 1. Temperature

The `temperature` parameter controls how random or predictable the generated text will be.

| Temperature | Behavior |
|-------------|----------|
| **0.3** | Conservative and focused |
| **0.7** | Balanced creativity |
| **1.2** | More creative and diverse |
| **1.5** | Highly creative and sometimes unpredictable |

Lower values make the model choose the highest-probability words more often, while higher values encourage exploration.

---

### 2. `max_new_tokens`

This parameter limits how many new tokens the model is allowed to generate.

Smaller values produce shorter responses, while larger values allow longer and more detailed outputs.

---

### 3. `do_sample`

When enabled (`True`), the model randomly samples from likely next tokens instead of always selecting the most probable one.

This produces more natural and varied responses.

---

### 4. `repetition_penalty`

Language models sometimes repeat words or phrases unnecessarily.

The `repetition_penalty` discourages the model from repeatedly selecting the same tokens, leading to smoother and more readable text.

---

### 5. `no_repeat_ngram_size`

This parameter prevents the model from generating the same sequence of words multiple times.

For example:

❌

```text
The firewall blocked the firewall blocked the firewall...
```

✅

```text
The firewall blocked incoming traffic before allowing new connections.
```

It is commonly used to improve generation quality.

---

### 6. Comparing Different Settings

This project generates multiple responses from the same prompt using different parameter values.

By comparing them side by side, it's easier to understand how generation settings influence:

- Creativity
- Consistency
- Repetition
- Overall writing style

---

## 📚 Key Concepts Summary

| Parameter | Purpose |
|-----------|---------|
| **temperature** | Controls randomness and creativity. |
| **max_new_tokens** | Limits response length. |
| **do_sample** | Enables probabilistic sampling. |
| **repetition_penalty** | Reduces repeated words and phrases. |
| **no_repeat_ngram_size** | Prevents repeated sequences of words. |

---

## 💻 Sample Output

```text
Prompt:
"My firewall keeps blocking"

Low Temperature (0.3)
My firewall keeps blocking incoming traffic from...

Balanced (0.7)
My firewall keeps blocking connections because...

High Temperature (1.2)
My firewall keeps blocking unusual requests while...

Very Creative (1.5)
My firewall keeps blocking mysterious packets as though...
```

The program also displays a **2×2 visualization** comparing each generated response, making it easy to observe how the parameters affect the model's behavior.

---

## 🎯 Key Takeaways

After completing this lesson, you'll understand:

- How generation parameters influence LLM outputs
- Why the same prompt can produce different responses
- How to reduce repetition during text generation
- How creativity can be controlled through temperature
- Why parameter tuning is an important part of building reliable AI applications

---

## 🚀 What's Next?

**Week 2 • Fine-Tuning Fundamentals**

Now that you've learned how to control the behavior of pretrained models, the next step is teaching them new behaviors through **Supervised Fine-Tuning (SFT)** using your own datasets.