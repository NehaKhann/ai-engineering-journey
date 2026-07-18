from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import os

print("=== Week 2 Project: Cybersecurity Assistant (SFT) ===")
print("Training a GPT-2 model on cybersecurity data...\n")

model_name = "gpt2"
output_dir = "./results/cybersecurity-assistant"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("json", data_files={
    "train": "dataset/train.jsonl",
    "validation": "dataset/val.jsonl"
})

print(f"Training samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}\n")

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    push_to_hub=False,
    report_to="none",
    load_best_model_at_end=True,
    fp16=True,                    # Faster training on GPU
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args,
    dataset_text_field="text",      # Matches your jsonl format
    max_seq_length=512,
)

print("Starting Supervised Fine-Tuning...\n")
trainer.train()

print(f"\n✅ Training Completed!")
print(f"Model saved to: {output_dir}")