
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

from datasets import Dataset

# 1. Model configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

# 2. Load training data
with open("train.json", "r") as f:
    train_data = json.load(f)

# 3. Prepare instruction-based prompts
def format_data(example):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a sentiment classification assistant. "
                "Classify customer feedback as Positive, Negative, "
                "or Neutral. Return only the sentiment label."
            )
        },
        {
            "role": "user",
            "content": example["input"]
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

formatted_data = [
    format_data(example) for example in train_data
]

# 4. Convert to Hugging Face Dataset
dataset = Dataset.from_dict({
    "text": formatted_data
})

# 5. Tokenization
def tokenize_function(example):

    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

    result["labels"] = [
        token if mask == 1 else -100
        for token, mask in zip(
            result["input_ids"],
            result["attention_mask"]
        )
    ]

    return result

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=False
)

# 6. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# 7. Training configuration
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    use_cpu=not torch.cuda.is_available(),
    fp16=False
)

# 8. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 9. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 10. Start training
print("\nStarting LLM fine-tuning...")

trainer.train()

# 11. Save trained adapter
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("\nFine-tuning completed!")
print("Model saved in fine_tuned_model folder.")
