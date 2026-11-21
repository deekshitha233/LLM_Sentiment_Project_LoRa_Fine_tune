
import json
import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load pretrained model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(
    base_model,
    "./fine_tuned_model"
)

model.eval()

# Load testing dataset
with open("test.json", "r") as f:
    test_data = json.load(f)


def predict_sentiment(review):

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
            "content": review
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    match = re.search(
        r"\b(Positive|Negative|Neutral)\b",
        response,
        re.IGNORECASE
    )

    if match:
        return match.group(1).capitalize()

    return "Invalid"


# Generate predictions
actual = []
predicted = []

print("\nModel Evaluation Results\n")

for item in test_data:

    review = item["input"]
    actual_label = item["output"]

    predicted_label = predict_sentiment(review)

    actual.append(actual_label)
    predicted.append(predicted_label)

    print("Review:", review)
    print("Actual:", actual_label)
    print("Predicted:", predicted_label)
    print("-" * 40)


# Calculate accuracy
accuracy = accuracy_score(actual, predicted)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:")

print(classification_report(
    actual,
    predicted,
    labels=["Positive", "Negative", "Neutral"],
    zero_division=0
))

# Confusion matrix
print("\nConfusion Matrix:")

print(confusion_matrix(
    actual,
    predicted,
    labels=["Positive", "Negative", "Neutral"]
))
