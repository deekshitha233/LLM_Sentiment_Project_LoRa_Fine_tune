
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Load base model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

# 2. Load fine-tuned LoRA adapter
print("Loading fine-tuned adapter...")

model = PeftModel.from_pretrained(
    base_model,
    "./fine_tuned_model"
)

model.eval()

# 3. Prediction function
def predict_sentiment(review):

    messages = [
        {
            "role": "system",
            "content": (
                "Classify customer feedback as Positive, "
                "Negative, or Neutral. Return only one label."
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

    # Extract only newly generated tokens
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

    return "Invalid prediction: " + response


# 4. Interactive testing
print("\nFine-tuned LLM is ready!")

while True:

    review = input("\nEnter customer feedback (or exit): ")

    if review.lower() == "exit":
        print("Exiting...")
        break

    prediction = predict_sentiment(review)

    print("Predicted Sentiment:", prediction)
