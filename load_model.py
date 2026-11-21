
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Select pretrained model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading pretrained LLM...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("\nModel loaded successfully!")

print("Model name:", MODEL_NAME)
print("Device: CPU")
print("Tokenizer loaded:", tokenizer is not None)
