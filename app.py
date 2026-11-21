
import streamlit as st
import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

st.set_page_config(
    page_title="LLM Sentiment Analyzer",
    page_icon="💬"
)

st.title("Customer Feedback Sentiment Analyzer")
st.write("Fine-tuned LLM with streaming generation")

# Load model once
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    model = PeftModel.from_pretrained(
        base_model,
        "./fine_tuned_model"
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


def stream_sentiment(review):

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

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 10,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id
    }

    def generate():
        with torch.no_grad():
            model.generate(**generation_kwargs)

    thread = Thread(target=generate)
    thread.start()

    for token in streamer:
        yield token


# User input
review = st.text_area(
    "Enter customer feedback:",
    placeholder="Example: The product quality is excellent!"
)

if st.button("Analyze Sentiment"):

    if review.strip():

        st.subheader("Streaming Model Output")

        output_placeholder = st.empty()
        full_response = ""

        for token in stream_sentiment(review):
            full_response += token
            output_placeholder.write(full_response)

        st.subheader("Final Sentiment")

        result = full_response.strip().lower()

        if "positive" in result:
            st.success("Positive 😊")

        elif "negative" in result:
            st.error("Negative 😞")

        elif "neutral" in result:
            st.info("Neutral 😐")

        else:
            st.warning("Unable to classify")

    else:
        st.warning("Please enter customer feedback.")
