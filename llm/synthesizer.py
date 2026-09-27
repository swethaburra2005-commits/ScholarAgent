"""Direct, evidence-aware local LLM synthesis for ScholarAgent."""

import re
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_INPUT_CHARS_PER_EVIDENCE = 1400
MAX_NEW_TOKENS = 180


@st.cache_resource(show_spinner="Loading local research synthesis model...")
def load_synthesis_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def build_evidence_prompt(question, evidence):
    blocks = []

    for index, item in enumerate(evidence, start=1):
        text = str(item.get("text", ""))[:MAX_INPUT_CHARS_PER_EVIDENCE]

        blocks.append(
            f"[Evidence {index}]\n"
            f"Paper: {item.get('title', 'Unknown paper')}\n"
            f"Chunk: {item.get('chunk_id', 'Unknown')}\n"
            f"Content: {text}"
        )

    evidence_text = "\n\n".join(blocks)

    return f"""You are ScholarAgent, a careful academic research assistant.

Answer the research question directly using ONLY the retrieved evidence.

Start with the answer immediately.
Write one clear, concise research-style response of about 120-180 words.
Combine related findings instead of listing them mechanically.
For each important claim, add the supporting citation in the form [Evidence 1], [Evidence 2], etc.
Do not invent facts, numbers, papers, citations, or conclusions that are not supported by the evidence.
If the evidence is insufficient for part of the question, explicitly say so.
Do not write headings such as "Direct answer", "Key findings", or "Limitations".
Do not repeat the question.

Research question:
{question}

Retrieved evidence:

{evidence_text}

Now write only the final research synthesis."""


def synthesize_answer(question, evidence):
    tokenizer, model, device = load_synthesis_model()

    prompt = build_evidence_prompt(question, evidence)

    messages = [
        {
            "role": "system",
            "content": (
                "You produce concise, evidence-grounded academic answers "
                "and never invent unsupported claims."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    if getattr(tokenizer, "chat_template", None):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = (
            "System:\n"
            + messages[0]["content"]
            + "\n\nUser:\n"
            + messages[1]["content"]
        )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[-1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    if not answer:
        return "The local model did not return a synthesis."

    # Qwen 0.5B may occasionally ignore the requested citation format.
    # In that case, add a transparent source-level evidence basis rather
    # than inventing sentence-level citations.
    if not re.search(
        r"\[Evidence\s+[1-9]\d*\]",
        answer,
        re.IGNORECASE,
    ):
        markers = " ".join(
            f"[Evidence {index}]"
            for index in range(1, len(evidence) + 1)
        )

        answer = (
            answer.rstrip()
            + "\n\n"
            + "Evidence basis: "
            + markers
        )

    return answer
