"""Run the fine-tuned corporate-ToV model on sample drafts.

Run:
    python infer.py                     # uses built-in demo cases
Loads the trained LoRA adapter from TRAIN.adapter_dir. If it is missing,
falls back to the base model so you can still smoke-test the prompt.
"""
import os

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

from config import MODEL, TRAIN, GEN
from prompts import build_messages

# (client dialog, operator's rough free-text draft)
DEMO_CASES = [
    (
        "Клиент: Оплатил заказ №5567, деньги списались, но статус «не оплачен».",
        "оплата зависла, деньги вернутся или зачтутся в течение суток, не волнуйся",
    ),
    (
        "Клиент: Хочу отменить заказ, передумал покупать.",
        "отменим заказ, деньги вернём на карту за 3-5 дней",
    ),
]


def load_model():
    """Load adapter if trained, else the base model."""
    src = TRAIN.adapter_dir if os.path.isdir(TRAIN.adapter_dir) else MODEL.model_name
    if src == MODEL.model_name:
        print("[warn] adapter not found -> loading BASE model (train.py first for ToV).")
    model, tokenizer = FastModel.from_pretrained(
        model_name=src,
        max_seq_length=MODEL.max_seq_length,
        dtype=MODEL.dtype,
        load_in_4bit=MODEL.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=MODEL.chat_template)
    FastModel.for_inference(model)
    return model, tokenizer


def rewrite(model, tokenizer, chat: str, draft: str) -> None:
    messages = build_messages(chat, draft, answer=None)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    print("\n" + "=" * 70)
    print(f"DRAFT: {draft}\n--- corporate answer ---")
    model.generate(
        **inputs,
        max_new_tokens=GEN.max_new_tokens,
        use_cache=True,                 # required: E4B shares KV across layers
        temperature=GEN.temperature,
        top_p=GEN.top_p,
        top_k=GEN.top_k,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


def main() -> None:
    model, tokenizer = load_model()
    for chat, draft in DEMO_CASES:
        rewrite(model, tokenizer, chat, draft)


if __name__ == "__main__":
    main()
