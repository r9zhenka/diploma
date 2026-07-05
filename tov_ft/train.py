"""Fine-tune Gemma E4B with Unsloth LoRA for corporate ToV rewriting.

Install (Windows, CUDA):
    irm https://unsloth.ai/install.ps1 | iex
    # or: pip install unsloth
Run:
    python train.py

Trains only on the assistant response (train_on_responses_only) so the model
learns to produce the corporate answer, not to echo the client dialog/draft.
"""
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

from config import MODEL, LORA, TRAIN
from data import build_dataset


def main() -> None:
    # 1. Load base model ----------------------------------------------------
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL.model_name,
        max_seq_length=MODEL.max_seq_length,
        dtype=MODEL.dtype,
        load_in_4bit=MODEL.load_in_4bit,
        full_finetuning=False,
    )

    # 2. Attach LoRA adapters (r=8) ----------------------------------------
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=LORA.finetune_vision_layers,
        finetune_language_layers=LORA.finetune_language_layers,
        finetune_attention_modules=LORA.finetune_attention_modules,
        finetune_mlp_modules=LORA.finetune_mlp_modules,
        r=LORA.r,
        lora_alpha=LORA.lora_alpha,
        lora_dropout=LORA.lora_dropout,
        bias=LORA.bias,
        random_state=LORA.random_state,
    )

    # 3. Chat template + dataset formatting --------------------------------
    tokenizer = get_chat_template(tokenizer, chat_template=MODEL.chat_template)
    dataset = build_dataset()

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. Trainer ------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=TRAIN.per_device_train_batch_size,
            gradient_accumulation_steps=TRAIN.gradient_accumulation_steps,
            warmup_steps=TRAIN.warmup_steps,
            num_train_epochs=TRAIN.num_train_epochs,
            max_steps=TRAIN.max_steps,
            learning_rate=TRAIN.learning_rate,
            logging_steps=TRAIN.logging_steps,
            optim=TRAIN.optim,
            weight_decay=TRAIN.weight_decay,
            lr_scheduler_type=TRAIN.lr_scheduler_type,
            seed=TRAIN.seed,
            output_dir=TRAIN.output_dir,
            report_to=TRAIN.report_to,
        ),
    )

    # 5. Mask everything except the model's answer -------------------------
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    # 6. Train + save the LoRA adapter -------------------------------------
    trainer.train()
    model.save_pretrained(TRAIN.adapter_dir)
    tokenizer.save_pretrained(TRAIN.adapter_dir)
    print(f"\nSaved LoRA adapter to: {TRAIN.adapter_dir}")


if __name__ == "__main__":
    main()
