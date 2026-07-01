"""Central config for the corporate tone-of-voice fine-tune prototype.

One place to tweak model / LoRA / training knobs. Values follow the Unsloth
Gemma 4 guide, adapted for a small dataset (LoRA r=8).
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # E4B LoRA needs ~17GB VRAM per the Unsloth guide. Swap to E2B for 8-10GB.
    model_name: str = "unsloth/gemma-4-E4B-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True          # 4-bit to cut memory; keeps quality on E4B
    dtype: str | None = None           # None -> auto (bf16 on Ampere+)
    # Non-thinking template for the small models (guide tip #1).
    chat_template: str = "gemma-4"


@dataclass
class LoraConfig:
    r: int = 8                         # small dataset -> keep rank low
    lora_alpha: int = 8               # alpha == r (guide recommendation)
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    finetune_vision_layers: bool = False   # text-only task


@dataclass
class TrainConfig:
    output_dir: str = "outputs"
    adapter_dir: str = "tov_adapter"   # where the trained LoRA is saved
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: int = 3          # small data -> a few epochs beats max_steps
    max_steps: int = -1               # -1 = disabled, use epochs instead
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "none"


@dataclass
class GenConfig:
    # Gemma 4 recommended sampling settings.
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64


MODEL = ModelConfig()
LORA = LoraConfig()
TRAIN = TrainConfig()
GEN = GenConfig()
