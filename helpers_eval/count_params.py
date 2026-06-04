"""Analytic count of trainable parameters per PEFT config (Qwen2.5-0.5B).

LoRA adds A (r x d_in) + B (d_out x r) per targeted Linear:
    params(module) = r * (d_in + d_out)
    trainable = r * sum(d_in + d_out for each targeted layer instance) * n_layers
lora_alpha is only a scale factor and adds no parameters.

IA3 learns one rescaling vector per targeted Linear:
    non-feedforward module -> vector length = d_out (rescales the layer output)
    feedforward module     -> vector length = d_in  (rescales the layer input)
    trainable = sum(vector lengths) * n_layers
"""

# --- Qwen2.5-0.5B architecture ---
N_LAYERS = 24
TOTAL_PARAMS = 494_032_768  # tied embeddings; matches print_trainable_parameters()

# (in_features, out_features) of each attention/MLP Linear
DIMS = {
    "q_proj":    (896, 896),
    "k_proj":    (896, 128),   # GQA: 2 KV heads x 64
    "v_proj":    (896, 128),
    "o_proj":    (896, 896),
    "gate_proj": (896, 4864),
    "up_proj":   (896, 4864),
    "down_proj": (4864, 896),
}

# config name -> spec; spec[0] is the method tag:
#   ("full",)                                 full fine-tuning
#   ("lora", rank, target_modules)            LoRA
#   ("ia3", target_modules, feedforward)      IA3
CONFIGS = {
    "Full fine-tuning":                ("full",),
    "qwen_lora_r8  (q,v)":             ("lora", 8,  ["q_proj", "v_proj"]),
    "qwen_lora_r16 (q,v)":             ("lora", 16, ["q_proj", "v_proj"]),
    "qwen_lora_r32 (q,v)":             ("lora", 32, ["q_proj", "v_proj"]),
    "qwen_lora_fan (q,k,v,o) r8":      ("lora", 8,  ["q_proj", "k_proj", "v_proj", "o_proj"]),
    "qwen_lora_ffn (gate,up,down) r8": ("lora", 8,  ["gate_proj", "up_proj", "down_proj"]),
    "qwen_ia3 (k,v,down)":             ("ia3", ["k_proj", "v_proj", "down_proj"], ["down_proj"]),
}


def lora_params(rank, modules):
    per_layer = sum(rank * (DIMS[m][0] + DIMS[m][1]) for m in modules)
    return per_layer * N_LAYERS


def ia3_params(modules, feedforward_modules):
    # a feedforward module rescales its input (d_in); others rescale the output (d_out)
    per_layer = sum(
        DIMS[m][0] if m in feedforward_modules else DIMS[m][1]
        for m in modules
    )
    return per_layer * N_LAYERS


def trainable_params(spec):
    method = spec[0]
    if method == "full":
        return TOTAL_PARAMS
    if method == "lora":
        return lora_params(spec[1], spec[2])
    if method == "ia3":
        return ia3_params(spec[1], spec[2])
    raise ValueError(f"unknown method: {method}")


print(f"{'config':<34}{'trainable':>14}{'% of total':>12}")
print("-" * 60)
for name, spec in CONFIGS.items():
    trainable = trainable_params(spec)
    pct = 100 * trainable / TOTAL_PARAMS
    print(f"{name:<34}{trainable:>14,}{pct:>11.3f}%")
print("-" * 60)
print(f"{'base model total':<34}{TOTAL_PARAMS:>14,}")
