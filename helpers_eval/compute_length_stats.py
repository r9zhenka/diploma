"""
Средняя длина сгенерированного комментария (в словах) на каждый конфиг
+ метрика combined_distinct_2 * len_mean / 20.5.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluating"

EVAL_PATH = EVAL_DIR / "evaluation_results.json"
QWEN_FILES = [
    EVAL_DIR / "qwen_lora_r8.json",
    EVAL_DIR / "qwen_lora_r16.json",
    EVAL_DIR / "qwen_lora_r32.json",
    # EVAL_DIR / "qwen_lora_ffn.json",
    # EVAL_DIR / "qwen_lora_fan.json",
    # EVAL_DIR / "qwen_ia3.json",
]
OUTPUT_PATH = EVAL_DIR / "len_stats.json"
NORM = 20.5


def mean_count_of_words(texts: list[str]) -> float:
    counts = [len(t.split()) for t in texts if t and t.strip()]
    return sum(counts) / len(counts) if counts else 0.0


def make_entry(gens: list[str], cd2: float) -> dict:
    len_mean = mean_count_of_words(gens)
    return {
        "len_mean": round(len_mean, 4),
        "comd_dist-per-len": round(cd2 * min(1, len_mean / NORM), 4),
    }


def collect() -> dict:
    out: dict[str, dict] = {}

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    for name, cfg in eval_data.items():
        gens = cfg.get("all_generated")
        cd2 = cfg.get("combined_distinct_2")
        if not gens or cd2 is None:
            print(f"Пропуск {name}: нет all_generated или combined_distinct_2")
            continue
        out[name] = make_entry(gens, cd2)

    for path in QWEN_FILES:
        if not Path(path).exists():
            print(f"Пропуск {path}: файл не найден")
            continue
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        stem = os.path.basename(path).rsplit(".", 1)[0]
        epochs = d.get("epochs", {})
        for ep_key in sorted(epochs, key=lambda k: int(k.split("_")[-1])):
            ep = epochs[ep_key]
            gens = ep.get("all_generated")
            cd2 = (ep.get("metrics") or {}).get("combined_distinct_2")
            if not gens or cd2 is None:
                print(
                    f"Пропуск {stem}/{ep_key}: "
                    f"нет all_generated или combined_distinct_2"
                )
                continue
            ep_n = ep_key.split("_")[-1]
            out[f"{stem}_ep{ep_n}"] = make_entry(gens, cd2)

    return out


if __name__ == "__main__":
    stats = collect()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n{len(stats)} конфигов -> {OUTPUT_PATH}\n")

    header = f"{'Config':<32} {'len_mean':>10} {'cd2/len':>10}"
    print(header)
    print("-" * len(header))
    for name in sorted(stats, key=lambda k: -stats[k]["comd_dist-per-len"]):
        s = stats[name]
        print(f"{name:<32} {s['len_mean']:>10.2f} {s['comd_dist-per-len']:>10.4f}")
