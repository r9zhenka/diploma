"""Build one big metrics table for every config / epoch.

Sources:
  evaluating/evaluation_results.json   baselines, full FT, old LoRA
  evaluating/qwen_*_metrics.json       per-epoch metrics for the 6 PEFT methods
  evaluating/judge_stats_1.json        LLM-judge accuracy per config
  evaluating/len_stats.json            mean length + combined-distinct-per-len
  count_params.py logic                trainable parameters per config

Outputs: metrics_table.csv and metrics_table.md.
"""
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluating"

QWEN_TOTAL = 494_032_768

LORA_PARAMS = {
    "qwen_lora_r8": 540_672, "qwen_lora_r16": 1_081_344, "qwen_lora_r32": 2_162_688,
    "qwen_lora_fan": 1_081_344, "qwen_lora_ffn": 3_317_760, "qwen_ia3": 24_576,
}
GROUP_B_PARAMS = {
    "rugpt3_base": 0, "qwen_base": 0,
    "rugpt3_ft_ep1": None, "rugpt3_ft_ep3": None,
    "qwen_ft_ep2": QWEN_TOTAL, "qwen_ft_ep3": QWEN_TOTAL,
    "lora_r8_ep3": 540_672,  "lora_r8_ep8": 540_672,
    "lora_r16_ep2": 1_081_344, "lora_r16_ep8": 1_081_344,
    "lora_r64_ep2": 4_325_376, "lora_r64_ep8": 4_325_376,
}
METHOD_LABEL = {
    "qwen_lora_r8": "LoRA r8 (q,v)", "qwen_lora_r16": "LoRA r16 (q,v)",
    "qwen_lora_r32": "LoRA r32 (q,v)", "qwen_lora_fan": "LoRA Attn (q,k,v,o)",
    "qwen_lora_ffn": "LoRA FFN (gate,up,down)", "qwen_ia3": "IA3 (q,v)",
}

evalres = json.load(open(EVAL_DIR / "evaluation_results.json", encoding="utf-8"))
judge   = json.load(open(EVAL_DIR / "judge_stats_1.json", encoding="utf-8"))["accuracy_per_config"]
lens    = json.load(open(EVAL_DIR / "len_stats.json", encoding="utf-8"))

COLUMNS = [
    "group", "config", "method", "epoch",
    "trainable_params", "pct_of_qwen",
    "train_loss", "eval_loss",
    "distinct_1", "distinct_2", "distinct_2_tok", "self_distinct_2", "combined_distinct_2",
    "bertscore_post_P", "bertscore_post_F1", "bertscore_ref_F1",
    "ppl_comments", "ppl_wiki",
    "len_mean", "comb_dist_per_len",
    "judge_acc", "judge_correct", "judge_total",
]


def pct(p):
    return round(100 * p / QWEN_TOTAL, 4) if p else ("" if p is None else 0.0)


def row(group, config, method, epoch, params, src):
    j = judge.get(config, {})
    ls = lens.get(config, {})
    return {
        "group": group, "config": config, "method": method, "epoch": epoch,
        "trainable_params": "" if params is None else params,
        "pct_of_qwen": pct(params),
        "train_loss": src.get("train_loss", ""), "eval_loss": src.get("eval_loss", ""),
        "distinct_1": src.get("distinct_1", ""), "distinct_2": src.get("distinct_2", ""),
        "distinct_2_tok": src.get("distinct_2_tok", ""),
        "self_distinct_2": src.get("self_distinct_2", ""),
        "combined_distinct_2": src.get("combined_distinct_2", ""),
        "bertscore_post_P": src.get("bertscore_post_P", ""),
        "bertscore_post_F1": src.get("bertscore_post_F1", ""),
        "bertscore_ref_F1": src.get("bertscore_ref_F1", ""),
        "ppl_comments": src.get("ppl_comments", ""), "ppl_wiki": src.get("ppl_wiki", ""),
        "len_mean": ls.get("len_mean", ""),
        "comb_dist_per_len": ls.get("comd_dist-per-len", ""),
        "judge_acc": j.get("accuracy", ""), "judge_correct": j.get("correct", ""),
        "judge_total": j.get("total", ""),
    }


rows = []

for cfg in evalres:
    if cfg in ("rugpt3_base", "qwen_base"):
        grp = "baseline"
    elif "_ft_" in cfg:
        grp = "full FT"
    else:
        grp = "LoRA (old)"
    rows.append(row(grp, cfg, cfg, "", GROUP_B_PARAMS.get(cfg, ""), evalres[cfg]))

for base in ["qwen_lora_r8", "qwen_lora_r16", "qwen_lora_r32",
             "qwen_lora_fan", "qwen_lora_ffn", "qwen_ia3"]:
    m = json.load(open(EVAL_DIR / (base + "_metrics.json"), encoding="utf-8"))["epochs"]
    for n in range(1, 9):
        ep = m.get("epoch_%d" % n, {})
        rows.append(row("PEFT", "%s_ep%d" % (base, n), METHOD_LABEL[base],
                        n, LORA_PARAMS[base], ep))

with open(ROOT / "metrics_table.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    w.writeheader()
    w.writerows(rows)

HEAD = ["group", "config", "method", "ep", "train.par", "%qwen",
        "tr_loss", "ev_loss", "dist-1", "dist-2", "dist-2tok", "self-d2",
        "comb-d2", "BS_postP", "BS_postF1", "BS_refF1", "ppl_com", "ppl_wiki",
        "len", "cd/len", "judge_acc", "j_corr", "j_tot"]
RND = {"train_loss": 3, "eval_loss": 3, "distinct_1": 3, "distinct_2": 3,
       "distinct_2_tok": 3, "self_distinct_2": 3, "combined_distinct_2": 3,
       "bertscore_post_P": 3, "bertscore_post_F1": 3, "bertscore_ref_F1": 3,
       "ppl_comments": 1, "ppl_wiki": 1, "len_mean": 1, "comb_dist_per_len": 3,
       "judge_acc": 3, "pct_of_qwen": 3}


def fmt(col, v):
    if v == "" or v is None:
        return "--"
    if col == "trainable_params":
        return "{:,}".format(v)
    if col in RND:
        return str(round(v, RND[col]))
    return str(v)


with open(ROOT / "metrics_table.md", "w", encoding="utf-8") as f:
    f.write("# Big metrics table -- все конфигурации x эпохи\n\n")
    f.write("Источники: evaluation_results.json, qwen_*_metrics.json, "
            "judge_stats_1.json, len_stats.json. judge_acc ниже = лучше "
            "(модель труднее отличить).\n\n")
    f.write("| " + " | ".join(HEAD) + " |\n")
    f.write("|" + "|".join(["---"] * len(HEAD)) + "|\n")
    for r in rows:
        cells = [fmt(c, r[c]) for c in COLUMNS]
        f.write("| " + " | ".join(cells) + " |\n")

print("Wrote metrics_table.csv and metrics_table.md  (%d rows)" % len(rows))
