"""
Significance tests for the thesis results, computed from the two result logs.

1. Each defence against no defence: Cochran-Mantel-Haenszel test, one 2x2 table
   per model and attack, so very different models are not pooled directly.
2. A few single claims: Fisher's exact test.
p-values are two-sided and Holm-corrected within each of the two parts.

Usage:
  python -m experiments.significance
"""
import json
from pathlib import Path

from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent

DEFENCES = {
    "canary": ["canary"],
    "intent": ["intent-anchor"],
    "plan_diff": ["plan-diff"],
    "spotlight": ["spotlight"],
    "trust": ["trust"],
    "all": ["canary", "intent-anchor", "plan-diff", "spotlight", "trust"],
}

# label, group, models, attack (None = all attacks), defence of the first sample.
# The second sample is the same models and attack without a defence,
# unless other models are given as the last element.
CLAIMS = [
    ("GPT vs. Claude, no defence", "paid", ["gpt-4o", "gpt-4o-mini"], None, [],
     ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]),
    ("Qwen2.5 7B+14B vs. 0.5B-3B, no defence", "local",
     ["ollama:qwen2.5:7b", "ollama:qwen2.5:14b"], None, [],
     ["ollama:qwen2.5:0.5b", "ollama:qwen2.5:1.5b", "ollama:qwen2.5:3b"]),
    ("Spotlight, Claude Haiku 4.5, Memory Poisoning", "paid",
     ["claude-haiku-4-5-20251001"], "memory_poison", ["spotlight"], None),
    ("Spotlight, Claude Haiku 4.5, Planner Hijacking", "paid",
     ["claude-haiku-4-5-20251001"], "planner_hijack", ["spotlight"], None),
    ("Spotlight, DeepSeek V4 Pro, Memory Poisoning", "paid",
     ["deepseek-v4-pro"], "memory_poison", ["spotlight"], None),
    ("Spotlight, DeepSeek V4 Pro, Planner Hijacking", "paid",
     ["deepseek-v4-pro"], "planner_hijack", ["spotlight"], None),
    ("Spotlight, DeepSeek Flash, Indirect File Injection", "paid",
     ["deepseek-flash"], "indirect_injection", ["spotlight"], None),
]


def load(path):
    trials = json.loads((ROOT / path).read_text())["attack_trials"]
    return [t for t in trials if not t.get("error")]


def table(trials, models, attack, defence):
    """One row of a 2x2 table: [successes, failures]."""
    s = [t["success"] for t in trials
         if t["model"] in models and t["defense"] == defence
         and attack in (None, t["attack"])]
    return [sum(s), len(s) - sum(s)]


trials = {"paid": load("logs/results_paid_api.json"),
          "local": load("logs/results_free_local.json")}

print("1. Defence vs. no defence (Cochran-Mantel-Haenszel)")
rows, pvalues = [], []
for group, ts in trials.items():
    models = sorted({t["model"] for t in ts})
    attacks = sorted({t["attack"] for t in ts})
    for name, defence in DEFENCES.items():
        strata = [[table(ts, [m], a, defence), table(ts, [m], a, [])]
                  for m in models for a in attacks]
        rows.append(f"{group:6} {name:10}")
        pvalues.append(StratifiedTable(strata).test_null_odds(correction=True).pvalue)
for row, p, p_holm in zip(rows, pvalues, multipletests(pvalues, method="holm")[1]):
    print(f"  {row} p={p:.4f}  Holm p={p_holm:.4f}")

print("\n2. Single claims (Fisher's exact test)")
rows, pvalues = [], []
for label, group, models, attack, defence, other_models in CLAIMS:
    first = table(trials[group], models, attack, defence)
    second = table(trials[group], other_models or models, attack, [])
    rows.append(f"{label:52} {first[0]}/{sum(first)} vs {second[0]}/{sum(second)}")
    pvalues.append(fisher_exact([first, second])[1])
for row, p, p_holm in zip(rows, pvalues, multipletests(pvalues, method="holm")[1]):
    print(f"  {row:70} p={p:.4f}  Holm p={p_holm:.4f}")
