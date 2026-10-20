"""Summarize five-fold LREC scores without loading third-party corpora or models.

Usage: python summarize_results.py [--csv summary.csv]
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).parent
PROMPTS = ["zero_shot", "no_context_no_example"] + [
    f"{context}_{shots}_shot"
    for context in ("no_context", "context") for shots in (1, 2, 4, 8)
]
MODELS = ["gaia-4b", "gemma3-12b", "llama3.1-8b", "mistral-7b", "phi3-14b", "qwen3-14b"]
DOMAINS = ["definitions_bete", "definitions_geocorpus"]


def collect(root=ROOT / "results"):
    rows = []
    for model in MODELS:
        for domain in DOMAINS:
            for prompt in PROMPTS:
                folds = []
                for fold in range(5):
                    path = root / model / domain / "iid" / f"fold_{fold}" / f"{prompt}.json"
                    record = json.loads(path.read_text(encoding="utf-8"))
                    assert set(record) == {"n_items", "scores"}, path
                    bert = record["scores"]["bertscore"]
                    assert len(bert["f1"]) == record["n_items"], path
                    folds.append({
                        "bert_f1": mean(bert["f1"]),
                        "bleu": record["scores"]["bleu"]["bleu"],
                        "meteor": record["scores"]["meteor"]["meteor"],
                    })
                row = {"model": model, "domain": domain, "prompt": prompt}
                for metric in ("bert_f1", "bleu", "meteor"):
                    vals = [item[metric] for item in folds]
                    row[f"{metric}_mean"] = mean(vals)
                    row[f"{metric}_sd"] = stdev(vals)
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Save fold-level summary as CSV")
    args = parser.parse_args()
    rows = collect()
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")
    else:
        print(f"Validated {len(rows)} model/domain/prompt combinations (five folds each).")
        print("Illustrative score (not a claim of exact table formatting):", rows[0])


if __name__ == "__main__":
    main()
