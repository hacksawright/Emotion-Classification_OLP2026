#!/usr/bin/env python3
"""
ACSTE (Aspect-Category-Sentiment Triple Extraction) evaluation script.

Uses Micro-F1 with exact triple match: a predicted triple is correct only if
aspect, category, and sentiment all match the gold triple.

Usage:
  # Evaluate a submission for one domain
  python evaluation_script.py --submission example.json --domain Restaurant --gold_merged dataset/Dev.json

  # Evaluate for all domains (submission must be a JSON object with domain names as keys)
  python evaluation_script.py --submission submission_all.json --domain all --gold_merged dataset/Dev.json
"""

import argparse
import json
from pathlib import Path


DOMAINS = ["Restaurant", "Laptop", "Hotel", "Books", "Clothing"]


def normalize_aspect(aspect) -> str:
    """Normalize aspect to a comparable string."""
    if aspect is None:
        return "NULL"
    if isinstance(aspect, list):
        if not aspect or (len(aspect) == 1 and aspect[0] in ("NULL", "null", "")):
            return "NULL"
        return " ".join(str(t).strip() for t in aspect).strip()
    s = str(aspect).strip()
    if s.upper() == "NULL" or s == "":
        return "NULL"
    return s


def quadruples_to_gold_triples(quadruples):
    """Convert gold quadruples to a list of (aspect_str, category, sentiment) for matching."""
    triples = []
    for q in quadruples:
        aspect = normalize_aspect(q["aspect"].get("term"))
        category = q["category"].strip()
        sentiment = q["sentiment"].strip().upper()
        triples.append((aspect, category, sentiment))
    return triples


def submission_item_to_pred_triples(item):
    """Convert a submission item's triples to (aspect_str, category, sentiment)."""
    triples = []
    for t in item.get("triples", []):
        aspect = normalize_aspect(t.get("aspect"))
        category = (t.get("category") or "").strip()
        sentiment = (t.get("sentiment") or "").strip().upper()
        triples.append((aspect, category, sentiment))
    return triples


def compute_micro_f1(gold_triples_per_sent, pred_triples_per_sent):
    """
    gold_triples_per_sent: list of lists of (aspect, category, sentiment)
    pred_triples_per_sent: list of lists of (aspect, category, sentiment)
    Returns (precision, recall, f1).
    """
    assert len(gold_triples_per_sent) == len(pred_triples_per_sent), (
        f"Length mismatch: gold {len(gold_triples_per_sent)} vs pred {len(pred_triples_per_sent)}"
    )
    correct = 0
    n_pred = 0
    n_gold = 0
    for gold_list, pred_list in zip(gold_triples_per_sent, pred_triples_per_sent):
        gold_set = set(gold_list)
        pred_set = set(pred_list)
        n_gold += len(gold_set)
        n_pred += len(pred_set)
        correct += len(gold_set & pred_set)
    precision = correct / n_pred if n_pred else 0.0
    recall = correct / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_test_set(dataset_dir: Path, domain: str):
    """Load gold test set for a domain."""
    test_path = dataset_dir / domain / "Test.json"
    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found: {test_path}")
    with open(test_path, encoding="utf-8") as f:
        return json.load(f)


def load_merged_gold(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Gold merged file must be a JSON object with domain keys: {path}")
    for d in DOMAINS:
        if d not in obj:
            raise KeyError(f"Gold merged file missing domain key {d!r}: {path}")
        if not isinstance(obj[d], list):
            raise ValueError(f"Gold merged[{d}] must be a list in {path}")
    return obj


def evaluate_domain_lists(gold_items: list, domain: str, submission_data: list) -> dict:
    """Evaluate submission for one domain. gold_items have quadruples; submission_data have triples."""
    if len(submission_data) != len(gold_items):
        raise ValueError(
            f"Submission length mismatch for {domain}: "
            f"submission has {len(submission_data)} items, gold has {len(gold_items)}"
        )
    gold_triples = [quadruples_to_gold_triples(item["quadruples"]) for item in gold_items]
    pred_triples = [submission_item_to_pred_triples(item) for item in submission_data]
    precision, recall, f1 = compute_micro_f1(gold_triples, pred_triples)
    # Use unique triple counts (per-sentence) to match the metric
    n_gold = sum(len(set(g)) for g in gold_triples)
    n_pred = sum(len(set(p)) for p in pred_triples)
    correct = sum(
        len(set(g) & set(p))
        for g, p in zip(gold_triples, pred_triples)
    )
    return {
        "domain": domain,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "n_gold_triples": n_gold,
        "n_pred_triples": n_pred,
        "n_correct": correct,
    }


def evaluate_domain(dataset_dir: Path, domain: str, submission_data: list) -> dict:
    """Gold từ dataset/<Domain>/Test.json (layout per-domain)."""
    test_data = load_test_set(dataset_dir, domain)
    return evaluate_domain_lists(test_data, domain, submission_data)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ACSTE submission(s) against MEMD-ABSA test sets."
    )
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to submission JSON. For --domain all, this must be a JSON object with 5 domain keys.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=DOMAINS + ["all"],
        default="Restaurant",
        help="Domain to evaluate, or 'all' for all domains.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MEMD_ABSA_Dataset",
        help="Path to dataset root (Restaurant/Test.json, ...). Bỏ qua nếu dùng --gold_merged.",
    )
    parser.add_argument(
        "--gold_merged",
        type=str,
        default=None,
        help="JSON gộp {Domain: [samples có quadruples]}, ví dụ competition_dataset/PublicTest_gold.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: write results to this JSON file.",
    )
    args = parser.parse_args()

    gold_merged: dict | None = None
    if args.gold_merged:
        gold_path = Path(args.gold_merged)
        if not gold_path.is_file():
            parser.error(f"--gold_merged not found: {gold_path}")
        gold_merged = load_merged_gold(gold_path)
    else:
        dataset_dir = Path(args.dataset)
        if not dataset_dir.is_dir():
            parser.error(f"dataset not found: {dataset_dir}")

    with open(args.submission, encoding="utf-8") as f:
        submission_obj = json.load(f)

    if args.domain == "all":
        if not isinstance(submission_obj, dict):
            raise ValueError("For --domain all, submission must be a JSON object with 5 domain keys.")
        for d in DOMAINS:
            if d not in submission_obj:
                raise KeyError(f"Submission missing key for domain: {d}")
        domains_to_eval = DOMAINS
    else:
        domains_to_eval = [args.domain]
        if isinstance(submission_obj, dict):
            if args.domain not in submission_obj:
                raise KeyError(f"Submission has no key for domain: {args.domain}")
            submission_obj = submission_obj[args.domain]
        if not isinstance(submission_obj, list):
            raise ValueError("For single-domain evaluation, submission must be a JSON list (or an object containing that domain key).")

    results = []
    for domain in domains_to_eval:
        if args.domain == "all":
            sub_data = submission_obj[domain]
        else:
            sub_data = submission_obj
        if gold_merged is not None:
            r = evaluate_domain_lists(gold_merged[domain], domain, sub_data)
        else:
            r = evaluate_domain(dataset_dir, domain, sub_data)
        results.append(r)

    # Print report
    print("ACSTE Evaluation (Micro-F1, exact triple match)\n" + "=" * 50)
    total_correct = 0
    total_gold = 0
    total_pred = 0
    micro_f1_list = []
    for r in results:
        print(f"\n{r['domain']}:")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")
        print(f"  Micro-F1:  {r['micro_f1']:.4f}")
        print(f"  (Gold: {r['n_gold_triples']}, Pred: {r['n_pred_triples']}, Correct: {r['n_correct']})")
        total_correct += r["n_correct"]
        total_gold += r["n_gold_triples"]
        total_pred += r["n_pred_triples"]
        micro_f1_list.append(r["micro_f1"])

    avg_micro_f1 = sum(micro_f1_list) / len(micro_f1_list) if micro_f1_list else 0.0
    print("\n" + "=" * 50)
    if len(results) > 1:
        print("Final score (average Micro-F1 over 5 domains):")
    else:
        print("Final score (Micro-F1 for the domain):")
    print(f"  Score:     {avg_micro_f1:.4f}")

    if args.output:
        out_data = {
            "per_domain": results,
            "overall": {
                "n_correct": total_correct,
                "n_gold": total_gold,
                "n_pred": total_pred,
                "avg_micro_f1": avg_micro_f1,
            }
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
