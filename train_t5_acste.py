#!/usr/bin/env python3
"""
Huấn luyện T5 (seq2seq) cho ACSTE trên MEMD-ABSA.

Dữ liệu (mặc định — layout merged, thư mục dataset):
  - Train.json, Dev.json: object { "Restaurant": [...], "Laptop": [...], ... }
  - PublicTest.json, PrivateTest.json: cùng cấu trúc key, mỗi sample chỉ {"raw_words": "..."}

Ánh xạ:
  - Input:  "acste: " + raw_words
  - Target: các triple "aspect | category | sentiment" nối bằng " ; "

Ví dụ:
  python train_t5_acste.py \\
    --dataset /home/data/TACVU1/ \\
    --dataset_layout merged \\
    --train_all_domains \\
    --domain all \\
    --predict_split PublicTest \\
    --output_dir outputs/t5-baseline

Nếu chỉ predict sử dụng mô hình đã train thì sử dụng flag --predict_only:

  python train_t5_acste.py --predict_only --dataset /home/data/TACVU1/ \\
    --domain Restaurant \\
    --dataset_layout merged --predict_split PrivateTest \\
    --output_dir outputs/t5-baseline --checkpoint_path outputs/t5-baseline/checkpoint-best

Một model / một domain (merged): không dùng --train_all_domains, --domain Restaurant|Laptop|...
  Mỗi run tạo output_dir/PublicTest/<Domain>.json và .../PrivateTest/<Domain>.json (mảng).
  Gộp submission: copy đủ 5 file *.json (tên file = tên domain) vào một thư mục rồi
  python merge_domain_submission.py --input_dir <dir> --output submission_all.json

Thi / máy không mạng: đặt thư mục chứa snapshot HF đầy đủ (config, tokenizer, weight) và truyền
  --pretrained_local_dir /path/to/t5-small  [--local_files_only]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# Khớp competition_dataset / evaluation_script (thứ tự key)
ALL_DOMAINS = ["Restaurant", "Laptop", "Hotel", "Books", "Clothing"]

INPUT_PREFIX = "acste: "
TRIPLE_JOIN = " ; "
FIELD_SEP = " | "
EMPTY_TARGET = "<empty>"


def _env_offline() -> bool:
    return os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")


def _local_pretrained_kw(local_files_only: bool) -> dict:
    return {"local_files_only": True} if local_files_only else {}


def resolve_pretrained_source(args) -> str:
    """Hub id hoặc đường dẫn thư mục local (snapshot HF)."""
    if getattr(args, "pretrained_local_dir", None):
        p = Path(args.pretrained_local_dir).resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"--pretrained_local_dir không phải thư mục: {p}")
        return str(p)
    return args.model_name


def configure_reproducibility(seed: int) -> None:
    """Đặt seed cho random / NumPy / torch / HF; trên CUDA bật cuDNN deterministic."""
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_split_per_domain(dataset_dir: Path, domain: str, split: str) -> list:
    path = dataset_dir / domain / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_merged_split(dataset_root: Path, split: str) -> dict[str, list]:
    if split in ["Train", "Dev"]:
        path = dataset_root / f"train/{split}.json"
    elif split == "PublicTest":
        path = dataset_root / f"public_test/{split}.json"
    elif split == "PrivateTest":
        path = Path("/kaggle/working/private_test") / f"{split}.json"
    else:
        raise ValueError(f"Invalid split: {split}")

    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object with domain keys")
    for d in ALL_DOMAINS:
        if d not in data:
            raise KeyError(f"Merged file {path} missing domain key: {d}")
        if not isinstance(data[d], list):
            raise ValueError(f"{path}[{d}] must be a list")
    return data


def merge_lists_from_merged(blob: dict[str, list], domains: list[str]) -> list:
    out = []
    for d in domains:
        out.extend(blob[d])
    return out


def merge_splits_per_domain(dataset_dir: Path, domains: list[str], split: str) -> list:
    out = []
    for d in domains:
        out.extend(load_split_per_domain(dataset_dir, d, split))
    return out


def quadruple_to_triple_parts(q: dict) -> tuple[str, str, str]:
    terms = q["aspect"].get("term") or []
    if terms == ["NULL"]:
        aspect = "NULL"
    else:
        aspect = " ".join(str(t).strip() for t in terms)
    category = (q.get("category") or "").strip()
    sentiment = (q.get("sentiment") or "").strip().upper()
    return aspect, category, sentiment


def target_text_from_quadruples(quadruples: list) -> str:
    parts = []
    for q in quadruples:
        a, c, s = quadruple_to_triple_parts(q)
        parts.append(f"{a}{FIELD_SEP}{c}{FIELD_SEP}{s}")
    return TRIPLE_JOIN.join(parts) if parts else EMPTY_TARGET


def item_to_input_text(raw_words: str) -> str:
    return f"{INPUT_PREFIX}{raw_words}"


def parse_generated_triples(text: str) -> list[dict]:
    if text is None:
        return []
    t = text.strip()
    if not t or t.lower() in ("<empty>", "none", "n/a"):
        return []
    triples = []
    for chunk in re.split(r"\s*;\s*", t):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(FIELD_SEP, 2)
        if len(parts) != 3:
            continue
        aspect, category, sentiment = (p.strip() for p in parts)
        triples.append(
            {
                "aspect": aspect if aspect.lower() != "null" else "NULL",
                "category": category,
                "sentiment": sentiment.upper() if sentiment else "",
            }
        )
    return triples


def build_records(data: list) -> dict:
    inputs = [item_to_input_text(x["raw_words"]) for x in data]
    targets = [target_text_from_quadruples(x["quadruples"]) for x in data]
    return {"input_text": inputs, "target_text": targets}


def main():
    parser = argparse.ArgumentParser(description="Train T5 for ACSTE (merged or per-domain dataset)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Thư mục chứa dữ liệu (merged: Train.json/Dev.json/... trong root)",
    )
    parser.add_argument(
        "--dataset_layout",
        type=str,
        choices=["merged", "per_domain"],
        default="merged",
        help="merged = một file JSON/split với key domain; per_domain = thư mục con theo domain",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Alias cho --dataset khi dùng layout per_domain (ưu tiên --dataset nếu cả hai được set)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="Restaurant",
        help="Train/eval/predict: một domain (không --train_all_domains) hoặc 'all' (train+predict cả 5, file submission đủ key)",
    )
    parser.add_argument("--train_all_domains", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google-t5/t5-small",
        help="Tên repo Hugging Face (khi không dùng --pretrained_local_dir).",
    )
    parser.add_argument(
        "--pretrained_local_dir",
        type=str,
        default=None,
        help="Thư mục local chứa snapshot đầy đủ (config, tokenizer, pytorch_model.bin/model.safetensors). "
        "Ưu tiên hơn --model_name cho train và fallback tokenizer khi predict_only.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Chỉ đọc file trên disk (không gọi Hub). Nên dùng cùng --pretrained_local_dir khi thi.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/t5-acste")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed tái lập (random, NumPy, torch, Trainer); trên GPU bật cuDNN deterministic",
    )
    parser.add_argument("--predict_only", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=None,
        help="File JSON đầu ra (single split). Với --predict_split both, dùng làm prefix hoặc bỏ qua (mặc định tên trong output_dir)",
    )
    parser.add_argument(
        "--predict_split",
        type=str,
        choices=["PublicTest", "PrivateTest", "Dev", "both"],
        default="PublicTest",
        help="PublicTest / PrivateTest / both (2 file).",
    )
    parser.add_argument("--generation_num_beams", type=int, default=4)
    args = parser.parse_args()

    local_only = args.local_files_only or _env_offline()
    pre_kw = _local_pretrained_kw(local_only)
    pretrained_src = resolve_pretrained_source(args)
    if local_only and not args.pretrained_local_dir and not args.predict_only:
        if not Path(pretrained_src).is_dir():
            raise SystemExit(
                "Đang local_files_only/OFFLINE nhưng --pretrained_local_dir chưa trỏ tới thư mục snapshot "
                "trên máy. Ví dụ: --pretrained_local_dir /data/models/t5-small --local_files_only"
            )

    root = Path(args.dataset_dir) if args.dataset_dir else Path(args.dataset)
    if not root.is_dir():
        raise FileNotFoundError(root)

    configure_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.domain != "all" and args.domain not in ALL_DOMAINS:
        raise ValueError(f"--domain must be one of {ALL_DOMAINS} or 'all'")

    domains_train = ALL_DOMAINS if args.train_all_domains else ([args.domain] if args.domain != "all" else ALL_DOMAINS)
    eval_domain = args.domain

    merged = args.dataset_layout == "merged"

    if merged:
        train_blob = load_merged_split(root, "Train")
        dev_blob = load_merged_split(root, "Dev")
        train_raw = merge_lists_from_merged(train_blob, domains_train)
        if args.train_all_domains or eval_domain == "all":
            eval_raw = merge_lists_from_merged(dev_blob, ALL_DOMAINS)
        else:
            eval_raw = merge_lists_from_merged(dev_blob, [eval_domain])
    else:
        train_raw = merge_splits_per_domain(root, domains_train, "Train")
        if args.train_all_domains or eval_domain == "all":
            eval_raw = merge_splits_per_domain(root, ALL_DOMAINS, "Dev")
        else:
            eval_raw = load_split_per_domain(root, eval_domain, "Dev")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_generation(model_gen, tok_gen, items: list) -> list[dict]:
        model_gen.eval()
        inputs_text = [item_to_input_text(x["raw_words"]) for x in items]
        preds_loc: list[str] = []
        bs = args.per_device_eval_batch_size
        for i in range(0, len(inputs_text), bs):
            batch_in = inputs_text[i : i + bs]
            enc = tok_gen(
                batch_in,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            ).to(device)
            with torch.no_grad():
                gen = model_gen.generate(
                    **enc,
                    max_new_tokens=args.max_target_length,
                    num_beams=args.generation_num_beams,
                    early_stopping=True,
                )
            preds_loc.extend(tok_gen.batch_decode(gen, skip_special_tokens=True))
        return [
            {"raw_words": item["raw_words"], "triples": parse_generated_triples(pred)}
            for item, pred in zip(items, preds_loc)
        ]

    def predict_merged_split(model_gen, tok_gen, split_name: str, out_path: Path) -> None:
        test_blob = load_merged_split(root, split_name)
        out_obj: dict[str, list] = {}
        for d in ALL_DOMAINS:
            out_obj[d] = run_generation(model_gen, tok_gen, test_blob[d])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        n = sum(len(out_obj[d]) for d in ALL_DOMAINS)
        print(f"Đã ghi {n} câu (5 domain) tới {out_path}")

    def predict_merged_one_domain(
        model_gen, tok_gen, split_name: str, domain: str, out_path: Path
    ) -> None:
        """Chỉ predict một domain; ghi mảng JSON (khớp merge_domain_submission: <Domain>.json)."""
        test_blob = load_merged_split(root, split_name)
        items = test_blob[domain]
        sub = run_generation(model_gen, tok_gen, items)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False, indent=2)
        print(f"Đã ghi {len(sub)} câu ({domain}, {split_name}) tới {out_path}")

    def predict_per_domain_test(model_gen, tok_gen, split_name: str, out_path: Path) -> None:
        if eval_domain == "all":
            out_obj: dict[str, list] = {}
            for d in ALL_DOMAINS:
                items = load_split_per_domain(root, d, split_name)
                out_obj[d] = run_generation(model_gen, tok_gen, items)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=2)
            print(f"Đã ghi dự đoán all-domains ({split_name}) tới {out_path}")
        else:
            items = load_split_per_domain(root, eval_domain, split_name)
            sub = run_generation(model_gen, tok_gen, items)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sub, f, ensure_ascii=False, indent=2)
            print(f"Đã ghi {len(sub)} câu tới {out_path}")

    def run_generation_save(model_gen, tok_gen) -> None:
        if merged:
            merged_all_domains = args.domain == "all"
            if merged_all_domains:
                if args.predict_split == "both":
                    p_pub = output_dir / "predictions_PublicTest.json"
                    p_priv = output_dir / "predictions_PrivateTest.json"
                    predict_merged_split(model_gen, tok_gen, "PublicTest", p_pub)
                    predict_merged_split(model_gen, tok_gen, "PrivateTest", p_priv)
                else:
                    split_file = args.predict_split
                    if split_file == "Test":
                        raise ValueError(
                            "Layout merged: không có Test.json — dùng PublicTest, PrivateTest hoặc both"
                        )
                    out = Path(
                        args.predictions_path or (output_dir / f"predictions_{split_file}.json")
                    )
                    predict_merged_split(model_gen, tok_gen, split_file, out)
            else:
                dom = args.domain

                def one_split(split_file: str) -> None:
                    if args.predictions_path and args.predict_split != "both":
                        out = Path(args.predictions_path)
                    else:
                        out = output_dir / split_file / f"{dom}.json"
                    predict_merged_one_domain(model_gen, tok_gen, split_file, dom, out)

                if args.predict_split == "both":
                    one_split("PublicTest")
                    one_split("PrivateTest")
                else:
                    split_file = args.predict_split
                    if split_file == "Test":
                        raise ValueError(
                            "Layout merged: không có Test.json — dùng PublicTest, PrivateTest hoặc both"
                        )
                    if args.predictions_path:
                        out = Path(args.predictions_path)
                    else:
                        out = output_dir / split_file / f"{dom}.json"
                    predict_merged_one_domain(model_gen, tok_gen, split_file, dom, out)
                print(
                    "Gộp 5 domain: copy các file <Domain>.json vào một thư mục rồi "
                    "python merge_domain_submission.py --input_dir <dir> --output submission.json"
                )
        else:
            if args.predict_split == "both":
                predict_per_domain_test(
                    model_gen, tok_gen, "PublicTest", output_dir / "predictions_PublicTest.json"
                )
                predict_per_domain_test(
                    model_gen, tok_gen, "PrivateTest", output_dir / "predictions_PrivateTest.json"
                )
            else:
                split_file = args.predict_split
                if split_file in ("PublicTest", "PrivateTest"):
                    p = root / ALL_DOMAINS[0] / f"{split_file}.json"
                    if not p.exists():
                        raise FileNotFoundError(
                            f"Không tìm thấy {split_file}.json trong layout per_domain: {p}"
                        )
                out = Path(
                    args.predictions_path
                    or (output_dir / f"predictions_{eval_domain}_{split_file}.json")
                )
                predict_per_domain_test(model_gen, tok_gen, split_file, out)

    if args.predict_only:
        ckpt_dir = Path(args.checkpoint_path or str(output_dir / "checkpoint-best"))
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {ckpt_dir}")
        if local_only and not (ckpt_dir / "tokenizer_config.json").exists() and not args.pretrained_local_dir:
            raise SystemExit(
                "OFFLINE/local_files_only: checkpoint không có tokenizer — thêm "
                "--pretrained_local_dir trỏ thư mục snapshot T5 gốc (cùng bộ tokenizer lúc train)."
            )
        if (ckpt_dir / "tokenizer_config.json").exists():
            tokenizer_po = AutoTokenizer.from_pretrained(str(ckpt_dir), **pre_kw)
        else:
            tokenizer_po = AutoTokenizer.from_pretrained(pretrained_src, **pre_kw)
        model_po = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir), **pre_kw)
        model_po.to(device)
        run_generation_save(model_po, tokenizer_po)
        return

    train_ds = Dataset.from_dict(build_records(train_raw))
    eval_ds = Dataset.from_dict(build_records(eval_raw))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_src, **pre_kw)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_src, **pre_kw)

    def preprocess_fn(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_kw = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "logging_steps": 50,
        "predict_with_generate": False,
        "fp16": torch.cuda.is_available(),
        "seed": args.seed,
        "report_to": "none",
    }
    try:
        training_args = Seq2SeqTrainingArguments(eval_strategy="epoch", **train_kw)
    except TypeError:
        training_args = Seq2SeqTrainingArguments(evaluation_strategy="epoch", **train_kw)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    best_dir = output_dir / "checkpoint-best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Đã lưu checkpoint tốt nhất (theo eval_loss) tới {best_dir}")

    trainer.model.to(device)
    run_generation_save(trainer.model, tokenizer)
    print(
        "Đánh giá (ví dụ): python evaluation_script.py --submission <file_predictions> --domain all --dataset <gold_dir>"
    )


if __name__ == "__main__":
    main()
