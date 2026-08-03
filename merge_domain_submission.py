#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

DOMAINS = ["Restaurant", "Laptop", "Hotel", "Books", "Clothing"]

def main():
    ap = argparse.ArgumentParser(description="Merge per-domain ACSTE submissions into one JSON object.")
    ap.add_argument("--input_dir", type=str, help="Directory containing <Domain>.json files (e.g. Restaurant.json).")
    ap.add_argument("--inputs", type=str, nargs="*", help="Optional explicit paths (overrides --input_dir).")
    ap.add_argument("--output", type=str, required=True, help="Output merged JSON path.")
    args = ap.parse_args()

    if args.inputs:
        paths = [Path(p) for p in args.inputs]
        data = {}
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)
            name = p.stem
            if name not in DOMAINS:
                raise ValueError(f"File {p} stem must be one of {DOMAINS}")
            data[name] = json.loads(p.read_text(encoding="utf-8"))
    else:
        if not args.input_dir:
            raise ValueError("Provide either --input_dir or --inputs")
        d = Path(args.input_dir)
        if not d.is_dir():
            raise FileNotFoundError(d)
        data = {}
        for dom in DOMAINS:
            p = d / f"{dom}.json"
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}")
            data[dom] = json.loads(p.read_text(encoding="utf-8"))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote merged submission to {out_path}")

if __name__ == "__main__":
    main()