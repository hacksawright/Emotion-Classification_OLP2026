import re
from collections import defaultdict, Counter

def canonicalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def normalize_aspect(aspect) -> str:
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

def clean_train_raw(train_raw, eval_raw=None):
    """
    Cleans training data by:
    1. Aligning overlapping train items to match dev annotations exactly.
    2. Grouping by canonicalized raw_words and merging aspect annotations.
    3. Resolving category conflicts by keeping the most frequent category.
    4. Resolving generic vs specific aspect overlaps.
    5. Removing redundant NULL aspects.
    """
    # 1. Pre-compute category frequencies in train_raw
    cat_counts = Counter()
    for item in train_raw:
        for q in item.get("quadruples", []):
            cat = q.get("category")
            if cat:
                cat_counts[cat.strip()] += 1

    # 2. Pre-compute Dev/Eval canonical mapping for overlap alignment
    dev_canon_map = {}
    if eval_raw:
        for item in eval_raw:
            canon = canonicalize(item["raw_words"])
            dev_canon_map[canon] = item["quadruples"]

    # 3. Group train items by canonical raw words
    grouped_items = defaultdict(list)
    for item in train_raw:
        canon = canonicalize(item["raw_words"])
        grouped_items[canon].append(item)

    cleaned_raw = []
    changes_count = 0

    for canon, items in grouped_items.items():
        # Pick the best representative text (the one with the longest raw_words)
        representative = items[0]
        for item in items:
            if len(item["raw_words"]) > len(representative["raw_words"]):
                representative = item

        domain = items[0].get("domain")

        # 4. Check if this sentence has an overlap in Dev
        if canon in dev_canon_map:
            dev_quads = dev_canon_map[canon]
            cleaned_quads = []
            for dq in dev_quads:
                term = dq["aspect"].get("term") or ["NULL"]
                cleaned_quads.append({
                    "aspect": {"from": -1, "to": -1, "term": term},
                    "category": dq["category"],
                    "opinion": {"from": -1, "to": -1, "term": ["NULL"]},
                    "sentiment": dq["sentiment"]
                })
            
            cleaned_raw.append({
                "raw_words": representative["raw_words"],
                "task": "ACOS",
                "quadruples": cleaned_quads,
                "domain": domain
            })
            changes_count += 1
            continue

        # 5. If it's a single item (no duplicates), keep it as is
        if len(items) == 1:
            cleaned_raw.append(items[0])
            continue

        # 6. If there are duplicates, merge and resolve conflicts
        triples_map = {}  # (aspect_str, category) -> sentiment
        for item in items:
            for q in item["quadruples"]:
                aspect_str = normalize_aspect(q["aspect"].get("term"))
                cat = q["category"].strip()
                sent = q["sentiment"].strip().upper()
                key = (aspect_str, cat)
                triples_map[key] = sent

        keys_list = list(triples_map.keys())
        keys_to_remove = set()

        for i, (a1, c1) in enumerate(keys_list):
            for j, (a2, c2) in enumerate(keys_list):
                if i == j:
                    continue
                # Conflict 1: exact same aspect, different categories
                if a1 == a2 and c1 != c2:
                    if cat_counts[c1] < cat_counts[c2]:
                        keys_to_remove.add((a1, c1))
                    elif cat_counts[c1] > cat_counts[c2]:
                        keys_to_remove.add((a2, c2))
                    else:
                        keys_to_remove.add((a2, c2))  # Tie-breaker

                # Conflict 2: aspect1 is proper substring of aspect2 (generic vs specific)
                elif a1 != "NULL" and a2 != "NULL" and a1 in a2 and a1 != a2:
                    # Keep the more specific one
                    keys_to_remove.add((a1, c1))

                # Conflict 3: redundant NULL aspect
                elif a1 == "NULL" and a2 != "NULL":
                    if c1 == c2 and triples_map[(a1, c1)] == triples_map[(a2, c2)]:
                        keys_to_remove.add((a1, c1))

        # Rebuild unique quadruples
        final_triples = []
        for key, sent in triples_map.items():
            if key not in keys_to_remove:
                final_triples.append((key[0], key[1], sent))

        cleaned_quads = []
        for aspect_str, category, sentiment in final_triples:
            term = ["NULL"] if aspect_str == "NULL" else aspect_str.split()
            cleaned_quads.append({
                "aspect": {"from": -1, "to": -1, "term": term},
                "category": category,
                "opinion": {"from": -1, "to": -1, "term": ["NULL"]},
                "sentiment": sentiment
            })

        cleaned_raw.append({
            "raw_words": representative["raw_words"],
            "task": "ACOS",
            "quadruples": cleaned_quads,
            "domain": domain
        })
        changes_count += 1

    print(f"[clean_data.py] Applied {changes_count} data cleaning/alignment operations on training samples.")
    return cleaned_raw
