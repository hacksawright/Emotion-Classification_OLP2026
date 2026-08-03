## Code Structure
- `main.ipynb`: Chứa luồng chạy baseline hoàn chỉnh
- `train_t5_acste.py`: File mã nguồn chính dùng để training baseline model (T5)
- `merge_domain_submission.py`: Nếu bạn muốn train 1 model cho mỗi domain. Cần gộp các file prediction trên từng domain để đúng format của submission.
- `evaluation_script.py`: File đánh giá kết quả theo công thức của ban tổ chức.
-  `example.json`: File mẫu submission cho tập Dev


## Data Description

1. **raw_words** — the review sentence;
2. **quadruples** — the quadruple list of the given review sentence;
3. **aspect** — contains the start ("from") and end ("to") indexes of the aspect span; the indexes are "-1" and "-1" when the aspect is implicit;
4. **category** — selected from the predefined category set;
5. **opinion** — contains the start ("from") and end ("to") indexes of the opinion span; the indexes are "-1" and "-1" when the opinion is implicit;
6. **sentiment** — sentiment polarity from `{"POS", "NEG", "NEU"}`.

---

## ACSTE Evaluation (Aspect–Category–Sentiment Triple Extraction)

This section describes the submission format, how to **merge** per-domain outputs into the final JSON, and how to run **`evaluation_script.py`**.

### Submission Format

#### Per-domain file (model output)

For each domain, one JSON file whose **filename stem** must be exactly the domain name:

`Restaurant.json`, `Laptop.json`, `Hotel.json`, `Books.json`, `Clothing.json`.

Each file is a **JSON array**. The array length must equal the number of sentences in the test split for that domain, and the order must match the test set.

Each element (one sentence) must have:

| Field        | Type   | Description |
|-------------|--------|-------------|
| `raw_words` | string | The review sentence (must match the test set for alignment). |
| `triples`   | array  | List of predicted triples for this sentence. |

Each **triple** in `triples` must have:

| Field        | Type   | Description |
|-------------|--------|-------------|
| `aspect`    | string | The aspect span as a single string (e.g. `"delivery driver"`). Use `"NULL"` for implicit aspects. |
| `category`  | string | Predefined category, e.g. `"Service#General"`, `"Food#Quality"`. Must match the dataset exactly (including `#` and casing). |
| `sentiment` | string | One of `"POS"`, `"NEG"`, `"NEU"`. |

**Example (one sentence in an array):**

```json
{
  "raw_words": "Very nice people and service !",
  "triples": [
    {"aspect": "service", "category": "Service#General", "sentiment": "POS"},
    {"aspect": "people", "category": "Service#General", "sentiment": "POS"}
  ]
}
```

#### Merged submission (format nộp / chấm `--domain all`)

Script **`merge_domain_submission.py`** gộp đủ 5 file trên thành **một object JSON** duy nhất: mỗi key là tên domain, value là mảng tương ứng.

```json
{
  "Restaurant": [ { "raw_words": "...", "triples": [] }, ... ],
  "Laptop": [ ... ],
  "Hotel": [ ... ],
  "Books": [ ... ],
  "Clothing": [ ... ]
}
```

- Với **`--input_dir`**: thư mục phải chứa đủ `Restaurant.json`, `Laptop.json`, `Hotel.json`, `Books.json`, `Clothing.json`.
- Với **`--inputs`**: liệt kê đường dẫn từng file; **tên file (stem)** phải trùng một trong năm domain ở trên.

**Gộp submission (ví dụ sau khi copy các file domain vào một thư mục split):**

```bash
python merge_domain_submission.py \
  --input_dir outputs/t5-small/PublicTest \
  --output outputs/t5-small/submission.json
```

**Đánh giá toàn bộ domain** (`--domain all`): tham số `--submission` phải trỏ tới **file JSON đã merge** dạng object 5 key như trên (không hỗ trợ chỉ thư mục trong `evaluation_script.py`).

**Đánh giá một domain**: `--submission` có thể là:

- một **mảng** JSON (danh sách câu), hoặc
- một **object** JSON có đúng một key domain (ví dụ file merge đầy đủ nhưng bạn chỉ chạy `--domain Restaurant`).

### Evaluation Metric

- **Micro-F1** with **exact triple match:** a predicted triple is correct only if **aspect**, **category**, and **sentiment** all match a gold triple. Triples are compared as sets per sentence (duplicate triples in one sentence do not double-count).
- Khi `--domain all`, điểm tổng in ra là **trung bình Micro-F1** trên 5 domain.


#### Gold reference: hai chế độ

1. **`--gold_merged PATH`** — một file JSON dạng `{ "Restaurant": [...], ... }`, mỗi mẫu có **`quadruples`** (nhãn vàng). Ví dụ trong repo: `dataset/Dev.json`. File gold cho Public/Private Test do BTC cung cấp cũng dùng cùng schema này.
2. **`--dataset PATH`** — cây MEMD-ABSA: mỗi domain có `Test.json` (mặc định tên thư mục gốc `MEMD_ABSA_Dataset`). Dùng khi không truyền `--gold_merged`.

Chỉ dùng **một** trong hai: nếu có `--gold_merged` thì `--dataset` bị bỏ qua.

#### Evaluate all domains (submission đã merge)

```bash
python evaluation_script.py \
  --submission outputs/t5-small/example.json \
  --domain all \
  --gold_merged /home/data/TACVU1/train/Dev.json \
  --output outputs/dev_public.json
```

Thay `/home/data/TACVU1/train/Dev.json` bằng file nhãn vàng đúng split (ví dụ gold Public Test) khi chấm thật.

#### Evaluate a single domain (file chỉ một mảng, gold từ MEMD `Test.json`)

```bash
python evaluation_script.py \
  --submission Restaurant.json \
  --domain Restaurant \
  --dataset /home/data/TACVU1
```

#### Evaluate a single domain (submission là file merge đủ 5 key)

```bash
python evaluation_script.py \
  --submission submission_merged.json \
  --domain Restaurant \
  --gold_merged /home/data/TACVU1/train/Dev.json
```

#### Optional: shell mẫu

`scripts/evaluation.sh` minh họa lệnh chấm Private Test — **sửa** `--submission`, `--gold_merged` và `--output` cho khớp file thực tế (và sửa lỗi chính tả thư mục nếu có, ví dụ `grouthtruth` → đúng đường dẫn gold).

**Arguments:**

| Argument         | Default             | Description |
|------------------|---------------------|-------------|
| `--submission`   | *(required)*        | File JSON: merged object (5 keys) khi `--domain all`; array hoặc object một domain khi chấm từng domain. |
| `--domain`       | `Restaurant`        | `Restaurant`, `Laptop`, `Hotel`, `Books`, `Clothing`, hoặc `all`. |
| `--dataset`      | `dataset` | Thư mục gốc dataset (có `Restaurant/Test.json`, …). Không dùng nếu có `--gold_merged`. |
| `--gold_merged`  | —                   | File JSON gold đã gộp 5 domain (có `quadruples`). |
| `--output`       | —                   | Ghi kết quả chi tiết JSON (per-domain + overall). |
