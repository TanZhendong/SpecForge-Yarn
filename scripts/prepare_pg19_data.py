"""Convert the hcyy/pg19-yarn-6400 parquet to the training jsonl.

The published parquet was saved through a pandas round-trip: the real text
sits in a nested struct column ``row_0.text`` (with a useless companion
column ``original_columns``), so a naive ``load_dataset`` read gets garbage.
This script handles both the nested schema and a plain ``text`` column.

Usage:
    python scripts/prepare_pg19_data.py \
        --src /path/to/pg19_yarn_6400_64k.parquet \
        --dst cache/dataset/train_pg19_64k_6400.json

Download the source first (hf-mirror friendly):
    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download hcyy/pg19-yarn-6400 \
        --repo-type dataset --local-dir ~/datasets/pg19-yarn-6400
"""
import argparse
import json

import pyarrow.parquet as pq


def extract_text(row):
    if "row_0" in row and isinstance(row["row_0"], dict) and "text" in row["row_0"]:
        return row["row_0"]["text"]
    if "text" in row:
        return row["text"]
    raise KeyError(f"cannot find text field in row keys: {list(row.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="pg19_yarn_6400_64k.parquet path")
    parser.add_argument("--dst", required=True, help="output jsonl path")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    pf = pq.ParquetFile(args.src)
    n = 0
    with open(args.dst, "w") as f:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            for row in batch.to_pylist():
                f.write(json.dumps({"text": extract_text(row)}, ensure_ascii=False) + "\n")
                n += 1
    print(f"written {n} rows -> {args.dst}")


if __name__ == "__main__":
    main()
