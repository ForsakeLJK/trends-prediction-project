import os, glob, json, re
from collections import defaultdict
from datetime import datetime
import pandas as pd

HASHTAG_RE = re.compile(r"#\w+", flags=re.UNICODE)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def extract_hashtags(text: str):
    return [t.lower() for t in HASHTAG_RE.findall(text or "")]

def approx_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))

def timestamp_from_filename(path: str) -> str:
    """bluesky_20260104_151726.jsonl -> 2026-01-04T15:17:26"""
    name = os.path.basename(path)
    m = re.search(r"(\d{8})_(\d{6})", name)
    if not m:
        return ""
    date_part, time_part = m.groups()
    dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
    return dt.isoformat()

def compute_file_features(path: str):
    post_count = defaultdict(int)
    sum_len = defaultdict(int)
    sum_tokens = defaultdict(int)

    for obj in iter_jsonl(path):
        if "text" not in obj:
            continue
        text = obj.get("text") or ""
        tags = extract_hashtags(text)
        if not tags:
            continue

        text_len = len(text)
        tok = approx_token_count(text)

        for t in set(tags):  
            post_count[t] += 1
            sum_len[t] += text_len
            sum_tokens[t] += tok

    file_ts = timestamp_from_filename(path)

    stats = {
        t: {
            "post_count": int(post_count[t]),
            "sum_len": int(sum_len[t]),
            "sum_tokens": int(sum_tokens[t]),
        }
        for t in post_count.keys()
    }
    return file_ts, stats

def collect_files(inp: str):
    if os.path.isdir(inp):
        files = sorted(glob.glob(os.path.join(inp, "*.jsonl")))
    else:
        files = sorted(glob.glob(inp))
        if not files and os.path.isfile(inp):
            files = [inp]
    return files

def build_training_rows(files):
    all_rows = []
    prev_counts = {}

    for fp in files:
        file_ts, stats = compute_file_features(fp)
        current_counts = {}

        for trend, s in stats.items():
            pc = int(s["post_count"])
            avg_len = (s["sum_len"] / pc) if pc else 0.0
            tok_vol = int(s["sum_tokens"])

            if not prev_counts:
                growth = 0
            else:
                growth = pc - prev_counts.get(trend, 0)

            all_rows.append({
                "Trend": trend,
                "time_stamp": file_ts,
                "post_count": pc,
                "avg_post_length": float(avg_len),
                "token_volume": tok_vol,
                "growth_rate": int(growth),
                "source_file": os.path.basename(fp),
            })
            current_counts[trend] = pc

        prev_counts = current_counts

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["time_stamp", "post_count"], ascending=[True, False]).reset_index(drop=True)
    return df

def build_csv_data(input_path = "data", output_csv = "feature_data.csv"):  
    files = collect_files(input_path)
    files = sorted(files, key=lambda p: timestamp_from_filename(p) or os.path.basename(p))
    df_all = build_training_rows(files)
    os.makedirs("train_data", exist_ok=True)
    OUTPUT_ALL = f"train_data/{output_csv}"
    TOP_N = 30
    SORT_BY = "post_count"  
    
    if df_all.empty:
        df_top = df_all
    else:
        df_top = (
            df_all.sort_values(["source_file", SORT_BY], ascending=[True, False])
                .groupby("source_file", as_index=False)
                .head(TOP_N)
                .reset_index(drop=True)
        )
    
    df_top.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")