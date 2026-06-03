#!/usr/bin/env python
"""Download AITW-General `standard` shards (one at a time). Reads HF token from $HF_TOKEN.

    HF_TOKEN=hf_xxx python scripts/download_aitw.py --split train --start 0 --n 1 --out ./data
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fast_dvlm.data import download_shard

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["train", "test"], default="train")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--out", default="./data")
a = ap.parse_args()
N = 256 if a.split == "train" else 32
for i in range(a.start, min(a.start + a.n, N)):
    shard = f"standard/{a.split}-{i:05d}-of-{N:05d}.parquet"
    print("downloading", shard, "...", flush=True)
    p = download_shard(shard, a.out)
    print("  ->", p, f"({os.path.getsize(p)/1e6:.0f} MB)", flush=True)
