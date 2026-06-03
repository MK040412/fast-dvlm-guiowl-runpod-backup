#!/usr/bin/env python
"""Upload a completed Fast-dVLM BARD stage/log bundle if HF_TOKEN is available."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from huggingface_hub import HfApi, create_repo, upload_folder


def read_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    token_file = Path(os.environ.get("HF_TOKEN_FILE", "/workspace/.secrets/hf_token"))
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--logs-dir", default="")
    parser.add_argument(
        "--repo",
        default=os.environ.get("HF_REPO", "KMK040412/fast-dvlm-guiowl-bard-bd32"),
    )
    args = parser.parse_args()

    token = read_token()
    folder = Path(args.folder)
    if not token:
        print(f"[hf] skip stage={args.stage}: HF_TOKEN not set", flush=True)
        return 0
    if not (folder / "model.safetensors").exists():
        print(f"[hf] skip stage={args.stage}: missing model.safetensors in {folder}", flush=True)
        return 2

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = args.repo
    if "/" not in repo:
        repo = f"{user}/{repo}"

    create_repo(repo, private=False, exist_ok=True, token=token, repo_type="model")
    upload_folder(
        folder_path=str(folder),
        repo_id=repo,
        repo_type="model",
        token=token,
        path_in_repo=f"checkpoints/{args.stage}",
        commit_message=f"Save Fast-dVLM BARD {args.stage}",
    )
    print(f"[hf] uploaded stage={args.stage} -> https://huggingface.co/{repo}/tree/main/checkpoints/{args.stage}", flush=True)
    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
        if logs_dir.exists():
            upload_folder(
                folder_path=str(logs_dir),
                repo_id=repo,
                repo_type="model",
                token=token,
                path_in_repo=f"paper_logs/{args.stage}",
                commit_message=f"Save Fast-dVLM BARD paper logs {args.stage}",
            )
            print(
                f"[hf] uploaded logs stage={args.stage} -> "
                f"https://huggingface.co/{repo}/tree/main/paper_logs/{args.stage}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
