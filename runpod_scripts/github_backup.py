#!/usr/bin/env python
"""Small GitHub backup for code, scripts, and compact log tails."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path("/workspace")
TOKEN_FILE = Path(os.environ.get("GITHUB_TOKEN_FILE", "/workspace/.secrets/github_token"))
REPO_NAME = os.environ.get("GITHUB_BACKUP_REPO", "fast-dvlm-guiowl-runpod-backup")
WORK = ROOT / "github_backup" / "repo"


def token() -> str:
    return TOKEN_FILE.read_text().strip()


def github(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        if exc.code == 422 and method == "POST":
            return {}
        raise


def run(cmd: list[str], cwd: Path = WORK) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(dirpath: str, names: list[str]) -> set[str]:
        banned = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "venv",
            "hf_cache",
            "triton_cache",
            "android-sdk",
            "android_world",
            "dvlm_ckpts",
            "models",
            "data",
        }
        return {n for n in names if n in banned or n.endswith(".safetensors") or n.endswith(".parquet")}

    shutil.copytree(src, dst, ignore=ignore)


def copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_tail(src: Path, dst: Path, lines: int = 240) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(errors="replace").splitlines()
    dst.write_text("\n".join(text[-lines:]) + "\n")


def prepare_snapshot() -> str:
    user = github("GET", "/user")["login"]
    github("POST", "/user/repos", {"name": REPO_NAME, "private": True, "auto_init": False})
    repo_url = f"https://github.com/{user}/{REPO_NAME}.git"

    WORK.parent.mkdir(parents=True, exist_ok=True)
    if not (WORK / ".git").exists():
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        run(["git", "init"])
        run(["git", "remote", "add", "origin", repo_url])
    else:
        run(["git", "remote", "set-url", "origin", repo_url])

    run(["git", "config", "user.email", "runpod-backup@example.local"])
    run(["git", "config", "user.name", "RunPod Backup"])

    for name in ["fast-dvlm-guiowl"]:
        copytree(ROOT / name, WORK / name)

    eval_dst = WORK / "androidworld_eval"
    eval_dst.mkdir(exist_ok=True)
    for pattern in ["*.py", "*.sh", "env.sh"]:
        for src in (ROOT / "androidworld_eval").glob(pattern):
            copy_file(src, eval_dst / src.name)

    for fname in [
        "bard_stage_run_safe.sh",
        "bard_stage_extend_bd32.sh",
        "run_androidworld_bd8_compare.sh",
        "run_androidworld_dual_parallel.py",
        "parse_androidworld_results.py",
        "hf_upload_bard_stage.py",
        "write_paper_stage_summary.py",
        "github_backup.py",
        "github_backup_loop.sh",
    ]:
        copy_file(ROOT / fname, WORK / "runpod_scripts" / fname)

    metrics = ROOT / "paper_logs" / "stage_metrics.jsonl"
    copy_file(metrics, WORK / "paper_logs" / "stage_metrics.jsonl")
    for summary in (ROOT / "paper_logs").glob("*/summary.json"):
        copy_file(summary, WORK / "paper_logs" / summary.parent.name / "summary.json")

    latest_files = [
        ROOT / "bard_stage_retrain.latest",
        ROOT / "bard_stage_extend_bd32.latest",
        ROOT / "androidworld_bd8_compare.latest",
    ]
    for latest in latest_files:
        if latest.exists():
            try:
                log = Path(latest.read_text().strip())
                write_tail(log, WORK / "log_tails" / f"{log.name}.tail.log")
            except Exception:
                pass
    for log in ROOT.glob("bard_bd*.log"):
        write_tail(log, WORK / "log_tails" / f"{log.name}.tail.log")

    status = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": repo_url,
        "notes": "Code/scripts and compact log tails only. Model checkpoints and full paper logs are uploaded to Hugging Face.",
    }
    (WORK / "RUNPOD_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    (WORK / ".gitignore").write_text(".secrets/\n*.safetensors\n*.parquet\nvenv/\nmodels/\ndata/\ndvlm_ckpts/\n")
    return repo_url


def main() -> int:
    repo_url = prepare_snapshot()
    askpass = ROOT / ".secrets" / "git_askpass.sh"
    askpass.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "*Username*) echo x-access-token ;;\n"
        "*Password*) cat /workspace/.secrets/github_token ;;\n"
        "*) echo x-access-token ;;\n"
        "esac\n"
    )
    askpass.chmod(0o700)
    run(["git", "add", "-A"])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=WORK)
    if diff.returncode == 0:
        print(f"[github] no changes repo={repo_url}", flush=True)
        return 0
    msg = f"RunPod backup {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    run(["git", "commit", "-m", msg])
    env = os.environ.copy()
    env.update({"GIT_ASKPASS": str(askpass), "GIT_TERMINAL_PROMPT": "0"})
    subprocess.run(["git", "push", "-u", "origin", "HEAD:main"], cwd=WORK, env=env, check=True)
    print(f"[github] pushed {repo_url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
