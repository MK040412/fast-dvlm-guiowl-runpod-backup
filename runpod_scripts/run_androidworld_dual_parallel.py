#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ANDROIDWORLD_ROOT = Path('/workspace/androidworld_eval')
RUN_SH = ANDROIDWORLD_ROOT / 'run_guiowl_androidworld.sh'
START_SERVER_SH = ANDROIDWORLD_ROOT / 'start_guiowl_server.sh'
PARSER = Path('/workspace/parse_androidworld_results.py')
HF_LOG_UPLOADER = Path('/workspace/hf_upload_logs_only.py')
ADB = Path('/workspace/android-sdk/platform-tools/adb')
EMULATOR = Path('/workspace/android-sdk/emulator/emulator')
GENERAL_CORE_TASK_FILE = Path('/workspace/paper_logs/androidworld_general_analysis/general_core_tasks.txt')
TASK_METADATA = Path('/workspace/androidworld_eval/android_world/android_world/task_metadata.json')

POLICY_SPECS = {
    'base_ar': ('/workspace/models/GUI-Owl-1.5-2B-Instruct', 'ar', 0, 8300),
    'bd32_dvlm_strict': ('/workspace/dvlm_ckpts/ckpt_bard_bd32_e1', 'dvlm', 0, 8400),
    'bd32_dvlm_repair': ('/workspace/dvlm_ckpts/ckpt_bard_bd32_e1', 'dvlm', 1, 8500),
}

BENCHMARKS = {
    'general_core_44': {
        'kind': 'task_file',
        'task_file': GENERAL_CORE_TASK_FILE,
        'log_root': Path('/workspace/paper_logs/androidworld_general_core_44_parallel'),
        'stage': 'androidworld_general_core_44_parallel',
        'description': 'Custom AiTW-general-style subset selected from AndroidWorld task metadata.',
    },
    'standard_androidworld_full': {
        'kind': 'standard_full',
        'task_file': None,
        'log_root': Path('/workspace/paper_logs/androidworld_standard_full_parallel'),
        'stage': 'androidworld_standard_full_parallel',
        'description': 'Standard AndroidWorld suite with all registry tasks explicitly sharded.',
    },
}

@dataclass(frozen=True)
class Policy:
    name: str
    model: str
    decode: str
    repair: int
    port: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_base() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault('ANDROID_HOME', '/workspace/android-sdk')
    env.setdefault('ANDROID_SDK_ROOT', '/workspace/android-sdk')
    env.setdefault('HF_HOME', '/workspace/hf_cache')
    env.setdefault('TRANSFORMERS_OFFLINE', '1')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    head = [
        '/workspace/androidworld_eval/venv/bin',
        '/workspace/android-sdk/cmdline-tools/latest/bin',
        '/workspace/android-sdk/platform-tools',
        '/workspace/android-sdk/emulator',
    ]
    env['PATH'] = ':'.join(head + [env.get('PATH', '')])
    return env


def run_capture(cmd: list[str], timeout: int | None = None, env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env or env_base(), timeout=timeout, check=False)
        return proc.returncode, proc.stdout
    except Exception as exc:
        return 999, f'{type(exc).__name__}: {exc}'


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + '\n')


def read_task_file(path: Path) -> list[str]:
    tasks = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        for item in line.split(','):
            item = item.strip()
            if item:
                tasks.append(item)
    return tasks


def read_standard_tasks() -> list[str]:
    data = json.loads(TASK_METADATA.read_text())
    if isinstance(data, dict):
        if all(isinstance(k, str) and isinstance(v, dict) for k, v in data.items()):
            return sorted(data.keys())
        for key in ('tasks', 'task_metadata', 'metadata'):
            val = data.get(key)
            if isinstance(val, dict):
                return sorted(val.keys())
            if isinstance(val, list):
                out = []
                for item in val:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        out.append(str(item.get('task_name') or item.get('name') or item.get('task_template')))
                return sorted([x for x in out if x and x != 'None'])
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(str(item.get('task_name') or item.get('name') or item.get('task_template')))
        return sorted([x for x in out if x and x != 'None'])
    raise RuntimeError(f'Cannot parse standard task metadata from {TASK_METADATA}')


def tasks_for_benchmark(name: str) -> list[str]:
    cfg = BENCHMARKS[name]
    if cfg['kind'] == 'task_file':
        return read_task_file(Path(cfg['task_file']))
    return read_standard_tasks()


def shard_tasks(tasks: list[str], workers: int) -> list[list[str]]:
    workers = max(1, min(workers, len(tasks)))
    shards = [[] for _ in range(workers)]
    for i, task in enumerate(tasks):
        shards[i % workers].append(task)
    return [s for s in shards if s]


def adb_devices() -> tuple[list[str], str]:
    code, out = run_capture([str(ADB), 'devices'], timeout=30)
    devices = []
    if code == 0:
        for line in out.splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == 'device':
                devices.append(parts[0])
    return devices, out


def console_port(serial: str, fallback: int) -> int:
    if serial.startswith('emulator-'):
        try:
            return int(serial.split('-', 1)[1])
        except Exception:
            return fallback
    return fallback


def tail(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ''
    return '\n'.join(path.read_text(errors='replace').splitlines()[-n:])


def stop_proc(proc: subprocess.Popen | None, timeout: int = 20) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=timeout)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    handle = getattr(proc, '_log_handle', None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def upload_logs(log_root: Path, stage: str) -> str | None:
    if not HF_LOG_UPLOADER.exists():
        return None
    py = '/workspace/venv/bin/python' if Path('/workspace/venv/bin/python').exists() else sys.executable
    code, out = run_capture([py, str(HF_LOG_UPLOADER), '--logs-dir', str(log_root), '--stage', stage], timeout=1800)
    (log_root / 'hf_upload.log').write_text(out + f'\nexit_code={code}\n')
    for line in out.splitlines():
        if 'https://huggingface.co/' in line:
            return line.strip()
    return None


def start_server(policy: Policy, log_root: Path, timeout_s: int) -> subprocess.Popen:
    log_path = log_root / policy.name / 'server.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, 'ab', buffering=0)
    env = env_base()
    env.update({
        'GUIOWL_MODEL': policy.model,
        'GUIOWL_DECODE': policy.decode,
        'GUIOWL_REPAIR': str(policy.repair),
        'GUIOWL_SERVER_PORT': str(policy.port),
    })
    proc = subprocess.Popen(['bash', str(START_SERVER_SH)], stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    setattr(proc, '_log_handle', log_handle)
    health = f'http://127.0.0.1:{policy.port}/health'
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'{policy.name} server exited with {proc.returncode}; tail:\n{tail(log_path)}')
        try:
            with urllib.request.urlopen(health, timeout=5) as resp:
                if resp.status == 200:
                    return proc
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(5)
    raise RuntimeError(f'{policy.name} server not healthy after {timeout_s}s; tail:\n{tail(log_path)}')


def start_recorder(serial: str, tag: str, video_dir: Path, log_dir: Path, segment_s: int, bitrate: int) -> subprocess.Popen:
    video_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    script = log_dir / f'record_{tag}.sh'
    script_lines = [
        '#!/bin/bash',
        'set +e',
        f'ADB="{ADB}"',
        f'SERIAL="{serial}"',
        f'VIDEO_DIR="{video_dir}"',
        'mkdir -p "$VIDEO_DIR"',
        'while true; do',
        '  TS=$(date +%Y%m%d_%H%M%S)',
        f'  REMOTE="/sdcard/androidworld_{tag}_${{TS}}.mp4"',
        f'  "$ADB" -s "$SERIAL" shell screenrecord --bit-rate {bitrate} --time-limit {segment_s} "$REMOTE"',
        '  "$ADB" -s "$SERIAL" pull "$REMOTE" "$VIDEO_DIR/" >/dev/null 2>&1',
        '  "$ADB" -s "$SERIAL" shell rm "$REMOTE" >/dev/null 2>&1',
        '  sleep 1',
        'done',
    ]
    script.write_text('\n'.join(script_lines) + '\n')
    script.chmod(0o755)
    log_handle = open(log_dir / f'record_{tag}.log', 'ab', buffering=0)
    proc = subprocess.Popen(['bash', str(script)], stdout=log_handle, stderr=subprocess.STDOUT, env=env_base())
    setattr(proc, '_log_handle', log_handle)
    return proc


def pickle_episodes(root: Path) -> list[Any]:
    episodes = []
    for path in root.rglob('*.pkl.gz'):
        try:
            with gzip.open(path, 'rb') as f:
                obj = pickle.load(f)
            if isinstance(obj, list):
                episodes.extend(obj)
            else:
                episodes.append(obj)
        except Exception:
            pass
    return episodes


def structural_records(obj: Any) -> list[dict[str, Any]]:
    records = []
    def rec(x: Any) -> None:
        if isinstance(x, dict):
            st = x.get('structural')
            if isinstance(st, dict):
                records.append(x)
            for v in x.values():
                rec(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                rec(v)
    rec(obj)
    return records


def summarize_structural(root: Path) -> dict[str, Any]:
    recs = []
    for ep in pickle_episodes(root):
        recs.extend(structural_records(ep))
    def bcount(section: str, key: str) -> int:
        total = 0
        for r in recs:
            val = r.get(section)
            if isinstance(val, dict) and bool(val.get(key)):
                total += 1
        return total
    def nums(section: str, key: str) -> list[float]:
        out = []
        for r in recs:
            val = r.get(section) if section else r
            if isinstance(val, dict):
                x = val.get(key)
                try:
                    if x is not None:
                        out.append(float(x))
                except Exception:
                    pass
        return out
    n = len(recs)
    server_lat = nums('server_prediction', 'latency_ms')
    total_lat = nums('', 'latency_total_ms')
    nfe = nums('server_prediction', 'nfe')
    return {
        'step_count': n,
        'has_tool_call': bcount('structural', 'has_tool_call'),
        'valid_json': bcount('structural', 'valid_json'),
        'valid_mobile_use': bcount('structural', 'valid_mobile_use'),
        'repaired': bcount('structural', 'repaired'),
        'android_action_valid': bcount('action_meta', 'android_action_valid'),
        'valid_json_rate': bcount('structural', 'valid_json') / n if n else None,
        'valid_mobile_use_rate': bcount('structural', 'valid_mobile_use') / n if n else None,
        'repair_rate': bcount('structural', 'repaired') / n if n else None,
        'android_action_valid_rate': bcount('action_meta', 'android_action_valid') / n if n else None,
        'server_latency_ms_mean': mean(server_lat) if server_lat else None,
        'latency_total_ms_mean': mean(total_lat) if total_lat else None,
        'nfe_mean': mean(nfe) if nfe else None,
    }


def parse_results(policy_root: Path, policy_name: str) -> dict[str, Any]:
    out = policy_root / f'{policy_name}_results.json'
    if PARSER.exists():
        py = '/workspace/androidworld_eval/venv/bin/python' if Path('/workspace/androidworld_eval/venv/bin/python').exists() else sys.executable
        run_capture([py, str(PARSER), '--root', str(policy_root), '--out', str(out)], timeout=600)
    result = {}
    if out.exists():
        try:
            result = json.loads(out.read_text())
        except Exception as exc:
            result = {'parse_error': f'{type(exc).__name__}: {exc}'}
    structural = summarize_structural(policy_root)
    write_json(policy_root / f'{policy_name}_structural_summary.json', structural)
    result['structural_summary'] = structural
    return result


def start_emulator(index: int, avd: str, console: int, grpc: int, log_root: Path, allow_software: bool) -> subprocess.Popen:
    if not Path('/dev/kvm').exists() and not allow_software:
        raise RuntimeError('No /dev/kvm; refusing software emulator without --allow-software-emulator')
    log_dir = log_root / 'emulator_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_dir / f'emulator_{index:02d}.log', 'ab', buffering=0)
    cmd = [str(EMULATOR), '-avd', avd, '-no-window', '-no-audio', '-no-boot-anim', '-gpu', 'swiftshader_indirect', '-no-snapshot', '-read-only', '-ports', f'{console},{console + 1}', '-grpc', str(grpc)]
    if not Path('/dev/kvm').exists():
        cmd.append('-no-accel')
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env_base())
    setattr(proc, '_log_handle', log_handle)
    return proc


def wait_boot(serial: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        code, out = run_capture([str(ADB), '-s', serial, 'shell', 'getprop', 'sys.boot_completed'], timeout=20)
        if code == 0 and out.strip() == '1':
            return True
        time.sleep(5)
    return False


def worker_cmd_env(policy: Policy, shard: list[str], worker_id: int, dev: dict[str, Any], policy_root: Path, n_combo: int) -> dict[str, str]:
    worker_root = policy_root / f'worker_{worker_id:02d}'
    worker_root.mkdir(parents=True, exist_ok=True)
    (worker_root / 'tasks.txt').write_text('\n'.join(shard) + '\n')
    env = env_base()
    env.update({
        'TASKS': ','.join(shard),
        'N_TASK_COMBINATIONS': str(n_combo),
        'CONSOLE_PORT': str(dev['console_port']),
        'GRPC_PORT': str(dev['grpc_port']),
        'SERVER_URL': f'http://127.0.0.1:{policy.port}',
        'OUT': str(worker_root / 'runs'),
        'GUIOWL_REPAIR': str(policy.repair),
    })
    return env


def run_policy(policy: Policy, shards: list[list[str]], devices: list[dict[str, Any]], log_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    policy_root = log_root / policy.name
    policy_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    server = None
    workers = []
    try:
        server = start_server(policy, log_root, args.server_startup_timeout_s)
        for i, shard in enumerate(shards):
            dev = devices[i % len(devices)]
            worker_root = policy_root / f'worker_{i:02d}'
            env = worker_cmd_env(policy, shard, i, dev, policy_root, args.n_task_combinations)
            log_handle = open(worker_root / 'androidworld.log', 'ab', buffering=0)
            rec = None
            if args.record_mp4:
                rec = start_recorder(dev['serial'], f'{policy.name}_w{i:02d}', log_root / 'videos' / policy.name / f'worker_{i:02d}', worker_root, args.video_segment_s, args.video_bitrate)
            proc = subprocess.Popen(['bash', str(RUN_SH)], stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            setattr(proc, '_log_handle', log_handle)
            workers.append((i, proc, rec))
        worker_status = []
        for i, proc, rec in workers:
            code = proc.wait()
            stop_proc(rec, timeout=10)
            status = {'worker': i, 'exit_code': code, 'log': str(policy_root / f'worker_{i:02d}' / 'androidworld.log')}
            worker_status.append(status)
        parsed = parse_results(policy_root, policy.name)
        ok = all(w['exit_code'] == 0 for w in worker_status)
        return {
            'status': 'completed' if ok else 'partial',
            'policy': policy.__dict__,
            'seconds': time.time() - started,
            'workers': worker_status,
            'episode_count': parsed.get('episode_count'),
            'scored_episode_count': parsed.get('scored_episode_count'),
            'success_rate': parsed.get('success_rate'),
            'success_count': parsed.get('success_count'),
            'results_json': str(policy_root / f'{policy.name}_results.json'),
            'structural_json': str(policy_root / f'{policy.name}_structural_summary.json'),
            'structural_summary': parsed.get('structural_summary'),
        }
    except Exception as exc:
        return {'status': 'failed', 'policy': policy.__dict__, 'seconds': time.time() - started, 'error': f'{type(exc).__name__}: {exc}', 'server_tail': tail(policy_root / 'server.log')}
    finally:
        for _, proc, rec in workers:
            stop_proc(proc, timeout=10)
            stop_proc(rec, timeout=10)
        stop_proc(server, timeout=30)


def make_summary(status: str, benchmark: str, tasks: list[str], log_root: Path, args: argparse.Namespace, devices_raw: str, devices: list[str], extra: dict[str, Any]) -> dict[str, Any]:
    _, free_h = run_capture(['free', '-h'], timeout=20)
    _, df_h = run_capture(['df', '-h', '/workspace', '/opt', '/'], timeout=20)
    _, nproc = run_capture(['nproc'], timeout=20)
    videos = [str(p) for p in log_root.rglob('*.mp4')] if log_root.exists() else []
    cfg = BENCHMARKS[benchmark]
    return {
        'timestamp': now_iso(),
        'status': status,
        'benchmark': benchmark,
        'description': cfg['description'],
        'log_root': str(log_root),
        'stage': cfg['stage'],
        'task_count': len(tasks),
        'tasks': tasks,
        'policies': args.policies,
        'workers_requested': args.workers,
        'n_task_combinations': args.n_task_combinations,
        'adb_devices': devices,
        'adb_devices_raw': devices_raw,
        'kvm_exists': Path('/dev/kvm').exists(),
        'record_mp4': args.record_mp4,
        'video_count': len(videos),
        'videos': videos[:250],
        'resources': {'nproc': nproc.strip(), 'free_h': free_h, 'df_h': df_h},
        'command': ' '.join(sys.argv),
        **extra,
    }


def run_benchmark(benchmark: str, args: argparse.Namespace) -> int:
    cfg = BENCHMARKS[benchmark]
    log_root = Path(args.log_root) if args.log_root else Path(cfg['log_root'])
    if args.log_root and args.benchmark == 'both':
        log_root = Path(args.log_root) / benchmark
    log_root.mkdir(parents=True, exist_ok=True)
    tasks = tasks_for_benchmark(benchmark)
    policies = []
    for name in [p.strip() for p in args.policies.split(',') if p.strip()]:
        spec = POLICY_SPECS[name]
        policies.append(Policy(name, spec[0], spec[1], spec[2], spec[3]))
    missing = [p.model for p in policies if not Path(p.model).exists()]
    devices, raw = adb_devices()
    if missing:
        summary = make_summary('blocked', benchmark, tasks, log_root, args, raw, devices, {'blocker': 'missing model path', 'missing_models': missing})
        write_json(log_root / 'summary.json', summary)
        if args.upload:
            summary['hf_upload'] = upload_logs(log_root, cfg['stage'])
            write_json(log_root / 'summary.json', summary)
        print(json.dumps(summary, indent=2))
        return 2

    emulators = []
    try:
        if not devices:
            if args.no_launch_emulators or (not Path('/dev/kvm').exists() and not args.allow_software_emulator):
                blocker = 'No ADB device is attached and /dev/kvm is absent. Attach AndroidWorld-compatible emulator(s) with gRPC, or use a KVM-enabled pod. Software emulator is intentionally disabled unless --allow-software-emulator is passed for slow smoke tests.'
                summary = make_summary('blocked', benchmark, tasks, log_root, args, raw, devices, {'blocker': blocker, 'policy_specs': [p.__dict__ for p in policies]})
                write_json(log_root / 'summary.json', summary)
                if args.upload:
                    summary['hf_upload'] = upload_logs(log_root, cfg['stage'])
                    write_json(log_root / 'summary.json', summary)
                print(json.dumps(summary, indent=2))
                return 2
            launch_n = max(1, min(args.workers, len(tasks)))
            for i in range(launch_n):
                con = args.base_console_port + 2 * i
                grpc = args.base_grpc_port + i
                emulators.append(start_emulator(i, args.avd, con, grpc, log_root, args.allow_software_emulator))
            expected = [f'emulator-{args.base_console_port + 2 * i}' for i in range(launch_n)]
            booted = [s for s in expected if wait_boot(s, args.boot_timeout_s)]
            devices = booted
            raw = adb_devices()[1]
            if not devices:
                summary = make_summary('blocked', benchmark, tasks, log_root, args, raw, devices, {'blocker': 'No launched emulator reached sys.boot_completed=1.', 'policy_specs': [p.__dict__ for p in policies]})
                write_json(log_root / 'summary.json', summary)
                if args.upload:
                    summary['hf_upload'] = upload_logs(log_root, cfg['stage'])
                    write_json(log_root / 'summary.json', summary)
                print(json.dumps(summary, indent=2))
                return 2

        worker_n = max(1, min(args.workers, len(tasks), len(devices)))
        shards = shard_tasks(tasks, worker_n)
        devmap = []
        for i in range(worker_n):
            serial = devices[i % len(devices)]
            devmap.append({'serial': serial, 'console_port': console_port(serial, args.base_console_port + 2 * i), 'grpc_port': args.base_grpc_port + i})
        write_json(log_root / 'task_shards.json', {'benchmark': benchmark, 'shards': shards})
        write_json(log_root / 'device_map.json', devmap)
        results = {}
        for policy in policies:
            results[policy.name] = run_policy(policy, shards, devmap, log_root, args)
            partial = make_summary('running', benchmark, tasks, log_root, args, raw, devices, {'device_map': devmap, 'results': results})
            write_json(log_root / 'summary_partial.json', partial)
        status = 'completed' if all(r.get('status') == 'completed' for r in results.values()) else 'partial'
        summary = make_summary(status, benchmark, tasks, log_root, args, raw, devices, {'device_map': devmap, 'results': results, 'note': 'Task success is AndroidWorld is_successful. Structural validity/repair metrics come from GUI-Owl step metadata. Videos are adb screenrecord mp4 chunks.'})
        write_json(log_root / 'summary.json', summary)
        if args.upload:
            summary['hf_upload'] = upload_logs(log_root, cfg['stage'])
            write_json(log_root / 'summary.json', summary)
        print(json.dumps(summary, indent=2))
        return 0 if status == 'completed' else 1
    finally:
        for proc in emulators:
            stop_proc(proc, timeout=20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', choices=['general_core_44', 'standard_androidworld_full', 'both'], default='both')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n-task-combinations', type=int, default=1)
    ap.add_argument('--policies', default='base_ar,bd32_dvlm_strict,bd32_dvlm_repair')
    ap.add_argument('--log-root', default='')
    ap.add_argument('--base-console-port', type=int, default=5554)
    ap.add_argument('--base-grpc-port', type=int, default=8554)
    ap.add_argument('--avd', default='Pixel_6_API_33')
    ap.add_argument('--no-launch-emulators', action='store_true')
    ap.add_argument('--allow-software-emulator', action='store_true')
    ap.add_argument('--boot-timeout-s', type=int, default=420)
    ap.add_argument('--server-startup-timeout-s', type=int, default=900)
    ap.add_argument('--record-mp4', action='store_true', default=True)
    ap.add_argument('--no-record-mp4', dest='record_mp4', action='store_false')
    ap.add_argument('--video-segment-s', type=int, default=180)
    ap.add_argument('--video-bitrate', type=int, default=4000000)
    ap.add_argument('--upload', action='store_true', default=True)
    ap.add_argument('--no-upload', dest='upload', action='store_false')
    args = ap.parse_args()
    benches = ['general_core_44', 'standard_androidworld_full'] if args.benchmark == 'both' else [args.benchmark]
    codes = []
    for bench in benches:
        codes.append(run_benchmark(bench, args))
    return max(codes) if codes else 0

if __name__ == '__main__':
    raise SystemExit(main())
