import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def deep_merge(base, override):
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_api_key(exp, env):
    if exp.get("api_key_env"):
        env_name = str(exp["api_key_env"])
    elif exp.get("api_key_slot") is not None:
        slot = str(exp["api_key_slot"])
        candidates = [f"GEMINI_API_KEY_{slot}", f"GOOGLE_API_KEY_{slot}"]
        env_name = next((name for name in candidates if env.get(name)), candidates[0])
    else:
        env_name = "GEMINI_API_KEY"

    value = env.get(env_name)
    if not value:
        raise RuntimeError(f"Missing API key env var {env_name}.")
    return env_name, value


def desired_api_env_name(exp, env):
    if exp.get("api_key_env"):
        return str(exp["api_key_env"])
    if exp.get("api_key_slot") is not None:
        slot = str(exp["api_key_slot"])
        candidates = [f"GEMINI_API_KEY_{slot}", f"GOOGLE_API_KEY_{slot}"]
        return next((name for name in candidates if env.get(name)), candidates[0])
    return "GEMINI_API_KEY"


def write_experiment_config(base_conf, exp, tmp_dir):
    name = safe_name(exp["name"])
    conf = deep_merge(base_conf, exp.get("overrides", {}))
    conf["experiment_name"] = name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_dir / f"{name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f, sort_keys=False, allow_unicode=True)
    return config_path


def launch_experiment(repo_root, exp, config_path, logs_dir, env):
    name = safe_name(exp["name"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")

    api_env_name, api_key = resolve_api_key(exp, env)
    child_env = dict(env)
    child_env.pop("GOOGLE_API_KEY", None)
    child_env["GEMINI_API_KEY"] = api_key
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["EXPERIMENT_NAME"] = name
    child_env["EXPERIMENT_API_KEY_ENV"] = api_env_name

    cmd = [sys.executable, "-u", "src/main.py", "--config", str(config_path)]
    if exp.get("start_idx") is not None:
        cmd.extend(["--start_idx", str(exp["start_idx"])])
    if exp.get("resume"):
        cmd.extend(["--resume", str(exp["resume"])])

    log_file.write(f"Experiment: {name}\n")
    log_file.write(f"Config: {config_path}\n")
    log_file.write(f"API key env: {api_env_name}\n")
    log_file.write(f"Command: {' '.join(cmd)}\n")
    log_file.write("=" * 80 + "\n")
    log_file.flush()

    process = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=child_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "name": name,
        "process": process,
        "log_file": log_file,
        "log_path": log_path,
        "api_env_name": api_env_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments concurrently with separate API keys.")
    parser.add_argument("--suite", required=True, help="Experiment suite YAML path")
    parser.add_argument("--base-config", default="config.yaml", help="Base config YAML path")
    parser.add_argument("--max-parallel", type=int, default=None, help="Override suite max_parallel")
    parser.add_argument("--dry-run", action="store_true", help="Write configs and print planned runs without launching")
    parser.add_argument("--stop-on-fail", action="store_true", help="Do not launch new jobs after a failure")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if load_dotenv is not None:
        load_dotenv(repo_root / ".env", encoding="utf-8-sig", override=False)

    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = repo_root / suite_path
    suite = load_yaml(suite_path)
    base_conf = load_yaml(repo_root / args.base_config)
    experiments = suite.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in {suite_path}")

    max_parallel = int(args.max_parallel or suite.get("max_parallel", 1))
    tmp_dir = repo_root / suite.get("tmp_config_dir", "experiments/tmp_configs")
    logs_dir = repo_root / suite.get("log_dir", "experiments/logs")

    planned = []
    for exp in experiments:
        if "name" not in exp:
            raise ValueError("Every experiment must have a name")
        config_path = write_experiment_config(base_conf, exp, tmp_dir)
        planned.append((exp, config_path))

    if args.dry_run:
        print(f"Suite: {suite_path}")
        print(f"max_parallel: {max_parallel}")
        for exp, config_path in planned:
            api_label = exp.get("api_key_env") or f"slot {exp.get('api_key_slot', 'default')}"
            api_env_name = desired_api_env_name(exp, os.environ)
            key_status = "OK" if os.environ.get(api_env_name) else "MISSING"
            print(f"- {exp['name']} | api={api_label} ({api_env_name}: {key_status}) | config={config_path}")
        return

    queue = list(planned)
    running = []
    completed = []
    failed = []
    env = os.environ.copy()

    print(f">>> Suite: {suite_path}")
    print(f">>> Experiments: {len(queue)} | max_parallel={max_parallel}")
    print(f">>> Logs: {logs_dir}")

    while queue or running:
        launched_one = True
        while queue and len(running) < max_parallel and launched_one and not (args.stop_on_fail and failed):
            launched_one = False
            busy_api_envs = {run["api_env_name"] for run in running}
            launch_idx = None
            for idx, (candidate_exp, _) in enumerate(queue):
                api_env_name = desired_api_env_name(candidate_exp, env)
                if api_env_name not in busy_api_envs:
                    launch_idx = idx
                    break
            if launch_idx is None:
                break
            exp, config_path = queue.pop(launch_idx)
            run = launch_experiment(repo_root, exp, config_path, logs_dir, env)
            running.append(run)
            launched_one = True
            print(f"[START] {run['name']} | api={run['api_env_name']} | log={run['log_path']}")

        time.sleep(2.0)
        still_running = []
        for run in running:
            code = run["process"].poll()
            if code is None:
                still_running.append(run)
                continue
            run["log_file"].close()
            completed.append(run)
            status = "OK" if code == 0 else f"FAIL({code})"
            if code != 0:
                failed.append(run)
            print(f"[DONE] {run['name']} | {status} | log={run['log_path']}")
        running = still_running

    print("\n=== Summary ===")
    print(f"completed: {len(completed)}")
    print(f"failed   : {len(failed)}")
    if failed:
        for run in failed:
            print(f"- {run['name']} | log={run['log_path']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
