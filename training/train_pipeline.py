#!/usr/bin/env python3
"""One-click multi-stage trainer for DBP-JSCC."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = ROOT_DIR / "training" / "train.py"
CKPT_RE = re.compile(r"checkpoint_step_(\d+)_epoch_(\d+)\.pth$")


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a JSON object: {path}")
    return data


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
    return safe.strip("_") or "stage"


def _merge_dicts(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(base)
    if not override:
        return merged
    for key, value in override.items():
        merged[key] = value
    return merged


def _merge_env(base: Dict[str, str], override: Optional[Dict[str, Any]]) -> Dict[str, str]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[str(key)] = str(value)
    return merged


def _args_to_cli(args_map: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    for key, value in args_map.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            cli.extend([flag, ",".join(str(v) for v in value)])
            continue
        cli.extend([flag, str(value)])
    return cli


def _latest_checkpoint(stage_dir: Path) -> Optional[Path]:
    ckpt_dir = stage_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None

    best: Optional[tuple[int, int, Path]] = None
    for path in ckpt_dir.glob("*.pth"):
        match = CKPT_RE.match(path.name)
        if not match:
            continue
        step = int(match.group(1))
        epoch = int(match.group(2))
        score = (step, epoch, path)
        if best is None or score[:2] > best[:2]:
            best = score
    return None if best is None else best[2]


def _resolve_resume(
    stage_index: int,
    stage_cfg: Dict[str, Any],
    stage_dirs: List[Path],
) -> Optional[Path]:
    if "resume" in stage_cfg:
        resume_val = stage_cfg.get("resume")
        if resume_val in (None, "", False):
            return None
        return _resolve_path(str(resume_val))

    if stage_index == 0:
        return None

    prev_ckpt = _latest_checkpoint(stage_dirs[stage_index - 1])
    if prev_ckpt is None:
        raise FileNotFoundError(
            f"Stage {stage_index} requires previous checkpoint, but none was found in "
            f"{stage_dirs[stage_index - 1] / 'checkpoints'}"
        )
    return prev_ckpt


def _build_stage_plan(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    stages = cfg.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Config must contain a non-empty 'stages' list")

    run_root = _resolve_path(str(cfg.get("run_root", "./runs/multistage_training")))
    run_root.mkdir(parents=True, exist_ok=True)

    base_args = cfg.get("base_args", {})
    if not isinstance(base_args, dict):
        raise TypeError("'base_args' must be an object")

    base_env_raw = cfg.get("env", {})
    if not isinstance(base_env_raw, dict):
        raise TypeError("'env' must be an object")
    base_env = {str(k): str(v) for k, v in base_env_raw.items()}

    append_stage_name = bool(cfg.get("append_stage_to_wandb_run_name", True))

    plans: List[Dict[str, Any]] = []
    stage_dirs: List[Path] = []

    for idx, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            raise TypeError(f"Each stage must be an object, got: {type(stage)!r}")
        name = str(stage.get("name", f"stage{idx}"))
        safe_name = _sanitize_name(name)
        stage_dir = run_root / f"{idx:02d}_{safe_name}"
        stage_dirs.append(stage_dir)

    for idx, stage in enumerate(stages):
        name = str(stage.get("name", f"stage{idx + 1}"))
        safe_name = _sanitize_name(name)
        stage_dir = stage_dirs[idx]
        stage_dir.mkdir(parents=True, exist_ok=True)

        merged_args = _merge_dicts(base_args, stage.get("args"))
        run_steps = int(stage.get("run_steps", 0) or 0)
        if run_steps > 0:
            merged_args["max_steps"] = run_steps
        merged_args["out_dir"] = str(stage_dir)

        if append_stage_name and merged_args.get("wandb_run_name"):
            merged_args["wandb_run_name"] = f"{merged_args['wandb_run_name']}__{safe_name}"

        env_map = _merge_env(base_env, stage.get("env"))

        plans.append(
            {
                "index": idx + 1,
                "name": name,
                "safe_name": safe_name,
                "run_steps": run_steps,
                "stage_dir": str(stage_dir),
                "args": merged_args,
                "env": env_map,
                "stage_cfg": stage,
            }
        )

    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-stage DBP-JSCC training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multistage_training.json",
        help="Path to pipeline JSON config",
    )
    parser.add_argument(
        "--start_stage",
        type=int,
        default=1,
        help="1-based stage index to start from",
    )
    parser.add_argument(
        "--stop_stage",
        type=int,
        default=0,
        help="1-based stage index to stop after; 0 means run all stages",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved commands and exit without launching training",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    cfg = _load_json(config_path)
    stage_plans = _build_stage_plan(cfg)

    total_stages = len(stage_plans)
    start_stage = max(1, int(args.start_stage))
    stop_stage = total_stages if int(args.stop_stage) <= 0 else min(total_stages, int(args.stop_stage))
    if start_stage > stop_stage:
        raise ValueError(f"Invalid stage range: start_stage={start_stage}, stop_stage={stop_stage}")

    run_root = Path(stage_plans[0]["stage_dir"]).parent
    resolved_plan_path = run_root / "resolved_pipeline_plan.json"
    with resolved_plan_path.open("w", encoding="utf-8") as f:
        json.dump(stage_plans, f, ensure_ascii=False, indent=2)

    python_bin = str(cfg.get("python_bin") or sys.executable)

    for plan in stage_plans[start_stage - 1:stop_stage]:
        stage_args = deepcopy(plan["args"])
        try:
            resume_path = _resolve_resume(plan["index"] - 1, plan["stage_cfg"], [Path(p["stage_dir"]) for p in stage_plans])
        except FileNotFoundError:
            if args.dry_run and plan["index"] > 1 and "resume" not in plan["stage_cfg"]:
                prev_ckpt_dir = Path(stage_plans[plan["index"] - 2]["stage_dir"]) / "checkpoints"
                resume_path = Path(f"<latest:{prev_ckpt_dir}>")
            else:
                raise

        if resume_path is not None:
            stage_args["resume"] = str(resume_path)
        else:
            stage_args.pop("resume", None)

        cli = [python_bin, str(TRAIN_SCRIPT)] + _args_to_cli(stage_args)
        env = os.environ.copy()
        env.update(plan["env"])

        cmd_pretty = " ".join(shlex.quote(part) for part in cli)
        print(f"[Pipeline] Stage {plan['index']}: {plan['name']}")
        print(f"[Pipeline] Work dir : {ROOT_DIR}")
        print(f"[Pipeline] Output   : {plan['stage_dir']}")
        if stage_args.get("resume"):
            print(f"[Pipeline] Resume   : {stage_args['resume']}")
        print(f"[Pipeline] Command  : {cmd_pretty}")

        if args.dry_run:
            continue

        subprocess.run(cli, cwd=str(ROOT_DIR), env=env, check=True)

        latest_ckpt = _latest_checkpoint(Path(plan["stage_dir"]))
        if latest_ckpt is None:
            raise FileNotFoundError(
                f"Stage {plan['index']} finished but no checkpoint was found in "
                f"{Path(plan['stage_dir']) / 'checkpoints'}"
            )
        print(f"[Pipeline] Stage {plan['index']} completed, latest checkpoint: {latest_ckpt}")


if __name__ == "__main__":
    main()
