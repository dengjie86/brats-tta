#!/usr/bin/env python3
"""Send a compact source-training progress report over SMTP."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import smtplib
import subprocess
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

ITERATION_RE = re.compile(
    r"Epoch (?P<epoch>\d+)(?:/(?P<epochs>\d+))? train iteration "
    r"(?P<iteration>\d+)/(?P<iterations>\d+): .*?"
    r"rank0_loss=(?P<loss>[-+0-9.eE]+), "
    r"rank0_running_loss=(?P<running_loss>[-+0-9.eE]+), "
    r"lr=(?P<lr>[-+0-9.eE]+)"
)


def _read_tail(path: Path, limit: int = 256 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")


def _last_json_record(path: Path) -> dict[str, Any] | None:
    for line in reversed(_read_tail(path).splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _last_validation_record(path: Path) -> dict[str, Any] | None:
    for line in reversed(_read_tail(path, limit=2 * 1024 * 1024).splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("val_dice_mean") is not None:
            return value
    return None


def _last_iteration(log_text: str, total_epochs: int) -> dict[str, Any] | None:
    matches = list(ITERATION_RE.finditer(log_text))
    if not matches:
        return None
    match = matches[-1]
    result = match.groupdict()
    result["epochs"] = result["epochs"] or total_epochs
    for key in ("epoch", "epochs", "iteration", "iterations"):
        result[key] = int(result[key])
    for key in ("loss", "running_loss", "lr"):
        result[key] = float(result[key])
    return result


def _last_gpu_sample(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = [row for row in csv.DictReader(handle) if any(value.strip() for value in row.values())]
    return rows[-1] if rows else None


def _gpu_value(sample: dict[str, str], key: str) -> str:
    return sample.get(f" {key}", sample.get(key, "N/A")).strip()


def _process_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "brats_tta.cli.train_source"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number or number in {float("inf"), float("-inf")}:
        return "N/A"
    return f"{number:.{digits}f}"


def collect_report(work_root: Path, total_epochs: int) -> tuple[str, str]:
    run_root = work_root / "run"
    log_text = _read_tail(run_root / "training.log")
    history_path = run_root / "history.jsonl"
    history = _last_json_record(history_path)
    validation = _last_validation_record(history_path)
    iteration = _last_iteration(log_text, total_epochs)
    gpu = _last_gpu_sample(work_root / "gpu_utilization.csv")
    running = _process_running()

    status = "RUNNING" if running else "STOPPED or FINISHED"
    if "Source training failed" in log_text:
        status = "FAILED"
    elif history and int(history.get("completed_epoch", 0)) >= total_epochs:
        status = "FINISHED"

    subject_epoch = "?"
    subject_iteration = "?"
    if iteration:
        subject_epoch = str(iteration["epoch"])
        subject_iteration = str(iteration["iteration"])
    subject = f"BraTS source training [{status}] epoch {subject_epoch}, iter {subject_iteration}"

    lines = [
        "BraTS source-domain training progress",
        "",
        f"Status: {status}",
        f"Time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Work root: {work_root}",
    ]
    if iteration:
        lines.extend(
            [
                f"Current epoch: {iteration['epoch']}/{iteration['epochs']}",
                f"Current iteration: {iteration['iteration']}/{iteration['iterations']}",
                f"Last loss: {_format_number(iteration['loss'], 6)}",
                f"Running loss: {_format_number(iteration['running_loss'], 6)}",
                f"Learning rate: {_format_number(iteration['lr'], 8)}",
            ]
        )
    else:
        lines.append("Current iteration: not available yet")

    if history:
        lines.extend(
            [
                f"Last completed epoch: {history.get('completed_epoch', 'N/A')}",
                f"Epoch train loss: {_format_number(history.get('train_loss'), 6)}",
                f"Best validation Dice: {_format_number(history.get('best_dice'), 6)}",
            ]
        )
    if validation:
        lines.extend(
            [
                f"Latest validation epoch: {validation.get('completed_epoch', 'N/A')}",
                f"Validation Dice ET: {_format_number(validation.get('val_dice_ET'), 6)}",
                f"Validation Dice TC: {_format_number(validation.get('val_dice_TC'), 6)}",
                f"Validation Dice WT: {_format_number(validation.get('val_dice_WT'), 6)}",
                f"Validation Dice mean: {_format_number(validation.get('val_dice_mean'), 6)}",
            ]
        )
    else:
        lines.append("Validation Dice: not available yet")

    if gpu:
        lines.extend(
            [
                f"GPU utilization: {_gpu_value(gpu, 'utilization.gpu [%]')}%",
                (
                    f"GPU memory: {_gpu_value(gpu, 'memory.used [MiB]')} MiB / "
                    f"{_gpu_value(gpu, 'memory.total [MiB]')} MiB"
                ),
                f"GPU power: {_gpu_value(gpu, 'power.draw [W]')} W",
                f"GPU temperature: {_gpu_value(gpu, 'temperature.gpu')} C",
            ]
        )
    else:
        lines.append("GPU telemetry: not available yet")
    lines.extend(["", "The report is generated automatically every two hours."])
    return subject, "\n".join(lines) + "\n"


def send_report(
    *,
    recipient: str,
    smtp_user: str,
    password_file: Path,
    smtp_host: str,
    smtp_port: int,
    subject: str,
    body: str,
) -> None:
    password = password_file.read_text(encoding="utf-8").strip()
    if not password:
        raise RuntimeError(f"SMTP password file is empty: {password_file}")
    message = EmailMessage()
    message["From"] = smtp_user
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
        server.login(smtp_user, password)
        server.send_message(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--recipient", required=True)
    parser.add_argument("--smtp-user", required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--smtp-host", default="smtp.qq.com")
    parser.add_argument("--smtp-port", type=int, default=465)
    parser.add_argument("--total-epochs", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    subject, body = collect_report(args.work_root.expanduser().resolve(), args.total_epochs)
    if args.dry_run:
        print(subject)
        print(body, end="")
        return
    send_report(
        recipient=args.recipient,
        smtp_user=args.smtp_user,
        password_file=args.password_file.expanduser().resolve(),
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        subject=subject,
        body=body,
    )


if __name__ == "__main__":
    main()
