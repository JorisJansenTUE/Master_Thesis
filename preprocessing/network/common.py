from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    root = logging.getLogger()
    root.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable '{name}' was not found on PATH. "
            f"On Windows, Maven is usually available as 'mvn.cmd'."
        )


def run_command(command: list[str | Path], cwd: Optional[Path] = None) -> None:
    command_as_str = [str(part) for part in command]
    logging.info("Running command: %s", " ".join(command_as_str))

    result = subprocess.run(
        command_as_str,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if result.stdout:
        logging.info("Command output:\n%s", result.stdout.strip())

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(command_as_str)}"
        )


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"Output directory already exists: {path}\n"
                f"Use --overwrite if you want to replace it."
            )
    path.mkdir(parents=True, exist_ok=True)


def create_parent_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()