from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunnerConfig:
    raw: dict[str, Any]
    root: Path

    @property
    def crs(self) -> str:
        return self.raw.get("crs", "EPSG:2056")

    @property
    def sample_fraction(self) -> float:
        return float(self.raw.get("sample_fraction", 1.0))

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("random_seed", 1234))

    @property
    def columns(self) -> dict[str, str]:
        return self.raw["columns"]

    def path(self, *keys: str) -> Path:
        value: Any = self.raw
        for key in keys:
            value = value[key]

        path = Path(value)

        if path.is_absolute():
            return path

        return self.root / path


def load_config(path: Path) -> RunnerConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return RunnerConfig(
        raw=raw,
        root=Path.cwd(),
    )