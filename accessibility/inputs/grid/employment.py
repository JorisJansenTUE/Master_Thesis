"""Extract relevant 100 m STATENT employment cells, without GIS dependencies.

Supports the original Employment-STATENT2023.zip and an extracted CSV.
The original Swiss grid cells are preserved rather than aggregating jobs to
origin hexagons. All coordinates are LV95 (EPSG:2056) metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

COLUMNS = ("E_KOORD", "N_KOORD", "B08EMPT")


@dataclass
class EmploymentDestinations:
    jobs: pd.DataFrame
    source_path: Path
    records_read: int
    records_within_bounds: int

    @classmethod
    def from_statent(
        cls,
        path: str | Path,
        *,
        bounds: tuple[float, float, float, float],
        chunksize: int = 40_000,
    ) -> "EmploymentDestinations":
        """Read STATENT 2023 cells within network bounds (plus snap buffer).

        ``bounds`` is (min_x, min_y, max_x, max_y) in EPSG:2056. It should
        cover the entire *network* catchment, not only origin hexagons.
        A cell's coordinates identify its 100 m southwest corner, so we add
        50 m to obtain its centre for snapping.
        """
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"STATENT file not found: {source}")
        minx, miny, maxx, maxy = map(float, bounds)
        if minx >= maxx or miny >= maxy:
            raise ValueError("Invalid STATENT filtering bounds")

        frames: list[pd.DataFrame] = []
        read_count = 0
        within_count = 0

        def consume(stream) -> None:
            nonlocal read_count, within_count
            reader = pd.read_csv(
                stream, sep=";", usecols=list(COLUMNS), chunksize=chunksize,
                encoding="utf-8-sig", low_memory=False,
            )
            for part in reader:
                read_count += len(part)
                x = pd.to_numeric(part["E_KOORD"], errors="coerce") + 50.0
                y = pd.to_numeric(part["N_KOORD"], errors="coerce") + 50.0
                count = pd.to_numeric(part["B08EMPT"], errors="coerce")
                inside = x.between(minx, maxx) & y.between(miny, maxy)
                within_count += int(inside.sum())
                keep = inside & count.notna() & (count > 0)
                if not keep.any():
                    continue
                sub = pd.DataFrame({
                    "x": x[keep].astype("float64"),
                    "y": y[keep].astype("float64"),
                    "jobs": count[keep].astype("float64"),
                })
                # IDs encode the original 100 m cell southwest corner.
                sub["job_id"] = (
                    (sub["x"] - 50).astype("int64").astype(str) + "_" +
                    (sub["y"] - 50).astype("int64").astype(str)
                )
                frames.append(sub[["job_id", "x", "y", "jobs"]])

        if source.suffix.lower() == ".zip":
            with ZipFile(source) as archive:
                candidates = [name for name in archive.namelist()
                              if Path(name).name == "STATENT_2023.csv"]
                if len(candidates) != 1:
                    raise ValueError("Expected exactly one STATENT_2023.csv in the ZIP")
                with archive.open(candidates[0]) as stream:
                    consume(stream)
        else:
            with source.open("rb") as stream:
                consume(stream)

        jobs = (pd.concat(frames, ignore_index=True) if frames else
                pd.DataFrame(columns=["job_id", "x", "y", "jobs"]))
        if jobs["job_id"].duplicated().any():
            raise ValueError("Duplicate STATENT cell coordinates in the selected area")
        return cls(jobs=jobs, source_path=source,
                   records_read=read_count, records_within_bounds=within_count)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.jobs.to_csv(target, index=False)
