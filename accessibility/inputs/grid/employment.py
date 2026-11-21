"""Extract relevant 100 m STATENT employment cells, without GIS dependencies.

Supports the original Employment-STATENT2023.zip and an extracted CSV.
The original Swiss grid cells are preserved rather than aggregating jobs to
origin hexagons. All coordinates are LV95 (EPSG:2056) metres.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import math
from xml.etree import ElementTree as ET
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


    @classmethod
    def from_matsim_facilities(
        cls,
        path: str | Path,
        *,
        bounds: tuple[float, float, float, float],
        work_activity_types: tuple[str, ...] = ("work",),
    ) -> "EmploymentDestinations":
        """Extract one opportunity per facility with a work activity option.

        This is a count of *work-capable facilities*, not actual employees or
        job vacancies. Facilities with several matching activity options are
        counted once. Facility IDs remain unique; no spatial aggregation is
        performed, so multiple facilities at the same coordinates are valid.

        MATSim coordinates are assumed to be LV95 (EPSG:2056). ``bounds``
        filters work facilities against the same road-network extent used for
        STATENT, but does not constrain destinations to the study-area polygon.
        The file is read incrementally so national .xml.gz files can be large.
        """
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"MATSim facilities file not found: {source}")
        minx, miny, maxx, maxy = map(float, bounds)
        if not all(map(math.isfinite, (minx, miny, maxx, maxy))):
            raise ValueError("Facility filtering bounds must be finite")
        if minx >= maxx or miny >= maxy:
            raise ValueError("Invalid facility filtering bounds")
        types = {str(t).strip().lower() for t in work_activity_types}
        if not types or "" in types:
            raise ValueError("work_activity_types must contain valid activity names")

        records: list[tuple[str, float, float, float]] = []
        read_count = within_count = 0
        seen_ids: set[str] = set()
        open_file = gzip.open if source.suffix.lower() == ".gz" else open
        root = None
        with open_file(source, "rb") as stream:
            for event, elem in ET.iterparse(stream, events=("start", "end")):
                if event == "start":
                    if root is None:
                        root = elem
                    continue
                if elem.tag.rsplit("}", 1)[-1] != "facility":
                    continue

                read_count += 1
                has_work = any(
                    child.tag.rsplit("}", 1)[-1] == "activity"
                    and (child.get("type") or "").strip().lower() in types
                    for child in elem
                )
                if has_work:
                    facility_id = elem.get("id")
                    if not facility_id:
                        raise ValueError("Encountered work facility without an id")
                    try:
                        x = float(elem.attrib["x"])
                        y = float(elem.attrib["y"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Work facility {facility_id!r} lacks valid x/y coordinates"
                        ) from exc
                    if not (math.isfinite(x) and math.isfinite(y)):
                        raise ValueError(f"Non-finite coordinates for facility {facility_id!r}")
                    if minx <= x <= maxx and miny <= y <= maxy:
                        within_count += 1
                        if facility_id in seen_ids:
                            raise ValueError(f"Duplicate work facility ID {facility_id!r}")
                        seen_ids.add(facility_id)
                        records.append((facility_id, x, y, 1.0))

                # Keep child activity elements until the parent is closed.
                # Clear only completed facilities to bound XML memory usage.
                elem.clear()
                if root is not None:
                    root.clear()

        if not records:
            raise ValueError(
                f"No work facilities found in {source} within the specified bounds. "
                f"Check WORK_ACTIVITY_TYPES={sorted(types)} and the input CRS (EPSG:2056)."
            )
        jobs = pd.DataFrame.from_records(
            records, columns=["job_id", "x", "y", "jobs"],
        )
        return cls(
            jobs=jobs,
            source_path=source,
            records_read=read_count,
            records_within_bounds=within_count,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.jobs.to_csv(target, index=False)
