"""MATSim-authoritative modal connectivity check for car, bike and truck.

The existing Python directed-node validator runs first. This authoritative
check invokes org.thesis.ValidateNetworkConnectivity from the Switzerland JAR
and validates completed network XML using MATSim's NetworkUtils.cleanNetwork,
which also accounts for DisallowedNextLinks (including multi-link restrictions).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


VALIDATOR_MAIN = "org.thesis.ValidateNetworkConnectivity"


class MatsimConnectivityError(ValueError):
    """A written scenario network fails MATSim mode connectivity."""


class MatsimConnectivityExecutionError(RuntimeError):
    """The Java validator could not run; do not classify this as an invalid scenario."""


def check_matsim_connectivity(
    network: Path,
    jar: Path,
    *,
    java: str = "java",
    log_path: Path | None = None,
    modes: tuple[str, ...] = ("car", "bike", "truck"),
) -> dict[str, str]:
    """Validate routed modes on the final MATSim network (one Java process).

    Returns the MATSim result lines keyed by mode on success. Disconnected
    modes raise MatsimConnectivityError so a generator can resample. Java errors
    raise RuntimeError rather than falsely rejecting an otherwise valid sample.
    Neither the input XML nor the JAR is modified.
    """
    network = Path(network).resolve(strict=True)
    jar = Path(jar).resolve(strict=True)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("modes must contain unique mode names")

    command = [
        java, "-Xmx1g", "-XX:ActiveProcessorCount=1", "-cp", str(jar),
        VALIDATOR_MAIN, str(network), *modes,
    ]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False, timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        raise MatsimConnectivityExecutionError(
            f"MATSim connectivity check timed out: {network}"
        ) from exc
    except OSError as exc:
        raise MatsimConnectivityExecutionError(
            f"Could not start MATSim connectivity validator: {exc}"
        ) from exc

    if log_path is not None:
        Path(log_path).write_text(result.stdout, encoding="utf-8")

    summaries: dict[str, str] = {}
    statuses: dict[str, str] = {}
    pattern = re.compile(r"^MATSIM_CONNECTIVITY_CHECK mode=([^ ]+) status=(PASS|FAIL) ")
    for line in result.stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            mode, status = match.groups()
            if mode in summaries:
                raise MatsimConnectivityExecutionError(f"MATSim connectivity validator reported {mode} twice")
            summaries[mode] = line.strip()
            statuses[mode] = status

    expected = set(modes)
    if (result.returncode == 0 and set(summaries) == expected
            and all(status == "PASS" for status in statuses.values())):
        return summaries
    if (result.returncode == 2 and set(summaries) == expected
            and any(status == "FAIL" for status in statuses.values())):
        raise MatsimConnectivityError("; ".join(summaries[mode] for mode in modes))

    tail = "\n".join(result.stdout.splitlines()[-15:])
    raise MatsimConnectivityExecutionError(
        f"MATSim connectivity validator failed to execute (exit={result.returncode}).\n{tail}"
    )


def main() -> None:
    """Manual check: python -m network_sampling.validation.matsim_connectivity NETWORK JAR"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path)
    parser.add_argument("jar", type=Path)
    parser.add_argument("--java", default="java")
    parser.add_argument("--modes", nargs="+", default=["car", "bike", "truck"])
    args = parser.parse_args()
    try:
        for summary in check_matsim_connectivity(args.network, args.jar, java=args.java, modes=tuple(args.modes)).values():
            print(summary)
    except MatsimConnectivityError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
