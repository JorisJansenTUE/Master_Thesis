from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunnerConfig, load_config
from .load import prepare_inputs
from .matsim_writer import MatsimPopulationWriter
from .synthesis import PopulationSynthesizer
from .validation import print_summary


def run(config_path: Path) -> None:
    cfg = load_config(config_path)

    np.random.seed(cfg.random_seed)

    print(f"Reading input data from config: {config_path}")
    input_data = prepare_inputs(cfg)

    pop_col = cfg.columns["statpop_population"]
    flow_col = cfg.columns["od_flow"]

    # Perform checks and print summary of population and commuters
    actual_residents = input_data.statpop[pop_col].sum()
    expected_resident_sample = actual_residents * cfg.sample_fraction

    actual_commuters = input_data.od[flow_col].sum()
    expected_commuter_sample = actual_commuters * cfg.sample_fraction

    print("\nPopulation checks")
    print("-----------------")
    print(f"Actual residents in clipped STATPOP:       {actual_residents:,.0f}")
    print(f"Expected {cfg.sample_fraction:.1%} resident sample:        {expected_resident_sample:,.0f}")
    print(f"Total commuters in filtered OD:            {actual_commuters:,.0f}")
    print(f"Expected {cfg.sample_fraction:.1%} commuter sample:        {expected_commuter_sample:,.0f}")

    print("\nSynthesizing population...")
    synthesizer = PopulationSynthesizer(
        cfg=cfg,
        input_data=input_data,
    )

    persons = synthesizer.synthesize()

    print_summary(persons)

    output_dir = cfg.path("output", "directory")
    population_path = output_dir / cfg.raw["output"]["population"]
    attributes_path = output_dir / cfg.raw["output"]["population_attributes"]

    writer = MatsimPopulationWriter(persons)
    writer.write_population(population_path)

    # Optionally write person attributes in a separate file (currently not used as the attributes are embedded in the population XML)
    #writer.write_attributes(attributes_path)

    print("\nWritten files:")
    print(f"  {population_path}")
    #print(f"  {attributes_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the Locarno minimal Eqasim demand config.yml",
    )

    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()