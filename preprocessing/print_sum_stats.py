from pathlib import Path
import gzip
import xml.etree.ElementTree as ET

from preprocessing.utils import DATA_DIR


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

SCENARIO = DATA_DIR / "NDA_Eth_Model" / "Locarno"

NETWORK_FILE = SCENARIO / "locarno_network.xml.gz"
POPULATION_FILE = SCENARIO / "locarno_population.xml.gz"


# ------------------------------------------------------------------
# Count XML elements
# ------------------------------------------------------------------

def count_elements(file_path, target_tags):
    counts = {tag: 0 for tag in target_tags}

    opener = gzip.open if str(file_path).endswith(".gz") else open

    with opener(file_path, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            tag = elem.tag.split("}")[-1]

            if tag in counts:
                counts[tag] += 1

            elem.clear()

    return counts


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    network = count_elements(NETWORK_FILE, {"node", "link"})
    population = count_elements(POPULATION_FILE, {"person"})

    print("\nScenario statistics")
    print("-------------------")
    print(f"Nodes:  {network['node']:,}")
    print(f"Links:  {network['link']:,}")
    print(f"Agents: {population['person']:,}")


if __name__ == "__main__":
    main()