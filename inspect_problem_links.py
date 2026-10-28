import gzip
import xml.etree.ElementTree as ET

NETWORK = "C:\\Users\\20201733\\Downloads\\final_baseline.xml.gz"

problem_links = {
    "661871",
    "204974",
    "527930",
    "204962",
    "897257",
    "954172",
    "1154989",
}

with gzip.open(NETWORK, "rb") as f:
    for event, elem in ET.iterparse(f, events=("end",)):
        if elem.tag == "link" and elem.get("id") in problem_links:
            print(
                f"Link: {elem.get('id')}, "
                f"Modes: {elem.get('modes')}, "
                f"From: {elem.get('from')}, "
                f"To: {elem.get('to')}"
            )

            problem_links.remove(elem.get("id"))

        elem.clear()

print("Links not found:", problem_links)