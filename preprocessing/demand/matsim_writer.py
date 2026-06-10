from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

from .data import SyntheticPerson


class MatsimPopulationWriter:
    def __init__(self, persons: list[SyntheticPerson]) -> None:
        self.persons = persons

    def write_population(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("population")

        for person in self.persons:
            person_el = ET.SubElement(
                root,
                "person",
                {"id": person.person_id},
            )

            plan_el = ET.SubElement(
                person_el,
                "plan",
                {"selected": "yes"},
            )

            ET.SubElement(
                plan_el,
                "act",
                {
                    "type": "home",
                    "x": f"{person.home.x:.3f}",
                    "y": f"{person.home.y:.3f}",
                    "end_time": person.departure_time,
                },
            )

            ET.SubElement(
                plan_el,
                "leg",
                {"mode": person.initial_mode},
            )

            ET.SubElement(
                plan_el,
                "act",
                {
                    "type": "work",
                    "x": f"{person.work.x:.3f}",
                    "y": f"{person.work.y:.3f}",
                    "end_time": person.work_end_time,
                },
            )

            ET.SubElement(
                plan_el,
                "leg",
                {"mode": person.initial_mode},
            )

            ET.SubElement(
                plan_el,
                "act",
                {
                    "type": "home",
                    "x": f"{person.home.x:.3f}",
                    "y": f"{person.home.y:.3f}",
                },
            )

        self._write_xml_gz(
            path=path,
            root=root,
            dtd='<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">',
        )

    def write_attributes(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("objectAttributes")

        for person in self.persons:
            attrs = person.attributes

            obj = ET.SubElement(
                root,
                "object",
                {"id": person.person_id},
            )

            self._add_attribute(
                obj,
                "age",
                "java.lang.Integer",
                str(attrs.age),
            )

            self._add_attribute(
                obj,
                "employed",
                "java.lang.Boolean",
                str(attrs.employed).lower(),
            )

            self._add_attribute(
                obj,
                "hasLicense",
                "java.lang.Boolean",
                str(attrs.has_license).lower(),
            )

            self._add_attribute(
                obj,
                "carAvail",
                "java.lang.String",
                attrs.car_avail,
            )

            self._add_attribute(
                obj,
                "hasBike",
                "java.lang.Boolean",
                str(attrs.has_bike).lower(),
            )

            self._add_attribute(
                obj,
                "ptSubscription",
                "java.lang.Boolean",
                str(attrs.pt_subscription).lower(),
            )

            self._add_attribute(
                obj,
                "homeMunicipality",
                "java.lang.String",
                person.home.municipality,
            )

            self._add_attribute(
                obj,
                "workMunicipality",
                "java.lang.String",
                person.work.municipality,
            )

        self._write_xml_gz(
            path=path,
            root=root,
            dtd='<!DOCTYPE objectAttributes SYSTEM "http://www.matsim.org/files/dtd/objectattributes_v1.dtd">',
        )

    @staticmethod
    def _add_attribute(
        parent: ET.Element,
        name: str,
        class_name: str,
        value: str,
    ) -> None:
        element = ET.SubElement(
            parent,
            "attribute",
            {
                "name": name,
                "class": class_name,
            },
        )

        element.text = value

    @staticmethod
    def _write_xml_gz(
        path: Path,
        root: ET.Element,
        dtd: str,
    ) -> None:
        xml_bytes = ET.tostring(root, encoding="utf-8")

        with gzip.open(path, "wb") as file:
            file.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
            file.write(dtd.encode("utf-8"))
            file.write(b"\n")
            file.write(xml_bytes)