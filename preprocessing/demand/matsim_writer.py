from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

from .data import SyntheticPerson


class MatsimPopulationWriter:
    """
    Writes a MATSim/eqasim-compatible population.

    The population contains one selected home-work-home plan per synthetic person:

        home -> leg -> work -> leg -> home

    No routes are written. MATSim/eqasim should route the legs later.
    """

    def __init__(
        self,
        persons: list[SyntheticPerson],
        write_person_attributes_inside_population: bool = True,
        write_leg_routing_mode_attribute: bool = True,
    ) -> None:
        self.persons = persons
        self.write_person_attributes_inside_population = write_person_attributes_inside_population
        self.write_leg_routing_mode_attribute = write_leg_routing_mode_attribute

    def write_population(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        root = ET.Element("population")

        for person in self.persons:
            person_el = ET.SubElement(root, "person", {"id": person.person_id})

            if self.write_person_attributes_inside_population:
                self._write_person_attributes_embedded(person_el, person)

            self._write_selected_home_work_home_plan(person_el, person)

        self._write_xml_gz(
            path=path,
            root=root,
            dtd='<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">',
        )

    def _write_selected_home_work_home_plan(self, person_el: ET.Element, person: SyntheticPerson) -> None:
        plan_el = ET.SubElement(person_el, "plan", {"selected": "yes"})

        self._add_activity(plan_el, "home", person.home.x, person.home.y, person.departure_time)
        self._add_leg(plan_el, person.initial_mode, person.departure_time)
        self._add_activity(plan_el, "work", person.work.x, person.work.y, person.work_end_time)
        self._add_leg(plan_el, person.initial_mode, person.work_end_time)
        self._add_activity(plan_el, "home", person.home.x, person.home.y)

    def _add_activity(
        self,
        parent: ET.Element,
        activity_type: str,
        x: float,
        y: float,
        end_time: str | None = None,
    ) -> None:
        attributes = {"type": activity_type, "x": f"{x:.3f}", "y": f"{y:.3f}"}
        if end_time is not None:
            attributes["end_time"] = end_time
        ET.SubElement(parent, "activity", attributes)

    def _add_leg(self, parent: ET.Element, mode: str, dep_time: str | None = None) -> None:
        attributes = {"mode": mode}
        if dep_time is not None:
            attributes["dep_time"] = dep_time

        leg_el = ET.SubElement(parent, "leg", attributes)

        if self.write_leg_routing_mode_attribute:
            attrs_el = ET.SubElement(leg_el, "attributes")
            self._add_attribute(attrs_el, "routingMode", "java.lang.String", mode)

    def _write_person_attributes_embedded(self, person_el: ET.Element, person: SyntheticPerson) -> None:
        attrs = person.attributes
        attributes_el = ET.SubElement(person_el, "attributes")

        self._add_attribute(attributes_el, "age", "java.lang.Integer", str(attrs.age))
        self._add_attribute(attributes_el, "employed", "java.lang.Boolean", str(attrs.employed).lower())
        self._add_attribute(attributes_el, "hasLicense", "java.lang.Boolean", str(attrs.has_license).lower())
        self._add_attribute(attributes_el, "carAvail", "java.lang.String", attrs.car_avail)
        self._add_attribute(attributes_el, "hasBike", "java.lang.Boolean", str(attrs.has_bike).lower())
        self._add_attribute(attributes_el, "ptSubscription", "java.lang.Boolean", str(attrs.pt_subscription).lower())
        self._add_attribute(attributes_el, "homeMunicipality", "java.lang.String", person.home.municipality)
        self._add_attribute(attributes_el, "workMunicipality", "java.lang.String", person.work.municipality)
        self._add_attribute(attributes_el, "initialMode", "java.lang.String", person.initial_mode)

    def write_attributes(self, path: Path) -> None:
        """
        Write a separate MATSim objectAttributes file.

        Keep this if your MATSim/eqasim config expects a separate personAttributesFile.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        root = ET.Element("objectAttributes")

        for person in self.persons:
            attrs = person.attributes
            obj = ET.SubElement(root, "object", {"id": person.person_id})

            self._add_attribute(obj, "age", "java.lang.Integer", str(attrs.age))
            self._add_attribute(obj, "employed", "java.lang.Boolean", str(attrs.employed).lower())
            self._add_attribute(obj, "hasLicense", "java.lang.Boolean", str(attrs.has_license).lower())
            self._add_attribute(obj, "carAvail", "java.lang.String", attrs.car_avail)
            self._add_attribute(obj, "hasBike", "java.lang.Boolean", str(attrs.has_bike).lower())
            self._add_attribute(obj, "ptSubscription", "java.lang.Boolean", str(attrs.pt_subscription).lower())
            self._add_attribute(obj, "homeMunicipality", "java.lang.String", person.home.municipality)
            self._add_attribute(obj, "workMunicipality", "java.lang.String", person.work.municipality)
            self._add_attribute(obj, "initialMode", "java.lang.String", person.initial_mode)

        self._write_xml_gz(
            path=path,
            root=root,
            dtd='<!DOCTYPE objectAttributes SYSTEM "http://www.matsim.org/files/dtd/objectattributes_v1.dtd">',
        )

    @staticmethod
    def _add_attribute(parent: ET.Element, name: str, class_name: str, value: str) -> None:
        element = ET.SubElement(parent, "attribute", {"name": name, "class": class_name})
        element.text = value

    @staticmethod
    def _write_xml_gz(path: Path, root: ET.Element, dtd: str) -> None:
        xml_bytes = ET.tostring(root, encoding="utf-8")

        with gzip.open(path, "wb") as file:
            file.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
            file.write(dtd.encode("utf-8"))
            file.write(b"\n")
            file.write(xml_bytes)