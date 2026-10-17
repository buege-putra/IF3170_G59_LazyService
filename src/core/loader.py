from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.model import Dimensions, Package, Truck
from core.problem import Problem


def get_field(data: Any, key: str, expected_type: type, context: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{context} harus berupa object JSON")
    if key not in data:
        raise ValueError(f"{context}: atribut '{key}' tidak ada")
    value = data[key]
    # bool turunan int, harus ditolak manual untuk atribut bilangan
    if not isinstance(value, expected_type) or (expected_type is not bool and isinstance(value, bool)):
        raise ValueError(f"{context}: atribut '{key}' harus bertipe {expected_type.__name__}")
    return value


def get_dimensions(data: Any, context: str) -> Dimensions:
    dims = get_field(data, "dimensions", dict, context)
    lengths = []
    for side in ("w", "l", "h"):
        length = get_field(dims, side, int, f"{context}.dimensions")
        if length < 1:
            raise ValueError(f"{context}.dimensions: '{side}' harus >= 1")
        lengths.append(length)
    return lengths[0], lengths[1], lengths[2]


def load_input(path: str | Path) -> Problem:
    with open(path) as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("isi file harus berupa object JSON")

    trucks = []
    for index, truck in enumerate(get_field(data, "trucks", list, "input")):
        context = f"trucks[{index}]"
        trucks.append(Truck(id=get_field(truck, "id", str, context),
                            dimensions=get_dimensions(truck, context),
                            max_capacity=get_field(truck, "max_capacity", int, context)))

    package_list = []
    for index, package in enumerate(get_field(data, "packages", list, "input")):
        context = f"packages[{index}]"
        package_list.append(Package(id=get_field(package, "id", str, context),
                                    dimensions=get_dimensions(package, context),
                                    value=get_field(package, "value", int, context),
                                    weight=get_field(package, "weight", int, context),
                                    is_fragile=get_field(package, "is_fragile", bool, context),
                                    eta=get_field(package, "eta", int, context)))

    if len(trucks) == 0:
        raise ValueError("minimal harus ada 1 truk")
    for truck in trucks:
        if truck.max_capacity < 0:
            raise ValueError(f"truk {truck.id}: max_capacity tidak boleh negatif")
    for package in package_list:
        if package.weight < 0:
            raise ValueError(f"paket {package.id}: weight tidak boleh negatif")
    # Move swap butuh minimal dua paket
    if len(package_list) < 2:
        raise ValueError("minimal harus ada 2 paket")
    for name, ids in (("truk", [truck.id for truck in trucks]), ("paket", [package.id for package in package_list])):
        if len(ids) != len(set(ids)):
            raise ValueError(f"id {name} harus unik")

    return Problem(trucks, package_list)
