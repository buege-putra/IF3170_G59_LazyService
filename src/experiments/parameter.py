from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# (nilai kontrol, daftar variasi), e.g. (200, [250, 500, 1000])
ControlConfiguration = tuple[int, list[int]]


def to_int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError("harus bilangan bulat") from None


def to_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError("harus bilangan") from None
    if not math.isfinite(value):
        raise ValueError("harus bilangan")
    return value


def positive_int(text: str) -> int:
    value = to_int(text)
    if value < 1:
        raise ValueError("harus >= 1")
    return value


def non_negative_int(text: str) -> int:
    value = to_int(text)
    if value < 0:
        raise ValueError("harus >= 0")
    return value


def positive_float(text: str) -> float:
    value = to_float(text)
    if value <= 0:
        raise ValueError("harus > 0")
    return value


def cooling_factor(text: str) -> float:
    value = to_float(text)
    if not 0 < value < 1:
        raise ValueError("harus di antara 0 dan 1")
    return value


def probability(text: str) -> float:
    value = to_float(text)
    if not 0 <= value <= 1:
        raise ValueError("harus di antara 0 dan 1")
    return value


def control_configuration(
    parse_control: Callable[[str], int], parse_variation: Callable[[str], int]
) -> Callable[[str], ControlConfiguration]:
    # Format: "kontrol : variasi1,variasi2,..."
    def parse(text: str) -> ControlConfiguration:
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError("format harus kontrol : nilai1,nilai2,...")
        control, variations = parts
        return parse_control(control), [
            parse_variation(value) for value in variations.split(",")
        ]

    return parse


@dataclass(frozen=True)
class ParameterDefinition:
    code: str
    name: str
    description: str
    default: Any
    # Mengubah teks input jadi nilai parameter, throw ValueError if not valid.
    # None untuk parameter yg langsung diganti pas dipilih di menu, dibalik kalo on/off
    # atau pindah ke pilihan berikutnya di choices.
    parse: Callable[[str], Any] | None
    choices: tuple[Any, ...] = ()


PARAMETER_DEFINITIONS: list[tuple[str, list[ParameterDefinition]]] = [
    (
        "Umum",
        [
            ParameterDefinition(
                "1a", "n_runs", "Banyak run per algoritma/konfigurasi", 3, positive_int
            ),
            ParameterDefinition(
                "1b",
                "objective",
                "Objective function (value/lifo)",
                "value",
                None,
                ("value", "lifo"),
            ),
            ParameterDefinition(
                "1c",
                "penalty_weight",
                "LIFO: bobot penalti lambda",
                10.0,
                positive_float,
            ),
            ParameterDefinition(
                "1d",
                "tie_breaker",
                "LIFO: penalti hanya sebagai tie-breaker (lambda diabaikan)",
                False,
                None,
            ),
        ],
    ),
    (
        "Hill-Climbing",
        [
            ParameterDefinition(
                "2a",
                "max_sideways",
                "Sideways Move: maksimum sideways move",
                20,
                non_negative_int,
            ),
            ParameterDefinition(
                "2b",
                "max_restarts",
                "Random Restart: maksimum restart",
                5,
                non_negative_int,
            ),
            ParameterDefinition(
                "2c",
                "max_iterations",
                "Stochastic: maksimum iterasi",
                500,
                non_negative_int,
            ),
        ],
    ),
    (
        "Simulated Annealing",
        [
            ParameterDefinition("3a", "T0", "Temperatur awal T0", 50.0, positive_float),
            ParameterDefinition(
                "3b", "alpha", "Faktor pendinginan alpha", 0.995, cooling_factor
            ),
            ParameterDefinition(
                "3c", "T_min", "Temperatur minimum T_min", 0.01, positive_float
            ),
            ParameterDefinition(
                "3d",
                "stuck_limit",
                "Iterasi tanpa perubahan untuk dihitung stuck",
                100,
                positive_int,
            ),
        ],
    ),
    (
        "Genetic Algorithm",
        [
            ParameterDefinition(
                "4a", "mutation_rate", "Peluang mutasi", 0.1, probability
            ),
            ParameterDefinition(
                "4b", "elite_count", "Jumlah elit", 2, non_negative_int
            ),
            ParameterDefinition(
                "4c",
                "ga_population_control",
                "Kontrol jumlah populasi (populasi : iterasi1,iterasi2,...)",
                (200, [250, 500, 1000]),
                control_configuration(positive_int, non_negative_int),
            ),
            ParameterDefinition(
                "4d",
                "ga_iteration_control",
                "Kontrol banyak iterasi (iterasi : populasi1,populasi2,...)",
                (50, [100, 200, 400]),
                control_configuration(non_negative_int, positive_int),
            ),
            ParameterDefinition(
                "4e",
                "reinsert",
                "Sisip ulang paket yang terbuang saat repair",
                True,
                None,
            ),
        ],
    ),
]


class Parameter:
    n_runs: int
    objective: str
    penalty_weight: float
    tie_breaker: bool
    max_sideways: int
    max_restarts: int
    max_iterations: int
    T0: float
    alpha: float
    T_min: float
    stuck_limit: int
    mutation_rate: float
    elite_count: int
    ga_population_control: ControlConfiguration
    ga_iteration_control: ControlConfiguration
    reinsert: bool

    def __init__(self) -> None:
        for _, entries in PARAMETER_DEFINITIONS:
            for definition in entries:
                setattr(self, definition.name, copy.deepcopy(definition.default))


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if isinstance(value, tuple):
        control, variations = value
        return f"{control} : {','.join(str(item) for item in variations)}"
    return str(value)
