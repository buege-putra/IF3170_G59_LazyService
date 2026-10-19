from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from algorithms.base import LocalSearch, SearchResult
from core.state import State
from visualization import plot

if TYPE_CHECKING:
    from experiments.runner import ExperimentResult


def to_json_value(value: Any) -> Any:
    if isinstance(value, State):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def fits_one_line(value: Any) -> bool:
    # Skalar, list skalar, atau object skalar. Dipake buat isi object yg ditulis satu baris
    if isinstance(value, dict):
        return all(is_scalar(item) for item in value.values())
    if isinstance(value, list):
        return all(is_scalar(item) for item in value)
    return True


def format_json(value: Any, depth: int = 0) -> str:
    # Similar to json.dumps(indent=2), tapi container yg isinya simpel ditulis satu baris
    # biar array gak boros tempat
    indent = "  " * (depth + 1)
    closing_indent = "  " * depth
    if isinstance(value, dict):
        if all(fits_one_line(item) for item in value.values()):
            return json.dumps(value, ensure_ascii=False)
        parts = [
            f"{indent}{json.dumps(key, ensure_ascii=False)}: {format_json(item, depth + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(parts) + f"\n{closing_indent}}}"
    if isinstance(value, list):
        if all(is_scalar(item) for item in value):
            return json.dumps(value, ensure_ascii=False)
        parts = [f"{indent}{format_json(element, depth + 1)}" for element in value]
        return "[\n" + ",\n".join(parts) + f"\n{closing_indent}]"
    return json.dumps(value, ensure_ascii=False)


def save_json(folder: Path, file_name: str, data: Any) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / file_name, "w") as file:
        file.write(format_json(to_json_value(data)) + "\n")


def save_figure(fig: Figure, folder: Path, file_name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / file_name, dpi=120)
    plt.close(fig)


def save_run(
    folder: Path,
    file_name: str,
    title: str,
    algorithm: LocalSearch,
    result: SearchResult,
) -> None:
    # Gambar state awal/akhir sama data lengkap satu run
    save_figure(
        plot.draw_initial_final(result.initial_state, result.final_state, title),
        folder,
        f"{file_name}.png",
    )
    save_json(
        folder,
        f"{file_name}.json",
        {"parameters": algorithm.parameter(), **result.to_dict()},
    )


def save_summary(experiment_result: ExperimentResult) -> None:
    save_figure(
        plot.draw_summary(experiment_result.plot, experiment_result.title),
        experiment_result.folder,
        "plot.png",
    )
