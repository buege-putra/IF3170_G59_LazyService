from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.state import State
    from experiments.runner import ExperimentResult

COLOR_ENABLED = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def style(text: str, *code: str) -> str:
    if not COLOR_ENABLED:
        return text
    return f"\033[{';'.join(code)}m{text}\033[0m"


def bold(text: str) -> str:
    return style(text, "1")


def dim(text: str) -> str:
    return style(text, "2")


def accent(text: str) -> str:
    return style(text, "1", "36")


def green(text: str) -> str:
    return style(text, "32")


def red(text: str) -> str:
    return style(text, "31")


def clear_screen() -> None:
    # + scrollback
    if sys.stdout.isatty():
        print("\033[H\033[2J\033[3J", end="", flush=True)


def heading(text: str) -> str:
    return f"  {bold(text)}"


def section_heading(text: str) -> str:
    return f"\n  {accent(text)}"


def table(header: list[str], rows: list[list[Any]], alignments: list[str] | None = None, indent: str = "  ") -> str:
    text_rows = [[str(cell) for cell in row] for row in rows]
    alignments = alignments or ["<"] * len(header)
    widths = [max(len(cell) for cell in column) for column in zip(header, *text_rows)]

    def border_line(left: str, middle: str, right: str) -> str:
        return indent + left + middle.join("─" * (width + 2) for width in widths) + right

    def format_row(cells: list[str], is_header: bool = False) -> str:
        texts = [f" {cell:{direction}{width}} " for cell, direction, width in zip(cells, alignments, widths)]
        if is_header:
            texts = [bold(text) for text in texts]
        return indent + "│" + "│".join(texts) + "│"

    lines = [border_line("┌", "┬", "┐"), format_row(header, True), border_line("├", "┼", "┤")]
    lines += [format_row(row) for row in text_rows]
    lines.append(border_line("└", "┴", "┘"))
    return "\n".join(lines)


def state_table(state: State, name: str) -> str:
    lines = [f"\n  {bold(name)}  {dim(f'value {state.objective()}')}"]
    for truck_index, truck in enumerate(state.trucks):
        packages_inside = state.inside(truck_index)
        lines.append(f"  {truck.id}: berat {state.total_weight(truck_index)}/{truck.max_capacity}, "
                     f"{len(packages_inside)} paket")
        if len(packages_inside) == 0:
            continue
        rows = []
        for package_id in packages_inside:
            package = state.packages[package_id]
            x, y, z, dx, dy, dz = state.box(package_id)
            rows.append([package_id, f"({x}, {y}, {z})", f"{dx} x {dy} x {dz}",
                         ", ".join(state.orientations[package_id]),
                         package.value, package.weight, "ya" if package.is_fragile else ""])
        lines.append(table(["Paket", "Posisi (x, y, z)", "Ukuran (x, y, z)", "Orientasi (x, y, z)",
                            "Value", "Weight", "Fragile"],
                           rows, ["<", "<", "<", "<", ">", ">", "<"]))
    outside = state.outside()
    lines.append(f"  Di luar truk ({len(outside)}): {', '.join(outside) if outside else 'tidak ada'}")
    return "\n".join(lines)


def result_summary(result: ExperimentResult, upper_bound: int) -> str:
    lines = [section_heading(result.title)]
    if result.parameter:
        lines.append("  " + dim("Parameter: " + ", ".join(f"{name} = {value}"
                                                           for name, value in result.parameter.items())))

    header = ["Run", "Value awal", "Value akhir", *result.runs[0].metrics]
    rows = []
    for run in result.runs:
        rows.append([run.label, run.initial_state.objective(), run.final_state.objective(),
                     *run.metrics.values()])
    lines.append(table(header, rows, ["<"] + [">"] * (len(header) - 1)))

    best = max(result.runs, key=lambda run: run.final_state.objective())
    best_value = best.final_state.objective()
    percentage = f"{best_value / upper_bound * 100:.1f}% dari batas atas {upper_bound}" if upper_bound else ""
    best_label = best.label
    # Di GA, label run bareng konfigurasinya
    if "Populasi" in best.metrics:
        best_label += f", populasi {best.metrics['Populasi']}, iterasi {best.metrics['Iterasi']}"
    lines.append(f"\n  Run terbaik ({best_label}): value {green(bold(str(best_value)))}"
                 f"  {dim(percentage)}")
    lines.append(state_table(best.initial_state, "State awal"))
    lines.append(state_table(best.final_state, "State akhir"))
    return "\n".join(lines)
