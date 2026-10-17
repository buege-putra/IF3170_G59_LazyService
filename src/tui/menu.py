from __future__ import annotations

import random
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.parameter import PARAMETER_DEFINITIONS, ParameterDefinition, format_value
from experiments.runner import Experiment, ExperimentResult
from tui.display import accent, bold, clear_screen, dim, heading, red, result_summary, table
from visualization.plot import show_results

ALL_HC = ["steepest", "sideways", "random_restart", "stochastic"]
ALL_GA = ["ga_population_control", "ga_iteration_control"]


class TuiState:
    def __init__(self, experiment: Experiment, path_input: str, seed: int) -> None:
        self.experiment = experiment
        self.path_input = path_input
        self.seed = seed
        self.visualize = True
        self.upper_bound = experiment.problem.upper_bound
        # Pesan ditampilin sekali setelah layar diclear, e.g. pesan error input
        self.message: str | None = None

    def print_message(self) -> None:
        if self.message:
            print(f"\n  {self.message}")
            self.message = None


# Aksi menu bisa berupa daftar nama method Experiment yg dijalanin, atau fungsi yg menerima TuiState
Action = list[str] | Callable[[TuiState], None]
MenuGroup = tuple[str, list[tuple[str, str]]]


def relative_path(path: Path) -> Path:
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


def ask(prompt: str) -> str:
    return input(prompt).strip().lower()


def wait_for_back() -> None:
    input(dim("\n  Tekan Enter untuk kembali..."))


def print_menu(menu_groups: list[MenuGroup]) -> None:
    for group_name, options in menu_groups:
        print(f"\n  {accent(group_name)}")
        for code, label in options:
            print(f"    {bold(f'[{code}]')} {label}")


def main_menu_items(status: TuiState) -> list[tuple[str, list[tuple[str, str, Action]]]]:
    # Tiap pilihan isinya (kode, label, aksi)
    return [
        ("Hill-Climbing", [
            ("1a", "Steepest Ascent", ["steepest"]),
            ("1b", "Sideways Move", ["sideways"]),
            ("1c", "Random Restart", ["random_restart"]),
            ("1d", "Stochastic", ["stochastic"]),
        ]),
        ("Simulated Annealing", [
            ("2", "Simulated Annealing", ["simulated_annealing"]),
        ]),
        ("Genetic Algorithm", [
            ("3a", "Kontrol jumlah populasi (variasi banyak iterasi)", ["ga_population_control"]),
            ("3b", "Kontrol banyak iterasi (variasi jumlah populasi)", ["ga_iteration_control"]),
        ]),
        ("Lainnya", [
            ("4a", "Jalankan semua eksperimen", ALL_HC + ["simulated_annealing"] + ALL_GA),
            ("4b", "Lihat data input", view_input),
        ]),
        ("Pengaturan", [
            ("5a", "Parameter algoritma", parameter_menu),
            ("5b", f"Seed: {status.seed}", change_seed),
            ("5c", f"Visualisasi setelah eksperimen: {'ON' if status.visualize else 'OFF'}", toggle_visualization),
        ]),
    ]


def view_input(status: TuiState) -> None:
    problem = status.experiment.problem
    clear_screen()
    print(heading(f"Data input: {status.path_input}"))
    print(f"\n  {accent('Truk')}")
    print(table(["Id", "Dimensi (w, l, h)", "Max capacity"],
                [[truck.id, str(truck.dimensions), truck.max_capacity] for truck in problem.trucks],
                ["<", "<", ">"]))
    print(f"\n  {accent('Paket')}")
    print(table(["Id", "Dimensi (w, l, h)", "Value", "Weight", "Fragile", "ETA"],
                [[package.id, str(package.dimensions), package.value, package.weight,
                  "ya" if package.is_fragile else "", package.eta] for package in problem.package_list],
                ["<", "<", ">", ">", "<", ">"]))
    print(f"\n  Batas atas value (knapsack berdasarkan berat dan volume): {bold(str(status.upper_bound))}")
    wait_for_back()


def parameter_menu(status: TuiState) -> None:
    parameter = status.experiment.parameter
    while True:
        clear_screen()
        print(heading("Parameter algoritma"))
        definition_by_code: dict[str, ParameterDefinition] = {}
        menu_groups: list[MenuGroup] = []
        for group_name, entries in PARAMETER_DEFINITIONS:
            options = []
            for definition in entries:
                definition_by_code[definition.code] = definition
                value = format_value(getattr(parameter, definition.name))
                options.append((definition.code, f"{definition.description}: {bold(value)}"))
            menu_groups.append((group_name, options))
        print_menu(menu_groups)
        status.print_message()

        selection = ask(f"\n  Pilihan {dim('(Enter untuk kembali)')}: ")
        if selection == "":
            return
        if selection not in definition_by_code:
            status.message = red(f"Pilihan tidak dikenal: {selection}")
            continue

        definition = definition_by_code[selection]
        if definition.parse is None:
            setattr(parameter, definition.name, not getattr(parameter, definition.name))
            continue

        text = input(f"  Nilai baru untuk {definition.description}: ")
        try:
            setattr(parameter, definition.name, definition.parse(text))
        except ValueError as error:
            status.message = red(f"Nilai tidak valid untuk {definition.description}: {error}")


def change_seed(status: TuiState) -> None:
    try:
        status.seed = int(input("  Seed baru: "))
    except ValueError:
        status.message = red("Seed harus bilangan bulat.")


def toggle_visualization(status: TuiState) -> None:
    status.visualize = not status.visualize


def report_progress(text: str) -> None:
    if sys.stdout.isatty():
        print(f"\r\033[K  {dim('Menjalankan ' + text + '...')}", end="", flush=True)


def clear_progress() -> None:
    if sys.stdout.isatty():
        print("\r\033[K", end="", flush=True)


def run_experiments(status: TuiState, methods: list[str]) -> None:
    experiment = status.experiment
    experiment.report_progress = report_progress
    # Seed di-reset tiap kali jalanin eksperimen biar reproducible
    random.seed(status.seed)

    results: list[ExperimentResult] = []
    try:
        for method in methods:
            results.append(getattr(experiment, method)())
    except KeyboardInterrupt:
        clear_progress()
        status.message = red("Eksperimen dibatalkan.")
        plt.close("all")
        return

    clear_screen()
    for result in results:
        print(result_summary(result, status.upper_bound))
        print(dim(f"\n  Hasil tersimpan di {relative_path(result.folder)}"))

    if status.visualize:
        print(dim("\n  Window visualisasi dibuka. Tutup window untuk melanjutkan."))
        if not show_results(results):
            print(red("  Visualisasi tidak dapat ditampilkan (tidak ada display). Gambar tetap tersimpan."))
    wait_for_back()


def run_tui(experiment: Experiment, path_input: str, seed: int) -> None:
    status = TuiState(experiment, path_input, seed)
    problem = experiment.problem

    while True:
        clear_screen()
        print(heading("Local Search: Pengisian Paket ke Truk Peti Kemas"))
        print(f"\n  {dim('Input ')}       {status.path_input} ({len(problem.package_list)} paket, "
              f"{len(problem.trucks)} truk)")
        print(f"  {dim('Batas atas')}   {status.upper_bound}  "
              f"{dim('(value maksimum jika hanya total berat dan volume dihitung, susunan paket diabaikan)')}")
        print(f"  {dim('Output')}       {relative_path(experiment.folder_output)}")

        menu = main_menu_items(status)
        print_menu([(name, [(code, label) for code, label, _ in entries]) for name, entries in menu])
        print(f"\n  {bold('[q]')} Keluar")
        status.print_message()
        action_by_code = {code: action for _, entries in menu for code, _, action in entries}

        try:
            selection = ask("\n  Pilihan: ")
            if selection == "q":
                clear_screen()
                return
            if selection == "":
                continue
            if selection not in action_by_code:
                status.message = red(f"Pilihan tidak dikenal: {selection}")
                continue

            action = action_by_code[selection]
            if isinstance(action, list):
                run_experiments(status, action)
            else:
                action(status)
        except (EOFError, KeyboardInterrupt):
            print()
            return
