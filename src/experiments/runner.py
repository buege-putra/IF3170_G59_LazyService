from __future__ import annotations

import os
import random
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from algorithms.base import LocalSearch, SearchResult
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.hill_climbing import (
    RandomRestartHC,
    SidewaysMoveHC,
    SteepestAscentHC,
    StochasticHC,
)
from algorithms.simulated_annealing import SimulatedAnnealing
from core.problem import Problem
from core.state import State
from experiments.output import save_run, save_summary
from experiments.parameter import Parameter, format_value
from visualization import plot

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Gambar satu plot di ax. Argumen kedua adalah index run yg disorot, atau None untuk semua run
PlotFn = Callable[["Axes", int | None], None]
# (index run dalam eksperimen, nomor run dalam konfigurasi, hasil)
RunEntry = tuple[int, int, SearchResult]


@dataclass
class ExperimentRun:
    label: str
    initial_state: State
    final_state: State
    # Metrik yg ditampilin di tabel hasil, e.g. {"Iterasi": 16, "Durasi": "1.74 s"}
    metrics: dict[str, Any]


class ExperimentResult:
    def __init__(self, title: str, parameter: dict[str, Any], folder: Path) -> None:
        self.title = title
        self.parameter = parameter
        self.folder = folder
        self.runs: list[ExperimentRun] = []
        self.plot: list[PlotFn] = []

    def add_run(
        self,
        label: str,
        initial_state: State,
        final_state: State,
        metrics: dict[str, Any],
    ) -> None:
        self.runs.append(ExperimentRun(label, initial_state, final_state, metrics))


def folder_with_parameter(name: str, parameter: dict[str, Any]) -> str:
    return " ".join(
        [name, *(f"{key}={format_value(value)}" for key, value in parameter.items())]
    )


def run_seeded(algorithm: LocalSearch, seed: int) -> SearchResult:
    random.seed(seed)
    return algorithm.run()


class Experiment:
    def __init__(
        self, problem: Problem, parameter: Parameter, folder_output: Path
    ) -> None:
        self.problem = problem
        self.parameter = parameter
        self.folder_output = folder_output
        self.report_progress: Callable[[str], None] = lambda text: None

    def objective_parameter(self) -> dict[str, Any]:
        # Kosong utk objective value biar nama folder output lama sama
        p = self.parameter
        if p.objective == "value":
            return {}
        if p.tie_breaker:
            return {"objective": "lifo", "tie_breaker": True}
        return {"objective": "lifo", "lambda": p.penalty_weight}

    def run_repeated(
        self,
        experiment_result: ExperimentResult,
        algorithm: LocalSearch,
        total_runs: int,
        configuration: str | None = None,
        file_prefix: str = "",
    ) -> list[RunEntry]:
        # Jalanin algoritma sebanyak n_runs kali.
        # configuration sama file_prefix buat ngebedain run antar-konfigurasi GA dlm satu eksperimen.
        p = self.parameter
        self.problem.configure_objective(p.objective, p.penalty_weight, p.tie_breaker)
        entries = []
        # Seed tiap run diturunkan dari seed global biar hasil tetap reproducible
        seeds = [random.getrandbits(32) for _ in range(self.parameter.n_runs)]
        description = f" ({configuration})" if configuration else ""
        finished = len(experiment_result.runs)

        def report_finished(future: Future[SearchResult]) -> None:
            nonlocal finished
            if not future.cancelled():
                finished += 1
                self.report_progress(
                    f"{experiment_result.title}: {finished}/{total_runs} run selesai{description}"
                )

        self.report_progress(
            f"{experiment_result.title}: {finished}/{total_runs} run selesai{description}"
        )
        pool = ProcessPoolExecutor(max_workers=min(len(seeds), os.cpu_count() or 1))
        try:
            futures = [pool.submit(run_seeded, algorithm, seed) for seed in seeds]
            for future in futures:
                future.add_done_callback(report_finished)
            for run_number, future in enumerate(futures, start=1):
                result = future.result()
                entries.append((len(experiment_result.runs), run_number, result))

                experiment_result.add_run(
                    f"Run {run_number}",
                    result.initial_state,
                    result.final_state,
                    {**algorithm.metrics(result), "Durasi": f"{result.duration:.2f} s"},
                )

                run_label = (
                    f"{configuration}, Run {run_number}"
                    if configuration
                    else f"Run {run_number}"
                )
                save_run(
                    experiment_result.folder,
                    f"{file_prefix}run_{run_number}",
                    f"{experiment_result.title} ({run_label})",
                    algorithm,
                    result,
                )
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
        return entries

    def run_algorithm(
        self, algorithm: LocalSearch, folder_name: str
    ) -> tuple[ExperimentResult, list[RunEntry]]:
        parameter = {**algorithm.parameter(), **self.objective_parameter()}
        folder_name = folder_with_parameter(
            folder_name, {**parameter, "n_runs": self.parameter.n_runs}
        )
        experiment_result = ExperimentResult(
            algorithm.name,
            {name: format_value(value) for name, value in parameter.items()},
            self.folder_output / folder_name,
        )
        entries = self.run_repeated(experiment_result, algorithm, self.parameter.n_runs)
        histories = [result.history for _, _, result in entries]
        experiment_result.plot.append(
            lambda ax, selected: plot.draw_objective(
                ax,
                histories,
                "Objective function",
                selected,
                xlabel=algorithm.iteration_label,
            )
        )
        return experiment_result, entries

    def finalize(self, experiment_result: ExperimentResult) -> ExperimentResult:
        save_summary(experiment_result)
        return experiment_result

    def steepest(self) -> ExperimentResult:
        experiment_result, _ = self.run_algorithm(
            SteepestAscentHC(self.problem), "hill_climbing/steepest_ascent"
        )
        return self.finalize(experiment_result)

    def sideways(self) -> ExperimentResult:
        experiment_result, _ = self.run_algorithm(
            SidewaysMoveHC(self.problem, self.parameter.max_sideways),
            "hill_climbing/sideways_move",
        )
        return self.finalize(experiment_result)

    def random_restart(self) -> ExperimentResult:
        experiment_result, _ = self.run_algorithm(
            RandomRestartHC(self.problem, self.parameter.max_restarts),
            "hill_climbing/random_restart",
        )
        return self.finalize(experiment_result)

    def stochastic(self) -> ExperimentResult:
        experiment_result, _ = self.run_algorithm(
            StochasticHC(self.problem, self.parameter.max_iterations),
            "hill_climbing/stochastic",
        )
        return self.finalize(experiment_result)

    def simulated_annealing(self) -> ExperimentResult:
        p = self.parameter
        algorithm = SimulatedAnnealing(
            self.problem, p.T0, p.alpha, p.T_min, p.stuck_limit
        )
        experiment_result, entries = self.run_algorithm(
            algorithm, "simulated_annealing"
        )
        histories_prob = [result.info["history_prob"] for _, _, result in entries]
        experiment_result.plot.append(
            lambda ax, selected: plot.draw_sa_probability(ax, histories_prob, selected)
        )
        return self.finalize(experiment_result)

    def run_ga(
        self,
        title: str,
        folder_name: str,
        control: dict[str, Any],
        configurations: list[tuple[int, int]],
    ) -> ExperimentResult:
        p = self.parameter
        parameter = {
            "mutation_rate": p.mutation_rate,
            "elite_count": p.elite_count,
            "reinsert": p.reinsert,
            **self.objective_parameter(),
        }
        folder_name = folder_with_parameter(
            folder_name, {**control, **parameter, "n_runs": p.n_runs}
        )
        experiment_result = ExperimentResult(
            title,
            {name: format_value(value) for name, value in parameter.items()},
            self.folder_output / "genetic_algorithm" / folder_name,
        )
        total_runs = len(configurations) * p.n_runs

        for population_size, num_iterations in configurations:
            algorithm = GeneticAlgorithm(
                self.problem,
                population_size,
                num_iterations,
                p.mutation_rate,
                p.elite_count,
                p.reinsert,
            )
            entries = self.run_repeated(
                experiment_result,
                algorithm,
                total_runs,
                f"populasi {population_size}, iterasi {num_iterations}",
                f"pop{population_size}_iter{num_iterations}_",
            )
            plot_title = f"populasi = {population_size}, iterasi = {num_iterations}"

            def draw_configuration(
                ax: Axes,
                selected: int | None,
                entries: list[RunEntry] = entries,
                plot_title: str = plot_title,
            ) -> None:
                plot.draw_ga(ax, entries, plot_title, selected)

            experiment_result.plot.append(draw_configuration)
        return self.finalize(experiment_result)

    def ga_population_control(self) -> ExperimentResult:
        population_size, iteration_options = self.parameter.ga_population_control
        return self.run_ga(
            "Genetic Algorithm: kontrol jumlah populasi",
            "population_control",
            {
                "population": population_size,
                "iterations": "-".join(map(str, iteration_options)),
            },
            [(population_size, num_iterations) for num_iterations in iteration_options],
        )

    def ga_iteration_control(self) -> ExperimentResult:
        num_iterations, population_options = self.parameter.ga_iteration_control
        return self.run_ga(
            "Genetic Algorithm: kontrol banyak iterasi",
            "iteration_control",
            {
                "iterations": num_iterations,
                "populations": "-".join(map(str, population_options)),
            },
            [
                (population_size, num_iterations)
                for population_size in population_options
            ],
        )
