from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.problem import Problem
from core.state import State


@dataclass
class SearchResult:
    initial_state: State
    final_state: State
    # Nilai objective tiap iterasi. Di GA, isinya nilai max populasi tiap generasi
    history: list[int]
    iterations: int
    duration: float = 0.0
    # Data khusus per algoritma, e.g. history_prob di SA atau history_avg di GA
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"initial_state": self.initial_state, "final_state": self.final_state, "history": self.history,
                "iterations": self.iterations, "duration": self.duration, **self.info}


class LocalSearch:
    name = ""
    iteration_label = "Iterasi"

    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def run(self) -> SearchResult:
        start_time = time.time()
        result = self.search()
        result.duration = time.time() - start_time
        return result

    def search(self) -> SearchResult:
        raise NotImplementedError

    def parameter(self) -> dict[str, Any]:
        return {}

    def metrics(self, result: SearchResult) -> dict[str, Any]:
        # Metrik di tabel hasil, selain value & durasi
        return {"Iterasi": result.iterations}
