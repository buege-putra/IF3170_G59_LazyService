from __future__ import annotations

import math
import random
from typing import Any

from algorithms.base import LocalSearch, SearchResult
from core.problem import Problem
from core.state import State


class SimulatedAnnealing(LocalSearch):
    name = "Simulated Annealing"

    def __init__(
        self, problem: Problem, T0: float, alpha: float, T_min: float, stuck_limit: int
    ) -> None:
        super().__init__(problem)
        self.T0 = T0
        self.alpha = alpha
        self.T_min = T_min
        self.stuck_limit = stuck_limit

    def parameter(self) -> dict[str, Any]:
        return {
            "T0": self.T0,
            "alpha": self.alpha,
            "T_min": self.T_min,
            "stuck_limit": self.stuck_limit,
        }

    def search(self) -> SearchResult:
        # Temperatur turun geometrik T_{t+1} = alpha * T_t, berhenti pas T < T_min
        initial_state = State.random_initial(self.problem)
        state = initial_state
        T = self.T0
        history = [state.objective()]
        # e^(ΔE/T) untuk neighbor yg lebih jelek, NaN untuk sisanya
        history_prob: list[float] = []
        stuck_count = 0
        unchanged_iterations = 0

        while T > self.T_min:
            neighbor = state.random_successor()
            delta_E = neighbor.objective() - state.objective()

            if delta_E > 0:
                state = neighbor
                history_prob.append(float("nan"))
            else:
                prob = math.exp(delta_E / T)
                if delta_E < 0:
                    history_prob.append(prob)
                else:
                    history_prob.append(float("nan"))
                if random.random() < prob:
                    state = neighbor

            # Dihitung stuck jika value tidak berubah selama stuck_limit iterasi berturut-turut
            if state.objective() == history[-1]:
                unchanged_iterations += 1
            else:
                unchanged_iterations = 0
            if unchanged_iterations == self.stuck_limit:
                stuck_count += 1
                unchanged_iterations = 0

            history.append(state.objective())
            T = self.alpha * T

        return SearchResult(
            initial_state,
            state,
            history,
            iterations=len(history) - 1,
            info={"stuck": stuck_count, "history_prob": history_prob},
        )

    def metrics(self, result: SearchResult) -> dict[str, Any]:
        return {"Iterasi": result.iterations, "Stuck": result.info["stuck"]}
