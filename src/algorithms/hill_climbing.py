from __future__ import annotations

import random
from typing import Any

from algorithms.base import LocalSearch, SearchResult
from core.problem import Problem
from core.state import State


class HillClimbing(LocalSearch):
    @staticmethod
    def best_successor(state: State) -> tuple[int, State]:
        # Successor dgn value tertinggi (diacak kalo ada yg seri)
        successors = state.all_successors()
        if len(successors) == 0:
            return state.objective(), state
        successor_values = [successor.objective() for successor in successors]
        best_value = max(successor_values)
        best_candidates = []
        for successor, value in zip(successors, successor_values):
            if value == best_value:
                best_candidates.append(successor)
        return best_value, random.choice(best_candidates)


class SteepestAscentHC(HillClimbing):
    name = "Steepest Ascent Hill-Climbing"

    def search(self) -> SearchResult:
        initial_state = State.random_initial(self.problem)
        state = initial_state
        history = [state.objective()]

        while True:
            neighbor_value, neighbor = self.best_successor(state)
            if neighbor_value <= state.objective():
                break
            state = neighbor
            history.append(neighbor_value)

        return SearchResult(initial_state, state, history, iterations=len(history) - 1)


class SidewaysMoveHC(HillClimbing):
    name = "Hill-Climbing with Sideways Move"

    def __init__(self, problem: Problem, max_sideways: int) -> None:
        super().__init__(problem)
        self.max_sideways = max_sideways

    def parameter(self) -> dict[str, Any]:
        return {"max_sideways": self.max_sideways}

    def search(self) -> SearchResult:
        initial_state = State.random_initial(self.problem)
        state = initial_state
        history = [state.objective()]
        sideways_count = 0

        while True:
            neighbor_value, neighbor = self.best_successor(state)

            if neighbor_value < state.objective():
                break

            # Sideways move dibatasi max_sideways kali berturut-turut
            if neighbor_value == state.objective():
                if sideways_count >= self.max_sideways:
                    break
                sideways_count += 1
            else:
                sideways_count = 0

            state = neighbor
            history.append(neighbor_value)

        return SearchResult(initial_state, state, history, iterations=len(history) - 1)


class RandomRestartHC(HillClimbing):
    name = "Random Restart Hill-Climbing"
    iteration_label = "Iterasi (seluruh restart disambung)"

    def __init__(self, problem: Problem, max_restarts: int) -> None:
        super().__init__(problem)
        self.max_restarts = max_restarts

    def parameter(self) -> dict[str, Any]:
        return {"max_restarts": self.max_restarts}

    def search(self) -> SearchResult:
        # Run pertama + max_restarts kali restart, tiap run pake Steepest Ascent dari state awal random
        steepest = SteepestAscentHC(self.problem)
        restart_results = [steepest.search() for _ in range(self.max_restarts + 1)]
        best = max(restart_results, key=lambda result: result.final_state.objective())

        # History semua restart disambung jadi satu kurva
        history = []
        for result in restart_results:
            history.extend(result.history)

        iterations_per_restart = [result.iterations for result in restart_results]
        return SearchResult(restart_results[0].initial_state, best.final_state, history,
                            iterations=sum(iterations_per_restart),
                            info={"restarts": self.max_restarts, "iterations_per_restart": iterations_per_restart})

    def metrics(self, result: SearchResult) -> dict[str, Any]:
        iterations_per_restart = result.info["iterations_per_restart"]
        return {"Restart": result.info["restarts"],
                "Iterasi per restart": ", ".join(str(iterations) for iterations in iterations_per_restart)}


class StochasticHC(HillClimbing):
    name = "Stochastic Hill-Climbing"

    def __init__(self, problem: Problem, max_iterations: int) -> None:
        super().__init__(problem)
        self.max_iterations = max_iterations

    def parameter(self) -> dict[str, Any]:
        return {"max_iterations": self.max_iterations}

    def search(self) -> SearchResult:
        initial_state = State.random_initial(self.problem)
        state = initial_state
        history = [state.objective()]

        # Tiap iterasi bangkitin satu successor random, pindah cuma kalo lebih baik
        for _ in range(self.max_iterations):
            neighbor = state.random_successor()
            if neighbor.objective() > state.objective():
                state = neighbor
            history.append(state.objective())

        return SearchResult(initial_state, state, history, iterations=self.max_iterations)
