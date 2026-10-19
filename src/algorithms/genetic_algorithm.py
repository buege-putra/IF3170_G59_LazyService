from __future__ import annotations

import random
from typing import Any

from algorithms.base import LocalSearch, SearchResult
from core.model import Orientation, Position
from core.occupancy_grid import OccupancyGrid
from core.problem import Problem
from core.state import State

# Satu gen per paket, isinya (truk, posisi, orientasi)
Gene = tuple[int | None, Position | None, Orientation]


class GeneticAlgorithm(LocalSearch):
    name = "Genetic Algorithm"
    iteration_label = "Iterasi (generasi)"

    def __init__(self, problem: Problem, population_size: int, num_iterations: int, mutation_rate: float,
                 elite_count: int, reinsert: bool = True) -> None:
        super().__init__(problem)
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.reinsert = reinsert

    def parameter(self) -> dict[str, Any]:
        return {"population_size": self.population_size, "num_iterations": self.num_iterations,
                "mutation_rate": self.mutation_rate, "elite_count": self.elite_count, "reinsert": self.reinsert}

    def metrics(self, result: SearchResult) -> dict[str, Any]:
        return {"Populasi": self.population_size, "Iterasi": self.num_iterations}

    @staticmethod
    def to_chromosome(state: State) -> list[Gene]:
        chromosome = []
        for package_id in state.packages:
            chromosome.append((state.truck_of[package_id], state.positions[package_id], state.orientations[package_id]))
        return chromosome

    def from_chromosome(self, chromosome: list[Gene]) -> State:
        state = State(self.problem)
        all_ids = list(state.packages)
        for package_id, (_, _, orientation) in zip(all_ids, chromosome):
            state.orientations[package_id] = orientation

        # Repair, masukkan paket urut dari z terendah, yg melanggar constraint dikeluarkan
        insertion_order = []
        for package_id, (truck_index, position, _) in zip(all_ids, chromosome):
            if truck_index is not None and position is not None:
                insertion_order.append((position[2], package_id, truck_index, position))
        insertion_order.sort()

        dropped = []
        for _, package_id, truck_index, position in insertion_order:
            state.place(package_id, truck_index, position)
            if not state.valid_after_change([package_id]):
                state.place(package_id, None, None)
                dropped.append(package_id)

        if self.reinsert and dropped:
            self.reinsert_dropped(state, sorted(dropped, key=all_ids.index))
        return state

    @staticmethod
    def reinsert_dropped(state: State, package_ids: list[str]) -> None:
        # Heuristik repair, paket yg kebuang dimasukkan lagi (orientasi tetap) ke posisi valid terendah
        # (urutan z, x, y) di truk pertama yg masih muat
        grids = [OccupancyGrid(state, truck_index) for truck_index in range(len(state.trucks))]
        for package_id in package_ids:
            package = state.packages[package_id]
            dx, dy, dz = package.oriented_dimensions(state.orientations[package_id])
            for truck_index, truck in enumerate(state.trucks):
                if state.total_weight(truck_index) + package.weight > truck.max_capacity:
                    continue
                position = grids[truck_index].lowest_position(dx, dy, dz, package.is_fragile)
                if position is not None:
                    state.place(package_id, truck_index, position)
                    grids[truck_index].fill(state.box(package_id), package.is_fragile)
                    break

    @staticmethod
    def select(population: list[State], fitness: list[float]) -> State:
        # Peluang kepilih sebanding dgn fitness.
        # Fitness bisa negatif di objective lifo, jadi kalo ada yg negatif semuanya digeser biar minimalnya 0
        lowest = min(fitness)
        if lowest < 0:
            fitness = [value - lowest for value in fitness]
        if sum(fitness) == 0:
            return random.choice(population)
        return random.choices(population, weights=fitness, k=1)[0]

    def reproduce(self, parent_1: State, parent_2: State) -> State:
        # Crossover satu titik potong, hasilnya satu offspring
        parent_1_genes = self.to_chromosome(parent_1)
        parent_2_genes = self.to_chromosome(parent_2)
        cut_point = random.randint(1, len(parent_1_genes) - 1)
        child_genes = parent_1_genes[:cut_point] + parent_2_genes[cut_point:]
        return self.from_chromosome(child_genes)

    def search(self) -> SearchResult:
        population = [State.random_initial(self.problem) for _ in range(self.population_size)]
        initial_state = max(population, key=State.objective)

        history_max = []
        history_avg = []

        for generation in range(self.num_iterations + 1):
            fitness = [individual.objective() for individual in population]
            history_max.append(max(fitness))
            history_avg.append(sum(fitness) / self.population_size)
            if generation == self.num_iterations:
                break

            # Elitisme, individu terbaik langsung dibawa ke generasi berikutnya
            order = sorted(range(self.population_size), key=lambda index: fitness[index], reverse=True)
            new_population = [population[index] for index in order[:self.elite_count]]

            while len(new_population) < self.population_size:
                parent_1 = self.select(population, fitness)
                parent_2 = self.select(population, fitness)
                child = self.reproduce(parent_1, parent_2)
                if random.random() < self.mutation_rate:
                    child = child.random_successor()
                new_population.append(child)
            population = new_population

        return SearchResult(initial_state, max(population, key=State.objective), history_max,
                            iterations=self.num_iterations, info={"history_avg": history_avg})
