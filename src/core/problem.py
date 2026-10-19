from __future__ import annotations

from functools import cached_property

import numpy as np

from core.model import Package, Truck


class Problem:
    """Data persoalan yg gak berubah selama pencarian: daftar truk, daftar paket, dan konfigurasi objective."""

    def __init__(self, trucks: list[Truck], package_list: list[Package]) -> None:
        self.trucks = trucks
        self.package_list = package_list
        self.packages: dict[str, Package] = {
            package.id: package for package in package_list
        }
        etas = [package.eta for package in package_list]
        self.eta_range = max(etas) - min(etas)
        self.configure_objective("value", 10.0, False)

    def configure_objective(
        self,
        objective: str,
        penalty_weight: float,
        tie_breaker: bool,
        show_penalty: bool = False,
    ) -> None:
        # objective "value" = total value
        # objective "lifo" = total value - penalty
        # penalty_scale ngubah raw penalty (jumlah selisih ETA) jadi pengurang objective
        self.objective = objective
        self.penalty_weight = penalty_weight
        self.tie_breaker = tie_breaker
        # Penalti selalu ditampilkan di objective lifo, di objective value cuma kalo diminta
        self.penalty_shown = objective == "lifo" or show_penalty
        if objective == "value" or self.eta_range == 0:
            self.penalty_scale = 0.0
        elif tie_breaker:
            # Tiap pasangan nyumbang max 1 setelah dinormalisasi, total penalty < 1
            num_packages = len(self.package_list)
            max_pairs = num_packages * (num_packages - 1) // 2
            self.penalty_scale = 1 / (self.eta_range * (max_pairs + 1))
        else:
            self.penalty_scale = penalty_weight / self.eta_range

    def objective_setting(self) -> str:
        # Parameter objective lifo yg ditampilkan di judul gambar
        if self.objective == "value":
            return ""
        if self.tie_breaker:
            return "objective lifo, tie-breaker"
        return f"objective lifo, lambda = {self.penalty_weight}"

    @cached_property
    def upper_bound(self) -> int:
        # Knapsack 0/1 dgn dua batasan (total berat sama total volume) terhadap gabungan semua truk.
        # Susunan paket, support, fragile, sama pembagian paket ke tiap truk diignore,
        # jadi hasilnya cuma batas atas global optimum, belum tentu bisa dicapai.
        capacity = sum(truck.max_capacity for truck in self.trucks)
        truck_volume = sum(
            W * L * H for W, L, H in (truck.dimensions for truck in self.trucks)
        )

        # max_value[batas_berat, batas_volume] = value max dgn total berat sama volume gak lebih dari batasnya
        max_value = np.zeros((capacity + 1, truck_volume + 1), dtype=np.int64)
        for package in self.package_list:
            w, l, h = package.dimensions
            volume = w * l * h
            if package.weight > capacity or volume > truck_volume:
                continue
            # Ruas kanan dihitung full dari tabel lama sblm ditulis, jadi tiap paket kepake maks sekali
            max_value[package.weight :, volume:] = np.maximum(
                max_value[package.weight :, volume:],
                max_value[: capacity + 1 - package.weight, : truck_volume + 1 - volume]
                + package.value,
            )
        return int(max_value[capacity, truck_volume])
