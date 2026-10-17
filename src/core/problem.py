from __future__ import annotations

from functools import cached_property

import numpy as np

from core.model import Package, Truck


class Problem:
    """Data persoalan yg gak berubah selama pencarian: daftar truk sama daftar paket."""

    def __init__(self, trucks: list[Truck], package_list: list[Package]) -> None:
        self.trucks = trucks
        self.package_list = package_list
        self.packages: dict[str, Package] = {package.id: package for package in package_list}

    @cached_property
    def upper_bound(self) -> int:
        # Knapsack 0/1 dgn dua batasan (total berat sama total volume) terhadap gabungan semua truk.
        # Susunan paket, support, fragile, sama pembagian paket ke tiap truk diignore,
        # jadi hasilnya cuma batas atas global optimum, belum tentu bisa dicapai.
        capacity = sum(truck.max_capacity for truck in self.trucks)
        truck_volume = sum(W * L * H for W, L, H in (truck.dimensions for truck in self.trucks))

        # max_value[batas_berat, batas_volume] = value max dgn total berat sama volume gak lebih dari batasnya
        max_value = np.zeros((capacity + 1, truck_volume + 1), dtype=np.int64)
        for package in self.package_list:
            w, l, h = package.dimensions
            volume = w * l * h
            if package.weight > capacity or volume > truck_volume:
                continue
            # Ruas kanan dihitung full dari tabel lama sblm ditulis, jadi tiap paket kepake maks sekali
            max_value[package.weight:, volume:] = np.maximum(
                max_value[package.weight:, volume:],
                max_value[:capacity + 1 - package.weight, :truck_volume + 1 - volume] + package.value)
        return int(max_value[capacity, truck_volume])
