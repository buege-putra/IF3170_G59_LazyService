from __future__ import annotations

import numpy as np

from core.model import Box, Position
from core.state import State


def window_sums(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    # Jumlah tiap window dx x dy di grid 2D (pake prefix sum), hasilnya berukuran (W - dx + 1, L - dy + 1)
    prefix = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1), dtype=np.int64)
    prefix[1:, 1:] = grid.cumsum(axis=0).cumsum(axis=1)
    return prefix[dx:, dy:] - prefix[:-dx, dy:] - prefix[dx:, :-dy] + prefix[:-dx, :-dy]


class OccupancyGrid:
    """
    Grid okupansi satu truk, buat nyari posisi valid paket baru tanpa ngecek kandidat satu2.
    occupied[x, y, z] = 1 kalo sel ditempatin paket.
    fragile_cells[x, y, z] = 1 kalo sel ditempatin paket fragile.
    """

    def __init__(self, state: State, truck_index: int) -> None:
        W, L, H = state.trucks[truck_index].dimensions
        self.occupied = np.zeros((W, L, H), dtype=np.int64)
        self.fragile_cells = np.zeros((W, L, H), dtype=np.int64)
        for package_id, box in state.boxes.items():
            if state.truck_of[package_id] == truck_index:
                self.fill(box, state.packages[package_id].is_fragile)

    def fill(self, box: Box, is_fragile: bool) -> None:
        x, y, z, dx, dy, dz = box
        self.occupied[x:x + dx, y:y + dy, z:z + dz] = 1
        if is_fragile:
            self.fragile_cells[x:x + dx, y:y + dy, z:z + dz] = 1

    def lowest_position(self, dx: int, dy: int, dz: int, is_fragile: bool) -> Position | None:
        # Posisi valid pertama dgn urutan (z, x, y), sama kayak urutan candidate_positions,
        # atau None kalo gak ada.
        # Constraint yg dicek sama dgn valid_after_change untuk paket yg baru dimasukin.
        W, L, H = self.occupied.shape
        if dx > W or dy > L or dz > H:
            return None
        for z in range(H - dz + 1):
            valid = window_sums(self.occupied[:, :, z:z + dz].sum(axis=2), dx, dy) == 0
            if z > 0:
                # Karena gak ada tabrakan, sel terisi pas di bawah alas pasti punya paket yg bagian atasnya di z
                valid &= window_sums(self.occupied[:, :, z - 1], dx, dy) > 0
                valid &= window_sums(self.fragile_cells[:, :, z - 1], dx, dy) == 0
            if is_fragile and z + dz < H:
                # Paket fragile gak boleh jadi support paket yg udah ada di atasnya
                valid &= window_sums(self.occupied[:, :, z + dz], dx, dy) == 0
            index = np.flatnonzero(valid)
            if index.size > 0:
                x, y = divmod(int(index[0]), L - dy + 1)
                return x, y, z
        return None
