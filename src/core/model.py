from __future__ import annotations

from itertools import permutations

# Orientasi ditulis sbg sisi paket yg sejajar sumbu (x, y, z),
# e.g. ("l", "w", "h") artinya sisi l sejajar sumbu x
Orientation = tuple[str, str, str]
Dimensions = tuple[int, int, int]
Position = tuple[int, int, int]
# (x, y, z, dx, dy, dz) isinya posisi sudut kiri-bawah-belakang sama ukuran paket setelah orientasinya diapply
Box = tuple[int, int, int, int, int, int]

ALL_ORIENTATIONS: list[Orientation] = [(a, b, c) for a, b, c in permutations(("w", "l", "h"))]


def rotate_90(orientation: Orientation, axis: str) -> Orientation:
    # Rotasi 90 derajat thd satu sumbu == tukar sisi yg sejajar sama dua sumbu lainnya
    x, y, z = orientation
    if axis == "x":
        return x, z, y
    if axis == "y":
        return z, y, x
    return y, x, z


class Package:
    def __init__(self, id: str, dimensions: Dimensions, value: int, weight: int, is_fragile: bool,
                 eta: int) -> None:
        self.id = id
        self.dimensions = dimensions   # (w, l, h)
        self.value = value
        self.weight = weight
        self.is_fragile = is_fragile
        self.eta = eta

    def oriented_dimensions(self, orientation: Orientation) -> Dimensions:
        # Panjang paket di sumbu x, y, z untuk orientasi tertentu
        w, l, h = self.dimensions
        side_lengths = {"w": w, "l": l, "h": h}
        return side_lengths[orientation[0]], side_lengths[orientation[1]], side_lengths[orientation[2]]


class Truck:
    def __init__(self, id: str, dimensions: Dimensions, max_capacity: int) -> None:
        self.id = id
        self.dimensions = dimensions   # (W, L, H)
        self.max_capacity = max_capacity
