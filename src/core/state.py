from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from itertools import combinations
from typing import Any

from core.model import ALL_ORIENTATIONS, Box, Orientation, Position, rotate_90
from core.problem import Problem

# ("swap", package_id_1, package_id_2), ("move", package_id, truck_index, new_position),
# atau ("rotate", package_id, axis)
Move = tuple[str, str, str] | tuple[str, str, int, Position]


def bases_overlap(box_1: Box, box_2: Box) -> bool:
    # Proyeksi dua kotak (x, y, z, dx, dy, dz) di bidang xy saling beririsan
    x1, y1, _, dx1, dy1, _ = box_1
    x2, y2, _, dx2, dy2, _ = box_2
    overlap_x = x1 < x2 + dx2 and x2 < x1 + dx1
    overlap_y = y1 < y2 + dy2 and y2 < y1 + dy1
    return overlap_x and overlap_y


def intervals_overlap(start_1: int, length_1: int, start_2: int, length_2: int) -> bool:
    return start_1 < start_2 + length_2 and start_2 < start_1 + length_1


def blocks(blocker: Box, target: Box, door_on_x: bool) -> bool:
    # blocker harus dipindah dulu supaya target bisa dikeluarin, karena blocker ada di atas target
    # atau ada di antara target dan pintu
    bx, by, bz, bdx, bdy, bdz = blocker
    tx, ty, tz, tdx, tdy, tdz = target
    if bz >= tz + tdz and bases_overlap(blocker, target):
        return True
    if not intervals_overlap(bz, bdz, tz, tdz):
        return False
    if door_on_x:
        return bx >= tx + tdx and intervals_overlap(by, bdy, ty, tdy)
    return by + bdy <= ty and intervals_overlap(bx, bdx, tx, tdx)


class State:
    """
    Utk tiap paket:
    - truck_of[package_id] = indeks truk tempat paketnya, atau None kalo di luar semua truk
    - positions[package_id] = sudut kiri-bawah-belakang (x, y, z) di truk itu, atau None
    - orientations[package_id] = orientasi paket

    Move neighbor:
    1. ("swap", package_id_1, package_id_2)
    2. ("move", package_id, truck_index, new_position)
    3. ("rotate", package_id, axis)
    Successor yg melanggar constraint dibuang.
    """

    def __init__(self, problem: Problem) -> None:
        # Awalnya semua paket di luar truk dgn orientasi (w, l, h)
        self.problem = problem
        self.trucks = problem.trucks
        self.packages = problem.packages
        self.truck_of: dict[str, int | None] = {
            package_id: None for package_id in self.packages
        }
        self.positions: dict[str, Position | None] = {
            package_id: None for package_id in self.packages
        }
        self.orientations: dict[str, Orientation] = {
            package_id: ("w", "l", "h") for package_id in self.packages
        }
        # Dicache biar gak dihitung ulang, diupdate sama place() dan rotate():
        # total value, berat per truk, kotak (x, y, z, dx, dy, dz) tiap paket di truk,
        # dan raw penalty urutan bongkar (cuma dihitung kalo objective "lifo")
        self.value = 0
        self.truck_weights = [0] * len(self.trucks)
        self.boxes: dict[str, Box] = {}
        self.penalty = 0

    @classmethod
    def random_initial(cls, problem: Problem, insert_probability: float = 0.5) -> State:
        # Truk kosong, orientasi diacak, trs tiap paket (urutan acak) dicoba masuk
        # ke truk dan koordinat valid acak dgn peluang insert_probability
        state = cls(problem)
        all_ids = list(state.packages)
        for package_id in all_ids:
            state.orientations[package_id] = random.choice(ALL_ORIENTATIONS)

        for package_id in random.sample(all_ids, len(all_ids)):
            if random.random() > insert_probability:
                continue
            truck_index = random.randrange(len(state.trucks))
            candidates = state.candidate_positions(package_id, truck_index)
            random.shuffle(candidates)
            for position in candidates:
                state.place(package_id, truck_index, position)
                if state.valid_after_change([package_id]):
                    break
                state.place(package_id, None, None)
        return state

    def copy(self) -> State:
        # Data persoalan gak berubah, jadi cukup copy tempat, posisi, sama orientasi tanpa lewat __init__
        new_state = State.__new__(State)
        new_state.problem = self.problem
        new_state.trucks = self.trucks
        new_state.packages = self.packages
        new_state.truck_of = dict(self.truck_of)
        new_state.positions = dict(self.positions)
        new_state.orientations = dict(self.orientations)
        new_state.value = self.value
        new_state.truck_weights = list(self.truck_weights)
        new_state.boxes = dict(self.boxes)
        new_state.penalty = self.penalty
        return new_state

    def place(
        self, package_id: str, truck_index: int | None, position: Position | None
    ) -> None:
        # Semua perubahan tempat paket lewat sini biar value, berat per truk, sama penalty tetap sinkron
        self.penalty -= self.package_penalty(package_id)
        package = self.packages[package_id]
        old_truck = self.truck_of[package_id]
        if old_truck is not None:
            self.value -= package.value
            self.truck_weights[old_truck] -= package.weight
        if truck_index is not None:
            self.value += package.value
            self.truck_weights[truck_index] += package.weight
        self.truck_of[package_id] = truck_index
        self.positions[package_id] = position
        self.update_box(package_id)
        self.penalty += self.package_penalty(package_id)

    def rotate(self, package_id: str, axis: str) -> None:
        self.penalty -= self.package_penalty(package_id)
        self.orientations[package_id] = rotate_90(self.orientations[package_id], axis)
        self.update_box(package_id)
        self.penalty += self.package_penalty(package_id)

    def update_box(self, package_id: str) -> None:
        position = self.positions[package_id]
        if self.truck_of[package_id] is None or position is None:
            self.boxes.pop(package_id, None)
            return
        x, y, z = position
        dx, dy, dz = self.packages[package_id].oriented_dimensions(
            self.orientations[package_id]
        )
        self.boxes[package_id] = (x, y, z, dx, dy, dz)

    def inside(self, truck_index: int | None = None) -> list[str]:
        # Paket di truk tertentu, atau di truk mana aja kalo truck_index = None
        result = []
        for package_id in self.packages:
            current_truck = self.truck_of[package_id]
            if current_truck is None:
                continue
            if truck_index is None or current_truck == truck_index:
                result.append(package_id)
        return result

    def outside(self) -> list[str]:
        return [
            package_id
            for package_id in self.packages
            if self.truck_of[package_id] is None
        ]

    def objective(self) -> float:
        if self.problem.penalty_scale == 0:
            return self.value
        return self.value - self.problem.penalty_scale * self.penalty

    def normalized_penalty(self) -> float:
        if self.problem.eta_range == 0:
            return 0.0
        return self.raw_penalty() / self.problem.eta_range

    def raw_penalty(self) -> int:
        # Cache cuma diupdate di objective lifo, selain itu dihitung langsung
        if self.problem.objective == "lifo":
            return self.penalty
        total = 0
        for truck_index, truck in enumerate(self.trucks):
            for id_1, id_2 in combinations(self.inside(truck_index), 2):
                total += self.pair_penalty(
                    self.box(id_1),
                    self.packages[id_1].eta,
                    self.box(id_2),
                    self.packages[id_2].eta,
                    truck.door_on_x,
                )
        return total

    @staticmethod
    def pair_penalty(
        box_1: Box, eta_1: int, box_2: Box, eta_2: int, door_on_x: bool
    ) -> int:
        # Pasangan melanggar kalo paket yg turun belakangan (ETA lebih besar) ngalangin paket yg turun duluan.
        # Nilainya selisih ETA, pelanggaran yg lebih parah dihukum lebih berat.
        if eta_1 < eta_2 and blocks(box_2, box_1, door_on_x):
            return eta_2 - eta_1
        if eta_2 < eta_1 and blocks(box_1, box_2, door_on_x):
            return eta_1 - eta_2
        return 0

    def package_penalty(self, package_id: str) -> int:
        # Total penalty semua pasangan yg ngelibatin package_id di truk yg sama
        if self.problem.objective != "lifo" or package_id not in self.boxes:
            return 0
        package_box = self.boxes[package_id]
        eta = self.packages[package_id].eta
        door_on_x = self.trucks[self.truck_index_of(package_id)].door_on_x
        total = 0
        for other_id, other_box in self.other_boxes_in_truck(package_id):
            total += self.pair_penalty(
                package_box, eta, other_box, self.packages[other_id].eta, door_on_x
            )
        return total

    def violations(self, truck_index: int) -> list[tuple[str, str]]:
        # Pasangan (paket yg dihalangi, paket penghalang) yg melanggar urutan bongkar di satu truk
        door_on_x = self.trucks[truck_index].door_on_x
        result = []
        for id_1, id_2 in combinations(self.inside(truck_index), 2):
            eta_1, eta_2 = self.packages[id_1].eta, self.packages[id_2].eta
            if (
                self.pair_penalty(
                    self.box(id_1), eta_1, self.box(id_2), eta_2, door_on_x
                )
                > 0
            ):
                result.append((id_1, id_2) if eta_1 < eta_2 else (id_2, id_1))
        return result

    def objective_text(self) -> str:
        if self.problem.objective == "value":
            return str(self.value)
        return f"{self.objective():.3f}"

    def objective_label(self) -> str:
        if self.problem.objective == "value":
            if self.problem.penalty_shown:
                return (
                    f"value = {self.value}, penalti = {self.normalized_penalty():.2f}"
                )
            return f"value = {self.value}"
        return f"objective = {self.objective_text()}, value = {self.value}, penalti = {self.normalized_penalty():.2f}"

    def total_weight(self, truck_index: int) -> int:
        return self.truck_weights[truck_index]

    def box(self, package_id: str) -> Box:
        return self.boxes[package_id]

    def truck_index_of(self, package_id: str) -> int:
        # Indeks truk utk paket yg pasti ada di dalam truk
        truck_index = self.truck_of[package_id]
        assert truck_index is not None
        return truck_index

    def other_boxes_in_truck(self, package_id: str) -> Iterator[tuple[str, Box]]:
        # Pasangan (id, kotak) paket lain di truk yg sama. Urutannya gak ngaruh ke pengecekan constraint
        truck_index = self.truck_of[package_id]
        for other_id, other_box in self.boxes.items():
            if other_id != package_id and self.truck_of[other_id] == truck_index:
                yield other_id, other_box

    def to_dict(self) -> dict[str, Any]:
        trucks_content = []
        for truck_index, truck in enumerate(self.trucks):
            truck_content = []
            for package_id in self.inside(truck_index):
                truck_content.append(
                    {
                        "id": package_id,
                        "position": list(self.box(package_id)[:3]),
                        "orientation": "".join(self.orientations[package_id]),
                        "size": list(self.box(package_id)[3:]),
                    }
                )
            truck_summary: dict[str, Any] = {
                "id": truck.id,
                "weight": self.total_weight(truck_index),
                "max_capacity": truck.max_capacity,
                "packages": truck_content,
            }
            if self.problem.objective == "lifo":
                truck_summary["violations"] = [
                    list(pair) for pair in self.violations(truck_index)
                ]
            trucks_content.append(truck_summary)
        summary: dict[str, Any] = {"value": self.value}
        if self.problem.objective == "lifo":
            summary.update(
                {"objective": self.objective(), "penalty": self.normalized_penalty()}
            )
        return {**summary, "trucks": trucks_content, "outside": self.outside()}

    # Constraint per paket. Semua method di bawah hanya untuk paket yg ada di dalam truk.

    def within_bounds(self, package_id: str) -> bool:
        # Paket gak nembus dinding peti kemas
        W, L, H = self.trucks[self.truck_index_of(package_id)].dimensions
        x, y, z, dx, dy, dz = self.box(package_id)
        return (
            0 <= x and 0 <= y and 0 <= z and x + dx <= W and y + dy <= L and z + dz <= H
        )

    def collides(self, package_id: str) -> bool:
        # Paket beririsan di x, y, dan z sekaligus sama paket lain di truk yg sama
        package_box = self.box(package_id)
        z, dz = package_box[2], package_box[5]
        for _, other_box in self.other_boxes_in_truck(package_id):
            overlap_z = z < other_box[2] + other_box[5] and other_box[2] < z + dz
            if overlap_z and bases_overlap(package_box, other_box):
                return True
        return False

    def is_supported(self, package_id: str) -> bool:
        # Paket gak melayang kalo z = 0, atau ada paket lain yg bagian atasnya pas
        # di bawah alas paket. Paket fragile gak boleh jadi support.
        package_box = self.box(package_id)
        z = package_box[2]
        if z == 0:
            return True

        supported = False
        for below_id, box_below in self.other_boxes_in_truck(package_id):
            if box_below[2] + box_below[5] == z and bases_overlap(
                package_box, box_below
            ):
                if self.packages[below_id].is_fragile:
                    return False
                supported = True
        return supported

    def resting_on(self, below_id: str) -> list[str]:
        # Paket lain di truk yg sama yg alasnya pas di atas paket below_id
        box_below = self.box(below_id)
        top_z = box_below[2] + box_below[5]
        result = []
        for other_id, other_box in self.other_boxes_in_truck(below_id):
            if other_box[2] == top_z and bases_overlap(other_box, box_below):
                result.append(other_id)
        return result

    def valid(self) -> bool:
        # Cek lengkap semua constraint
        for truck_index, truck in enumerate(self.trucks):
            if self.total_weight(truck_index) > truck.max_capacity:
                return False
        for package_id in self.inside():
            if (
                not self.within_bounds(package_id)
                or self.collides(package_id)
                or not self.is_supported(package_id)
            ):
                return False
        return True

    def valid_after_change(
        self, changed: Sequence[str], before: State | None = None
    ) -> bool:
        # Setara sama valid() kalo state sblm perubahan (before) valid, tapi cuma ngecek yg bisa
        # kena dampak: paket yg berubah, truk yg nerima paket, dan paket yg di state before numpu di
        # paket yg berubah. Constraint lain gak berubah jadi pasti tetap terpenuhi.
        # before boleh None kalo paket yg berubah sblmnya ada di luar truk.
        for package_id in changed:
            truck_index = self.truck_of[package_id]
            if truck_index is None:
                continue
            if self.total_weight(truck_index) > self.trucks[truck_index].max_capacity:
                return False
            if (
                not self.within_bounds(package_id)
                or self.collides(package_id)
                or not self.is_supported(package_id)
            ):
                return False
            if self.packages[package_id].is_fragile and self.resting_on(package_id):
                return False
        # Dicek terakhir, paling mahal dan kebanyakan move udah gugur duluan
        if before is not None:
            for changed_id in changed:
                if before.truck_of[changed_id] is None:
                    continue
                for package_id in before.resting_on(changed_id):
                    if package_id not in changed and not self.is_supported(package_id):
                        return False
        return True

    def z_levels(self, package_id: str, truck_index: int) -> list[int]:
        # z cuma di lantai (0) atau pas di atas paket lain, karena z lainnya pasti melayang
        H = self.trucks[truck_index].dimensions[2]
        dz = self.packages[package_id].oriented_dimensions(
            self.orientations[package_id]
        )[2]
        level = {0}
        for other_id, other_box in self.boxes.items():
            if other_id != package_id and self.truck_of[other_id] == truck_index:
                level.add(other_box[2] + other_box[5])
        return [z for z in sorted(level) if z + dz <= H]

    def candidate_positions(self, package_id: str, truck_index: int) -> list[Position]:
        W, L, _ = self.trucks[truck_index].dimensions
        dx, dy, _ = self.packages[package_id].oriented_dimensions(
            self.orientations[package_id]
        )
        candidates = []
        for z in self.z_levels(package_id, truck_index):
            for x in range(W - dx + 1):
                for y in range(L - dy + 1):
                    candidates.append((x, y, z))
        return candidates

    def random_candidate_position(
        self, package_id: str, truck_index: int
    ) -> Position | None:
        # Milih satu posisi acak dari candidate_positions selain posisi sekarang.
        W, L, _ = self.trucks[truck_index].dimensions
        dx, dy, _ = self.packages[package_id].oriented_dimensions(
            self.orientations[package_id]
        )
        count_x, count_y = W - dx + 1, L - dy + 1
        if count_x <= 0 or count_y <= 0:
            return None
        levels = self.z_levels(package_id, truck_index)
        per_level = count_x * count_y
        total = len(levels) * per_level

        # Posisi skrg pasti termasuk kandidat karena state valid, jadi dilewatin
        current_index = None
        if self.truck_of[package_id] == truck_index:
            x, y, z = self.box(package_id)[:3]
            current_index = levels.index(z) * per_level + x * count_y + y
            total -= 1
        if total <= 0:
            return None

        index = random.randrange(total)
        if current_index is not None and index >= current_index:
            index += 1
        level_index, remainder = divmod(index, per_level)
        x, y = divmod(remainder, count_y)
        return x, y, levels[level_index]

    def apply(self, move: Move) -> State | None:
        # Hasilin successor dari move, atau None kalo successor melanggar constraint
        successor = self.copy()
        changed: tuple[str, ...]

        # Cuma move "move" yg isinya empat elemen
        if len(move) == 4:
            _, package_id, truck_index, new_position = move
            changed = (package_id,)
            successor.place(package_id, truck_index, new_position)
        elif move[0] == "swap":
            _, package_id_1, package_id_2 = move
            changed = (package_id_1, package_id_2)
            successor.place(
                package_id_1, self.truck_of[package_id_2], self.positions[package_id_2]
            )
            successor.place(
                package_id_2, self.truck_of[package_id_1], self.positions[package_id_1]
            )
        else:
            _, package_id, axis = move
            changed = (package_id,)
            successor.rotate(package_id, axis)

        if successor.valid_after_change(changed, self):
            return successor
        return None

    def all_moves(self) -> list[Move]:
        moves: list[Move] = []
        # Tukar dua paket, minimal satu ada di dalam truk
        for package_id_1, package_id_2 in combinations(self.packages, 2):
            if (
                self.truck_of[package_id_1] is not None
                or self.truck_of[package_id_2] is not None
            ):
                moves.append(("swap", package_id_1, package_id_2))
        for package_id in self.packages:
            for truck_index in range(len(self.trucks)):
                for new_position in self.candidate_positions(package_id, truck_index):
                    same = (
                        truck_index == self.truck_of[package_id]
                        and new_position == self.positions[package_id]
                    )
                    if not same:
                        moves.append(("move", package_id, truck_index, new_position))
        for package_id in self.inside():
            for axis in "xyz":
                moves.append(("rotate", package_id, axis))
        return moves

    def all_successors(self) -> list[State]:
        successors = []
        for move in self.all_moves():
            successor = self.apply(move)
            if successor is not None:
                successors.append(successor)
        return successors

    def random_successor(self, max_attempts: int = 200) -> State:
        # Pilih jenis move sama parameternya secara acak sampe dapet successor yg valid
        all_ids = list(self.packages)
        for _ in range(max_attempts):
            kind = random.choice(["swap", "move", "rotate"])
            move: Move

            if kind == "swap":
                package_id_1, package_id_2 = random.sample(all_ids, 2)
                if (
                    self.truck_of[package_id_1] is None
                    and self.truck_of[package_id_2] is None
                ):
                    continue
                move = ("swap", package_id_1, package_id_2)

            elif kind == "move":
                package_id = random.choice(all_ids)
                truck_index = random.randrange(len(self.trucks))
                position = self.random_candidate_position(package_id, truck_index)
                if position is None:
                    continue
                move = ("move", package_id, truck_index, position)

            else:
                packages_inside = self.inside()
                if len(packages_inside) == 0:
                    continue
                move = ("rotate", random.choice(packages_inside), random.choice("xyz"))

            successor = self.apply(move)
            if successor is not None:
                return successor

        return self.copy()
