from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, KeyEvent
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

    from core.problem import Problem
    from core.state import State
    from experiments.runner import ExperimentResult, PlotFn, RunEntry


DOOR_COLOR = "limegreen"
ETA_COLORMAP = "Blues"
ETA_SHADE_RANGE = (0.3, 0.85)


def eta_shade(problem: Problem, eta: int) -> float:
    low, high = ETA_SHADE_RANGE
    if problem.eta_range == 0:
        return (low + high) / 2
    eta_min = min(package.eta for package in problem.package_list)
    return low + (high - low) * (eta - eta_min) / problem.eta_range


def draw_truck(ax: Axes3D, state: State, truck_index: int, title: str) -> None:
    # Outline merah = paket fragile. Di objective lifo, warna paket makin tua kalo ETA-nya makin besar
    truck = state.trucks[truck_index]
    W, L, H = truck.dimensions
    colors = cast(Any, plt.cm.tab20).colors
    eta_colormap = plt.get_cmap(ETA_COLORMAP)
    for number, package_id in enumerate(state.packages):
        if state.truck_of[package_id] != truck_index:
            continue
        package = state.packages[package_id]
        x, y, z, dx, dy, dz = state.box(package_id)
        color, text_color = colors[number % 20], "black"
        if state.problem.objective == "lifo":
            shade = eta_shade(state.problem, package.eta)
            color, text_color = eta_colormap(shade), "white" if shade > 0.7 else "black"
        edge_color = "red" if package.is_fragile else "black"
        ax.bar3d(
            x,
            y,
            z,
            dx,
            dy,
            dz,
            color=color,
            alpha=0.7,
            edgecolor=edge_color,
            linewidth=1,
        )
        ax.text(
            x + dx / 2,
            y + dy / 2,
            z + dz / 2,
            package_id,
            fontsize=8,
            ha="center",
            color=text_color,
        )
    draw_door(ax, truck.dimensions, truck.door_on_x)

    ax.set_xlim(0, W)
    ax.set_ylim(0, L)
    ax.set_zlim(0, H)
    ax.set_box_aspect((W, L, H))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"{truck.id}: {title}\nberat = {state.total_weight(truck_index)}/{truck.max_capacity}",
        fontsize=10,
    )


def draw_door(ax: Axes3D, dimensions: tuple[int, int, int], door_on_x: bool) -> None:
    W, L, H = dimensions
    if door_on_x:
        corners = [(W, 0, 0), (W, L, 0), (W, L, H), (W, 0, H)]
    else:
        corners = [(0, 0, 0), (W, 0, 0), (W, 0, H), (0, 0, H)]
    ax.add_collection3d(
        Poly3DCollection(
            [corners], facecolor=DOOR_COLOR, alpha=0.25, edgecolor="green", linewidth=2
        )
    )


def add_package_legend(fig: Figure, problem: Problem, **position: Any) -> None:
    handles = [
        Patch(facecolor="white", edgecolor="red", label="Paket fragile"),
        Patch(facecolor="white", edgecolor="black", label="Paket biasa"),
        Patch(facecolor=DOOR_COLOR, alpha=0.5, edgecolor="green", label="Pintu"),
    ]
    if problem.objective == "lifo":
        eta_colormap = plt.get_cmap(ETA_COLORMAP)
        etas = [package.eta for package in problem.package_list]
        for eta in (min(etas), max(etas)):
            handles.append(
                Patch(
                    facecolor=eta_colormap(eta_shade(problem, eta)),
                    alpha=0.7,
                    edgecolor="black",
                    label=f"ETA {eta}",
                )
            )
    fig.legend(handles=handles, fontsize=9, **position)


def draw_initial_final(initial_state: State, final_state: State, title: str) -> Figure:
    # Kolom kiri state awal, kolom kanan state akhir, satu baris per truk
    num_trucks = len(initial_state.trucks)
    fig = plt.figure(figsize=(11, 4.5 * num_trucks))
    for truck_index in range(num_trucks):
        for column, (name, state) in enumerate(
            [("State awal", initial_state), ("State akhir", final_state)]
        ):
            ax = cast(
                "Axes3D",
                fig.add_subplot(
                    num_trucks, 2, truck_index * 2 + column + 1, projection="3d"
                ),
            )
            draw_truck(ax, state, truck_index, f"{name}\n{state.objective_label()}")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05 / num_trucks, 1, 1))
    add_package_legend(fig, initial_state.problem, loc="lower center", ncol=5)
    return fig


# Fungsi draw_* di bawah nenerima `selected`, yaitu index run yg disorot, atau None untuk semua run


def draw_objective(
    ax: Axes,
    histories: list[list[float]],
    title: str,
    selected: int | None = None,
    xlabel: str = "Iterasi",
) -> None:
    for index, history in enumerate(histories):
        if selected is None or index == selected:
            ax.plot(history, color=f"C{index % 10}", label=f"Run {index + 1}")
        else:
            ax.plot(history, color="lightgray", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Objective function")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def draw_sa_probability(
    ax: Axes, histories_prob: list[list[float]], selected: int | None = None
) -> None:
    # Cuma untuk neighbor yg lebih jelek (ΔE < 0)
    for index, history_prob in enumerate(histories_prob):
        if selected is None or index == selected:
            ax.scatter(
                range(len(history_prob)),
                history_prob,
                s=4,
                color=f"C{index % 10}",
                label=f"Run {index + 1}",
            )
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("e^(ΔE/T)")
    ax.set_title("Peluang menerima neighbor yang lebih buruk", fontsize=10)
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.3)


def draw_ga(
    ax: Axes, runs: list[RunEntry], title: str, selected: int | None = None
) -> None:
    # Biru untuk nilai max, orange untuk rata-rata populasi. Antar-run dibedain dari gradasinya.
    for order, (global_index, run_number, result) in enumerate(runs):
        if selected is None or global_index == selected:
            shade = (
                0.75
                if selected is not None
                else 0.9 - 0.45 * order / max(1, len(runs) - 1)
            )
            ax.plot(
                result.history,
                color=plt.cm.Blues(shade),
                label=f"Run {run_number} maksimum",
            )
            ax.plot(
                result.info["history_avg"],
                color=plt.cm.Oranges(shade),
                label=f"Run {run_number} rata-rata",
            )
        else:
            ax.plot(result.history, color=plt.cm.Blues(0.25), linewidth=1)
            ax.plot(result.info["history_avg"], color=plt.cm.Oranges(0.25), linewidth=1)
    ax.set_xlabel("Iterasi (generasi)")
    ax.set_ylabel("Objective function")
    ax.set_title(title, fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)


def draw_summary(plots: list[PlotFn], title: str) -> Figure:
    fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 4), squeeze=False)
    for ax, plot_fn in zip(axes[0], plots):
        plot_fn(ax, None)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


class Viewer:
    """Satu window buat semua hasil, state awal/akhir run terpilih di atas, plot di bawah."""

    def __init__(self, results: list[ExperimentResult]) -> None:
        self.results = results
        self.result_index = 0
        self.run_index = 0
        self.axes_plot: list[Axes] = []
        self.buttons: list[Button] = []
        self.fig = plt.figure(figsize=(15, 9))
        add_package_legend(
            self.fig, results[0].runs[0].initial_state.problem, loc="upper right"
        )

        # Semua truk digambar dlm satu baris, state awal sama akhir tiap truk sebelahan
        self.num_trucks = len(results[0].runs[0].initial_state.trucks)
        outer_grid = self.fig.add_gridspec(
            2,
            1,
            height_ratios=[1.4, 1],
            left=0.05,
            right=0.98,
            top=0.88,
            bottom=0.13,
            hspace=0.3,
        )
        upper_grid = outer_grid[0].subgridspec(1, 2 * self.num_trucks)
        self.plot_area = outer_grid[1]
        self.truck_axes = [
            cast("Axes3D", self.fig.add_subplot(upper_grid[0, column], projection="3d"))
            for column in range(2 * self.num_trucks)
        ]

        button_positions = [("◀ Run", -1, 0, 0.36), ("Run ▶", 1, 0, 0.51)]
        if len(results) > 1:
            button_positions += [
                ("◀ Eksperimen", 0, -1, 0.21),
                ("Eksperimen ▶", 0, 1, 0.66),
            ]
        for label, delta_run, delta_result, x in button_positions:

            def on_click(
                _: Event, delta_run: int = delta_run, delta_result: int = delta_result
            ) -> None:
                self.shift(delta_run, delta_result)

            button = Button(self.fig.add_axes((x, 0.015, 0.13, 0.04)), label)
            button.on_clicked(on_click)
            self.buttons.append(button)

        hint = "← → pindah run"
        if len(results) > 1:
            hint += ", ↑ ↓ pindah eksperimen"
        self.fig.text(
            0.5,
            0.068,
            f"{hint}. Tutup window untuk kembali ke menu.",
            ha="center",
            fontsize=9,
            color="gray",
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.redraw()

    def shift(self, delta_run: int, delta_result: int) -> None:
        if delta_result != 0:
            self.result_index = (self.result_index + delta_result) % len(self.results)
            self.run_index = 0
        else:
            num_runs = len(self.results[self.result_index].runs)
            self.run_index = (self.run_index + delta_run) % num_runs
        self.redraw()
        self.fig.canvas.draw_idle()

    def on_key_press(self, event: Event) -> None:
        direction = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
        if isinstance(event, KeyEvent) and event.key in direction:
            self.shift(*direction[event.key])

    def redraw(self) -> None:
        result = self.results[self.result_index]
        run = result.runs[self.run_index]

        for truck_index in range(self.num_trucks):
            for column, (name, state) in enumerate(
                [("State awal", run.initial_state), ("State akhir", run.final_state)]
            ):
                ax = self.truck_axes[truck_index * 2 + column]
                ax.clear()
                # Objective udah ada di suptitle, judul subplot dipendekin biar gak tumpang tindih antar truk
                draw_truck(ax, state, truck_index, name)

        for ax_plot in self.axes_plot:
            ax_plot.remove()
        lower_grid = self.plot_area.subgridspec(1, len(result.plot), wspace=0.25)
        self.axes_plot = []
        for column, plot_fn in enumerate(result.plot):
            ax_plot = self.fig.add_subplot(lower_grid[0, column])
            plot_fn(ax_plot, self.run_index)
            self.axes_plot.append(ax_plot)

        title = result.title
        if len(self.results) > 1:
            title = f"[{self.result_index + 1}/{len(self.results)}] {title}"
        metrics = ", ".join(f"{name} = {value}" for name, value in run.metrics.items())
        initial, final = run.initial_state, run.final_state
        change = f"{objective_name(final)} {initial.objective_text()} → {final.objective_text()}"
        penalty = f"penalti {initial.normalized_penalty():.2f} → {final.normalized_penalty():.2f}"
        if final.problem.objective == "lifo":
            change += f" (value {initial.value} → {final.value}, {penalty})"
        elif final.problem.penalty_shown:
            change += f" ({penalty})"
        self.fig.suptitle(
            f"{title}\n{run.label} ({self.run_index + 1}/{len(result.runs)}): {change}, {metrics}"
        )


def objective_name(state: State) -> str:
    return "value" if state.problem.objective == "value" else "objective"


def show_results(results: list[ExperimentResult]) -> bool:
    # Return False kalo windownya gak bisa di-display (non-interactive)
    # Remap arrow left/right buat navigasi
    for key in ("keymap.back", "keymap.forward"):
        plt.rcParams[key] = [
            binding for binding in plt.rcParams[key] if binding not in ("left", "right")
        ]

    viewer = Viewer(results)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        plt.show()
    plt.close(viewer.fig)
    return not any(
        "non-interactive" in str(caught.message) for caught in caught_warnings
    )
