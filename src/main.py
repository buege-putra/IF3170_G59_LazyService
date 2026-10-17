import argparse
import sys
from pathlib import Path

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    sys.exit("matplotlib belum terinstal. Instal terlebih dahulu dengan:\n"
             "  pip install -r requirements.txt")

from core.loader import load_input
from experiments.parameter import Parameter
from experiments.runner import Experiment
from tui.menu import run_tui

OUTPUT_FOLDER = Path(__file__).resolve().parent.parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local search untuk pengisian paket ke truk peti kemas")
    parser.add_argument("input", help="path file JSON berisi data truk dan paket")
    parser.add_argument("--seed", type=int, default=67, help="random seed (default: 67)")
    args = parser.parse_args()

    try:
        problem = load_input(args.input)
    except (OSError, ValueError) as error:
        parser.error(f"gagal membaca {args.input}: {error}")

    experiment = Experiment(problem, Parameter(), OUTPUT_FOLDER / Path(args.input).stem)
    run_tui(experiment, args.input, args.seed)


if __name__ == "__main__":
    main()
