# Launcher for the end-to-end simulation, which now lives in experiments/end_to_end.py
# (kept here so the entry point stays `python run_simulation.py` from src/).

import sys

from experiments.end_to_end import (
    run_edge_simulation_test,
    run_production_iteration_simulation_test,
    run_interactive_simulation,
    run_scripted_simulation,
)

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    quiet = (
        "--verbose" not in sys.argv
    )  # demo output unless the full logs are asked for

    # run_edge_simulation_test()
    # run_production_iteration_simulation_test()
    if args:
        run_scripted_simulation(args[0], quiet=quiet)
    else:
        run_interactive_simulation(quiet=quiet)
