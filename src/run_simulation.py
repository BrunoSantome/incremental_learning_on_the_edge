# Launcher for the end-to-end simulation, which now lives in experiments/end_to_end.py
# (kept here so the entry point stays `python run_simulation.py` from src/).

from experiments.end_to_end import (
    run_edge_simulation_test,
    run_production_iteration_simulation_test,
    run_interactive_simulation,
)

if __name__ == "__main__":
    # run_edge_simulation_test()
    # run_production_iteration_simulation_test()
    run_interactive_simulation()
