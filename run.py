from experiment_utils import run_all as part1_run_all
from nn_experiment import run_all as part2_run_all
from visualization import generate_all_plots

if __name__ == "__main__":
    print("Running Part 1")
    part1_run_all()

    print("Running Part 2")
    part2_run_all()

    print("Generating Plots")
    generate_all_plots()
