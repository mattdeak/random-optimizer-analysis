import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "output"


def plot_ga_gsearch(input_path, output_filename, title):

    df = pd.read_csv(input_path, index_col=0)
    ax = df.plot(x="mutation_prob", y="Fitness Evaluations")
    
    
    df["Found Global Optimum"] = df["Best Fitness"] == df["Global Optimum"]

    found_global_optima = df[df["Found Global Optimum"] == True]
    did_not_find_global_optima = df[df["Found Global Optimum"] == False]

    ax.scatter(
        found_global_optima["mutation_prob"],
        found_global_optima["Fitness Evaluations"],
        marker="o",
        c="green",
    )

    ax.scatter(
        did_not_find_global_optima["mutation_prob"],
        did_not_find_global_optima["Fitness Evaluations"],
        marker="o",
        c="red",
    )

    ax.set_xlabel("Mutation Probability")
    ax.set_ylabel("# of Evaluations")
    ax.get_legend().remove()
    ax.set_title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()


def plot_mimic_gsearch(input_path, output_filename, title):
    df = pd.read_csv(input_path, index_col=0)
    ax = df.plot(x="keep_pct", y="Fitness Evaluations")
    df["Found Global Optimum"] = df["Best Fitness"] == df["Global Optimum"]

    found_global_optima = df[df["Found Global Optimum"] == True]
    did_not_find_global_optima = df[df["Found Global Optimum"] == False]

    ax.scatter(
        found_global_optima["keep_pct"],
        found_global_optima["Fitness Evaluations"],
        marker="o",
        c="green",
    )

    ax.scatter(
        did_not_find_global_optima["keep_pct"],
        did_not_find_global_optima["Fitness Evaluations"],
        marker="o",
        c="red",
    )

    ax.set_xlabel("Keep %")
    ax.set_ylabel("# of Evaluations")
    ax.get_legend().remove()
    ax.set_title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()


def plot_sa_gsearch(input_path, output_filename, title):

    df = pd.read_csv(input_path, index_col=0)
    ax = df.plot(x="schedule", y="Fitness Evaluations")

    df["Found Global Optimum"] = df["Best Fitness"] == df["Global Optimum"]

    found_global_optima = df[df["Found Global Optimum"] == True]
    did_not_find_global_optima = df[df["Found Global Optimum"] == False]

    ax.scatter(
        found_global_optima["schedule"],
        found_global_optima["Fitness Evaluations"],
        marker="o",
        c="green",
    )

    ax.scatter(
        did_not_find_global_optima["schedule"],
        did_not_find_global_optima["Fitness Evaluations"],
        marker="o",
        c="red",
    )

    ax.set_xlabel("Decay Exponent")
    ax.set_ylabel("Fitness")
    ax.get_legend().remove()
    ax.set_title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()


def plot_all_gridsearches(problem_name):
    name_keymap = {
        "ga": "GA",
        "sa": "SA",
        "mimic": "MIMIC",
        "flipflop": "FlipFlop",
        "knapsack": "Knapsack",
        "fourpeaks": "FourPeaks",
    }

    function_map = {
        "ga": plot_ga_gsearch,
        "sa": plot_sa_gsearch,
        "mimic": plot_mimic_gsearch,
    }
    algos = ["ga", "sa", "mimic"]

    for algo in algos:
        infile_path = os.path.join(OUTPUT_DIR, f"{algo}_{problem_name}_gsresults.csv")
        outfile_name = f"{name_keymap[problem_name]}_{name_keymap[algo]}_GridSearch.png"
        title = (
            f"Grid Search Results of {name_keymap[algo]} on {name_keymap[problem_name]}"
        )
        function_map[algo](infile_path, outfile_name, title)


plot_all_gridsearches("knapsack")
# plot_sa_gsearch('output/sa_knapsack_gsresults.csv','Test.png','Test')
# df = pd.read_csv('output/ga_knapsack_gsresults.csv', index_col=0)
