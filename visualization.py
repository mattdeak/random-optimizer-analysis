import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import MultipleLocator, AutoLocator
import numpy as np

OUTPUT_DIR = "output"


def calculate_marker_sizes(df, modifier=6):
    return 3 ** (df["Best Fitness"] / df["Global Optimum"] * modifier).astype(int)


def plot_ga_gsearch(input_path, output_filename, title):

    df = pd.read_csv(input_path, index_col=0)
    ax = df.plot(x="mutation_prob", y="Fitness Evaluations", zorder=0)

    df["Found Global Optimum"] = df["Best Fitness"] == df["Global Optimum"]

    found_global_optima = df[df["Found Global Optimum"] == True]
    did_not_find_global_optima = df[df["Found Global Optimum"] == False]

    recs = []
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc="g"))
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc="r"))
    ax.scatter(
        found_global_optima["mutation_prob"],
        found_global_optima["Fitness Evaluations"],
        marker="o",
        c="green",
        s=calculate_marker_sizes(found_global_optima),
        alpha=0.8,
        zorder=5,
    )

    ax.scatter(
        did_not_find_global_optima["mutation_prob"],
        did_not_find_global_optima["Fitness Evaluations"],
        marker="o",
        c="red",
        s=calculate_marker_sizes(did_not_find_global_optima),
        alpha=0.8,
        zorder=5,
    )

    ax.set_xlabel("Mutation Probability")
    ax.set_ylabel("# of Evaluations")
    ax.get_legend().remove()
    ax.legend(recs, ["Found Global Optimum", "Failed to Find Global Optimum"])
    ax.set_title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()


def plot_mimic_gsearch(input_path, output_filename, title):
    df = pd.read_csv(input_path, index_col=0)
    ax = df.plot(x="keep_pct", y="Fitness Evaluations", zorder=0)
    df["Found Global Optimum"] = df["Best Fitness"] == df["Global Optimum"]

    found_global_optima = df[df["Found Global Optimum"] == True]
    did_not_find_global_optima = df[df["Found Global Optimum"] == False]

    recs = []
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc="g"))
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc="r"))
    ax.scatter(
        found_global_optima["keep_pct"],
        found_global_optima["Fitness Evaluations"],
        marker="o",
        c="green",
        s=calculate_marker_sizes(found_global_optima),
        alpha=0.8,
        zorder=5,
    )

    ax.scatter(
        did_not_find_global_optima["keep_pct"],
        did_not_find_global_optima["Fitness Evaluations"],
        marker="o",
        c="red",
        s=calculate_marker_sizes(did_not_find_global_optima),
        alpha=0.8,
        zorder=5,
    )

    ax.set_xlabel("Keep %")
    ax.set_ylabel("# of Evaluations")
    ax.get_legend().remove()
    ax.legend(recs, ["Found Global Optimum", "Failed to Find Global Optimum"])
    ax.set_title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()


def plot_sa_gsearch(input_path, output_filename, title):

    df = pd.read_csv(input_path, index_col=0)
    ax = df.plot(x="schedule", y="Fitness Evaluations", zorder=0)

    df["Found Global Optimum"] = df["Best Fitness"] == df["Global Optimum"]

    found_global_optima = df[df["Found Global Optimum"] == True]
    did_not_find_global_optima = df[df["Found Global Optimum"] == False]

    recs = []
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc="g"))
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc="r"))
    ax.scatter(
        found_global_optima["schedule"],
        found_global_optima["Fitness Evaluations"],
        marker="o",
        c="green",
        s=calculate_marker_sizes(found_global_optima),
        alpha=0.8,
        zorder=5,
    )

    ax.scatter(
        did_not_find_global_optima["schedule"],
        did_not_find_global_optima["Fitness Evaluations"],
        marker="o",
        c="red",
        s=calculate_marker_sizes(did_not_find_global_optima),
        alpha=0.8,
        zorder=5,
    )

    ax.set_xlabel("Decay Exponent")
    ax.set_ylabel("Fitness")
    ax.get_legend().remove()
    ax.legend(recs, ["Found Global Optimum", "Failed to Find Global Optimum"])
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
        "tsp": "Travelling Salesman",
        "queens": "N-Queens",
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


def get_group_stats(df):
    df["Found Global Optimum"] = df["Best Fitness"] == df["Global Optimum"]
    df["Fitness Ratio"] = df["Best Fitness"] / df["Global Optimum"]

    grouped = df.groupby("Parameter Value")

    std_evals = grouped["Fitness Evaluations"].std()
    mean_evals = grouped["Fitness Evaluations"].mean()
    std_time = grouped["Time Elapsed"].std()
    mean_time = grouped["Time Elapsed"].mean()

    mean_fitness_ratio = grouped["Fitness Ratio"].mean()
    std_fitness_ratio = grouped["Fitness Ratio"].std()

    mean_iterations = grouped["Iterations"].mean()
    std_iterations = grouped["Iterations"].std()

    global_optimum_percentage = (
        grouped["Found Global Optimum"].sum() / grouped["Found Global Optimum"].count()
    )

    results = {
        "Mean Evaluations": mean_evals,
        "STD Evaluations": std_evals,
        "Mean Time": mean_time,
        "STD Time": std_time,
        "Success Rate": global_optimum_percentage,
        "Mean Fitness Ratio": mean_fitness_ratio,
        "STD Fitness Ratio": std_fitness_ratio,
        "Mean Iterations": mean_iterations,
        "STD Iterations": std_iterations,
    }
    return results


def plot_comparison(
    df_rhc,
    df_sa,
    df_ga,
    df_mimic,
    x_name,
    y_name,
    title,
    filename,
    y_name_alternate=None,
    error_name=None,
):

    mimic_results = get_group_stats(df_mimic)
    ga_results = get_group_stats(df_ga)
    sa_results = get_group_stats(df_sa)
    rhc_results = get_group_stats(df_rhc)

    print(mimic_results.keys())
    mimic_data = mimic_results[y_name]
    rhc_data = rhc_results[y_name]
    sa_data = sa_results[y_name]
    ga_data = ga_results[y_name]

    if error_name:
        mimic_error = mimic_results[error_name]
        rhc_error = rhc_results[error_name]
        ga_error = ga_results[error_name]
        sa_error = sa_results[error_name]
        errors = [mimic_error, ga_error, sa_error, rhc_error]

    else:
        errors = [None] * 4  # Probably not great design but whatever

    xs = rhc_data.index.values

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    colors = ["g", "b", "y", "m"]
    data = [mimic_data, ga_data, sa_data, rhc_data]
    labels = [
        "Mimic",
        "Genetic Algorithms",
        "Simulated Annealling",
        "Random Hill Climb",
    ]

    for d, c, l, e in zip(data, colors, labels, errors):
        ax.plot(xs, d.values, c=c, label=l, marker="o")

        if error_name:
            ax.fill_between(
                xs, d.values - e.values, d.values + e.values, color=c, alpha=0.5
            )

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_name)

    if y_name_alternate:
        ax.set_ylabel(y_name_alternate)
    else:
        ax.set_ylabel(y_name)

    ax.xaxis.set_major_locator(AutoLocator())
    ax.set_ylim(bottom=0)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_axisbelow(True)
    ax.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def show_success_rate(reliability_df):
    reliability_df["Success"] = (
        reliability_df["Best Fitness"] == reliability_df["Global Optimum"]
    )
    grouped = reliability_df.groupby("Parameter Value")

    success_rates = grouped["Success"].sum() / grouped["Success"].count()
    print(success_rates)


def get_reliability_df(problem_name, algorithm_name):
    filepath = os.path.join(
        OUTPUT_DIR, f"{algorithm_name}_{problem_name}_reliability.csv"
    )
    return pd.read_csv(filepath, index_col=0)


def plot_all_comparisons(problem_name, x_name):
    metrics = ["Fitness Ratio", "Time Elapsed", "Fitness Evaluations", "Iterations"]

    algos = ["rhc", "sa", "ga", "mimic"]

    df_rhc = get_reliability_df(problem_name, "rhc")
    df_sa = get_reliability_df(problem_name, "sa")
    df_ga = get_reliability_df(problem_name, "ga")
    df_mimic = get_reliability_df(problem_name, "mimic")

    X_names = [x_name] * 4
    y_names = [
        "Mean Fitness Ratio",
        "Mean Time",
        "Mean Evaluations",
        "Mean Iterations",
    ]
    y_labels = [
        "Fitness as % of Global Optimum",
        "Time Elapsed (s)",
        "Fitness Evaluations",
        "Iterations",
    ]

    titles = [
        "Comparison of Fitness Scores",
        "Comparison of Wall Clock Time",
        "Comparison in # of Evaluations",
        "Comparison in # of Iterations",
    ]
    error_names = [
        "STD Fitness Ratio",
        "STD Time",
        "STD Evaluations",
        "STD Iterations",
    ]

    for x_name, y_name, title, y_alternate, error_name in zip(
        X_names, y_names, titles, y_labels, error_names
    ):
        filename = f"{problem_name}_{title}.png"
        plot_comparison(
            df_rhc,
            df_sa,
            df_ga,
            df_mimic,
            x_name,
            y_name,
            title,
            filename,
            y_name_alternate=y_alternate,
            error_name=error_name,
        )


def plot_nn_lc(algorithm_name, title):

    filename = os.path.join(OUTPUT_DIR, f"{algorithm_name}_nn_curves.npy")
    arr = np.load(filename, allow_pickle=True)



    mean = arr.mean(axis=1)

    
    last_value = mean[-1]
    end_ix = np.argwhere(mean != last_value)[-1][0] + 1

    mean = mean[:end_ix]

    std = arr[:end_ix, :].std(axis=1)
    xs = np.arange(end_ix)


    fig, ax = plt.subplots(1, 1)

    ax.plot(xs, mean)
    ax.fill_between(xs, mean - std, mean + std, alpha=0.5)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Log Loss")
    ax.set_ylim(bottom=0)

    # Find index at which improvement stops.
    
    
    ax.set_xlim((0, end_ix-1))
    ax.set_title(title)
    ax.grid()

    # Find minimal loss
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'{algorithm_name}_nn_curve.png'))


def plot_nn_gsearch(algorithm_name):
    filename = os.path.join(OUTPUT_DIR, f"{algorithm_name}_nn_gsresults.csv")
    METRICS = ["Train_Acc", "Train_F1", "Val_Acc", "Val_F1", "Train_Time", "Iterations"]

    df = pd.read_csv(filename, index_col=0)
    params = [col for col in df.columns if col not in METRICS + ["Fold"]]
    print(params)

    mean_scores = df.groupby(params).mean().reset_index()
    val_pivotted = mean_scores.pivot(
        index=params[0], columns=params[1], values="Val_F1"
    )
    train_pivotted = mean_scores.pivot(
        index=params[0], columns=params[1], values="Train_F1"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1 = axes[0]
    ax2 = axes[1]

    sns.heatmap(train_pivotted, vmin=0, vmax=1, cmap="Blues", annot=True, ax=ax1)
    sns.heatmap(val_pivotted, vmin=0, vmax=1, cmap="Blues", annot=True, ax=ax2)

    ax1.set_title(f"{algorithm_name.upper()}: Train F1 Score")
    ax2.set_title(f"{algorithm_name.upper()}: Val F1 Score")

    outfile_name = os.path.join(OUTPUT_DIR, f'{algorithm_name}_nn_gridsearch.png')

    plt.savefig(outfile_name)
    plt.close()


def generate_nn_comparison_table(output_dir):
    algos = ["gs", "ga", "rhc", "sa"]
    srs = []
    for algo in algos:
        filename = os.path.join(output_dir, f"{algo}_nn_results.csv")
        algo_df = pd.read_csv(filename, index_col=0).T
        serie = pd.Series(algo_df.values[0], index=algo_df.columns)

        srs.append(serie)

    df = pd.DataFrame({k: v for k, v in zip(algos, srs)})
    return df


def plot_nn_gridsearches():
    algos = ['ga','rhc','sa']
    for algo in algos:
        plot_nn_gsearch(algo)




# plot_nn_lc("ga")
# plot_comparison(df_rhc, df_sa, df_ga, df_mimic, "Number of Items in Knapsack", "Mean Fitness Ratio", "Fitness Ratio Comparison", y_name_alternate="Fitness Ratio", error_name= "STD Fitness Ratio")
# df_mimic = pd.read_csv("output/mimic_queens_reliability.csv", index_col=0)
# df_ga = pd.read_csv("output/ga_queens_reliability.csv", index_col=0)
# df_sa = pd.read_csv("output/sa_queens_reliability.csv", index_col=0)
# df_rhc = pd.read_csv("output/rhc_queens_reliability.csv", index_col=0)
# plot_comparison(df_rhc, df_sa, df_ga, df_mimic, "Number of Queens", "Mean Fitness Ratio", "Fitnes Ratio", y_name_alternate="Fitness Ratio", error_name="STD Fitness Ratio")
# plot_comparison(df_rhc, df_sa, df_ga, df_mimic, "Number of Queens", "Mean Fitness Ratio", "Fitnes Ratio", y_name_alternate="Fitness Ratio", error_name="STD Fitness Ratio")


# plot_nn_gridsearches()
# plot_all_gridsearches("fourpeaks")
# plot_all_comparisons("tsp", "Cities")
# plot_all_gridsearches("knapsack")
# plot_all_gridsearches("flipflop")
# plot_all_gridsearches("fourpeaks")
# plot_sa_gsearch('output/sa_knapsack_gsresults.csv','Test.png','Test')
# df = pd.read_csv('output/mimic_knapsack_gsresults.csv', index_col=0)
# plot_all_comparisons("knapsack", "Number of Items")
# plot_all_comparisons("fourpeaks", "Size of Problem Space (bits)")


# plot_nn_gridsearches()

plot_nn_lc('ga', title='Genetic Algorithm Learning Curve')
plot_nn_lc('gs', title='Gradient Descent Learning Curve')
plot_nn_lc('rhc', title='RHC Learning Curve')
plot_nn_lc('sa', title='SA Learning Curve')
