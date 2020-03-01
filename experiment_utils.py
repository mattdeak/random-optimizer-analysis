import os
import warnings

from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from multiprocessing import pool
from heldkarp import held_karp
from fitnesses import QueensCustom
from scipy.special import comb
from sklearn.neural_network import MLPClassifier

import mlrose
from mlrose import ExpDecay
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import json
from tqdm import tqdm

MAX_ITERS = 5000
MAX_ATTEMPTS = 10
EVALUATION_COUNT = 0
METRICS = [
    "Time Elapsed",
    "Best Fitness",
    "Fitness Evaluations",
    "Global Optimum",
    "Found Global Optimum",
]
OUTPUT_DIR = "output"
MAX_RESTARTS = 10

GRID = {
    "rhc": {"restarts": [10]},
    "sa": {
        "schedule": [
            ExpDecay(exp_const=0.001),
            ExpDecay(exp_const=0.005),
            ExpDecay(exp_const=0.01),
            ExpDecay(exp_const=0.05),
            ExpDecay(exp_const=0.1),
        ],
    },
    "ga": {"pop_size": [200], "mutation_prob": [0.05, 0.1, 0.2, 0.4]},
    "mimic": {"pop_size": [200], "keep_pct": [0.1, 0.3, 0.5, 0.7, 0.9]},
}

# Utility functions here are used to instantiate the problems we aim to solve
def count_evaluations(fitness_fn, *args, **kwargs):
    def wrapper(state):
        fitness = fitness_fn(*args, **kwargs)
        global EVALUATION_COUNT
        EVALUATION_COUNT += 1
        return fitness.evaluate(state)

    return wrapper


def calculate_knapsack_global_optimum(values, weights, n_items, capacity):
    """The dynamic programming solution for the 0-1 knapsack problem as given in https://en.wikipedia.org/wiki/Knapsack_problem#0-1_knapsack_problem"""
    m = np.zeros((n_items, capacity + 1))
    for j in range(capacity):
        pass

    for i in range(0, n_items):
        for j in range(capacity + 1):
            if weights[i] > j:
                m[i, j] = m[i - 1, j]
            else:
                m[i, j] = max(m[i - 1, j], m[i - 1, j - weights[i]] + values[i])

    return m.max()  # The maximum value


def generate_distances(coords):
    N = len(coords)
    dists = np.zeros((N, N), dtype=np.float64)
    for i in range(N - 1):
        for j in range(i + 1, N):
            node1 = coords[i]
            node2 = coords[j]
            dist = np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
            dists[i, j] = dists[j, i] = dist

    return dists


def instantiate_problem_factories():
    def flipflop_factory(length=30):
        fitness = count_evaluations(mlrose.FlipFlop)
        fitness_final = mlrose.CustomFitness(fitness)
        flipflop = mlrose.DiscreteOpt(length, fitness_final)

        global_optimum = length - 1
        return flipflop, global_optimum

    def fourpeaks_factory(length=50, t_pct=0.1):
        fourpeaks_fitness = count_evaluations(mlrose.FourPeaks, t_pct=t_pct)
        fourpeaks_fitness_final = mlrose.CustomFitness(fourpeaks_fitness)
        fourpeaks = mlrose.DiscreteOpt(
            length=length, fitness_fn=fourpeaks_fitness_final
        )

        T = int(t_pct * length)
        global_optimum = 2 * length - T - 1
        return fourpeaks, global_optimum

    def knapsack_factory(N_items=20, max_weight=20, max_value=20, max_weight_pct=0.6):
        weights = np.random.randint(1, high=max_weight, size=N_items)
        values = np.random.randint(1, high=max_value, size=N_items)

        capacity = int(np.ceil(max_weight_pct * weights.sum()))

        global_optimum = calculate_knapsack_global_optimum(
            values, weights, N_items, capacity
        )

        knapsack_fitness = count_evaluations(
            mlrose.Knapsack, weights, values, max_weight_pct
        )

        knapsack_fitness_final = mlrose.CustomFitness(knapsack_fitness)
        knapsack = mlrose.DiscreteOpt(length=N_items, fitness_fn=knapsack_fitness_final)

        return knapsack, global_optimum

    def tsp_factory(length=10, min_nodeval=1, max_nodeval=20):

        coords = []
        for i in range(length):

            node = (
                np.random.randint(min_nodeval, max_nodeval),
                np.random.randint(min_nodeval, max_nodeval),
            )
            while node in coords:
                node = (
                    np.random.randint(min_nodeval, max_nodeval),
                    np.random.randint(min_nodeval, max_nodeval),
                )

            coords.append(node)

        fitness = count_evaluations(mlrose.TravellingSales, coords)
        fitness_final = mlrose.CustomFitness(fitness)
        fitness_final.get_prob_type = (
            lambda: "tsp"
        )  # Just a hack to make it work with a custom function.

        problem = mlrose.TSPOpt(length=length, fitness_fn=fitness_final, maximize=False)
        dists = generate_distances(coords)
        optimum, _ = held_karp(dists)

        return problem, optimum

    def queens_factory(length=8):
        fitness = count_evaluations(QueensCustom)
        fitness_final = mlrose.CustomFitness(fitness)
        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness_final, max_val=length)
        global_optimum = int(comb(length, 2))  # I think?

        return problem, global_optimum

    return {
        "fourpeaks": fourpeaks_factory,
        "knapsack": knapsack_factory,
        "flipflop": flipflop_factory,
        "tsp": tsp_factory,
        "queens": queens_factory,
    }


PROBLEM_MAP = instantiate_problem_factories()
ALGO_KEYMAP = {
    "rhc": mlrose.random_hill_climb,
    "sa": mlrose.simulated_annealing,
    "ga": mlrose.genetic_alg,
    "mimic": mlrose.mimic,
}


UNIVERSAL_PARAMETERS = {
    "curve": True,
    "max_iters": MAX_ITERS,
    "max_attempts": MAX_ATTEMPTS,
}

PROBLEM_GRID = {
    "knapsack": {"N_items": [20, 40, 60, 80, 100]},
    "fourpeaks": {"length": [50, 60, 70, 80, 90]},
    "flipflop": {"length": [20, 40, 60, 80, 100]},
    "tsp": {"length": [10, 12, 14, 16, 18]},
    "queens": {"length": [10, 20, 30, 40, 50]},
}


def combine_dicts(dict1, dict2):
    "Returns a dictionary with keys and values combined"
    combined = dict1.copy()
    combined.update(dict2)
    return combined


def parse_parameter_value(value):
    """Since hyperparameters can have multiple types, this is a simple utility function that gets a reasonable representation of that parameter for data collection"

    Parameters
    ----------

    value :

    Returns
    -------
    """
    if isinstance(value, ExpDecay):
        return value.exp_const
    elif isinstance(value, timedelta):
        return value.total_seconds()
    else:
        return value


def generate_gsearch_filename(problem_name, algo_name, output_dir):
    return f"{os.path.join(output_dir, algo_name)}_{problem_name}_gsresults.csv"


def run_gridsearch_experiments(
    problem_name, parameters, output_dir="output", exclude=[]
):

    try:
        problem, global_optimum = PROBLEM_MAP[problem_name]()
    except KeyError:
        raise KeyError(f"Problem not yet supported: {problem_name}")

    results = {}

    algos = set(["rhc", "sa", "ga", "mimic"]) - set(exclude)
    for algo_name in algos:
        print(f"Running Grid Search for {algo_name}")
        algo = ALGO_KEYMAP[algo_name]
        params = parameters[algo_name]
        results[algo_name] = grid_search(problem, global_optimum, algo, params)

    for algo_name, df in results.items():
        output_path = generate_gsearch_filename(problem_name, algo_name, OUTPUT_DIR)
        df.to_csv(output_path)

    return results


def run_reliability_experiments(
    problem_name, output_dir="output", params=None, problem_space={}, exclude=[]
):
    """run_reliability_experiments

    Parameters
    ----------

    problem_name : Which problem to analyze
    output_dir : Directory in which to store results
    params : explicit parameters to use per algorithm. If not provided, will look for a grid search results file and extract best hyperparameters.
    exclude : Algorithms to exlude from the experiments

    Returns
    -------
    """

    algos = set(["rhc", "sa", "ga", "mimic"]) - set(exclude)
    for algo_name in algos:
        if not params:
            # If algorithm parameters weren't provided, see if there are results in output folder to draw from
            gsearch_file = generate_gsearch_filename(
                problem_name, algo_name, OUTPUT_DIR
            )
            param_df = pd.read_csv(gsearch_file, index_col=0)
            algo_params = extract_best_hyperparameters(param_df)
        else:
            try:
                algo_params = params[algo_name]
            except:
                warnings.warn(
                    f"Couldn't find parameters for algorithm {algo_name}. Continuing with default parameters"
                )
                algo_params = {}

        algo = ALGO_KEYMAP[algo_name]
        results, curves = reliability_test(
            problem_name, algo, algo_params, problem_space=problem_space
        )

        results_output_path = os.path.join(
            output_dir, f"{algo_name}_{problem_name}_reliability.csv"
        )
        curves_output_path = os.path.join(
            output_dir, f"{algo_name}_{problem_name}_reliability_curves.json"
        )
        results.to_csv(results_output_path)

        with open(curves_output_path, "w+") as curve_output:
            json.dump(curves, curve_output)


def grid_search(problem, global_optimum, algo, parameter_grid, cores=-1):
    param_list = list(ParameterGrid(parameter_grid))

    N_keys = len(parameter_grid.keys())
    N_entries = len(param_list)

    columns = list(parameter_grid.keys()) + [
        "Time Elapsed",
        "Best Fitness",
        "Fitness Evaluations",
        "Global Optimum",
        "Found Global Optimum",
    ]

    N_cols = len(columns)
    result_array = np.zeros((N_entries, N_cols))

    result_df = pd.DataFrame(result_array, columns=columns)

    results = defaultdict(dict)

    # TODO: Multiprocess if it takes too long
    for i, params in enumerate(param_list):

        start = datetime.now()

        restart_count = 0
        total_evaluations = 0
        best_fitness = -1

        while best_fitness < global_optimum and restart_count < MAX_RESTARTS:
            try:
                _, fitness, curve = run_optimizer(algo, problem, params)

                if fitness > best_fitness:
                    best_fitness = fitness

                total_evaluations += EVALUATION_COUNT
                restart_count += 1
            except ValueError:
                print("Math Domain Error")
                continue

        end = datetime.now()
        time_elapsed = end - start

        result_df.iloc[i]["Time Elapsed"] = parse_parameter_value(time_elapsed)
        result_df.iloc[i]["Best Fitness"] = best_fitness
        result_df.iloc[i]["Fitness Evaluations"] = total_evaluations
        result_df.iloc[i]["Global Optimum"] = global_optimum
        result_df.iloc[i]["Found Global Optimum"] = best_fitness == global_optimum

        for key, value in params.items():
            result_df.iloc[i][key] = parse_parameter_value(value)

    return result_df


def reliability_test(problem_name, algo, algo_parameters, problem_space={}, N_tests=20):
    curves = {}

    assert (
        len(problem_space) == 1
    ), "Reliability test only supports one variable in problem instantiation"

    key = list(problem_space)[0]
    values = problem_space[key]

    result_df = pd.DataFrame(
        np.zeros((N_tests * len(values), len(METRICS) + 2)),
        columns=METRICS + ["Parameter Value", "Iterations"],
    )
    for i, param in enumerate(values):
        print(f"Running Test with {key}: {param}")
        j = 0
        error_count = 0  # Weird float errors that I can't fix
        pbar = tqdm(total=N_tests)
        while j < N_tests:

            try:
                row = i * N_tests + j
                kwargs = {key: param}

                np.random.seed(j + error_count)  # For replicability

                problem, global_optimum = PROBLEM_MAP[problem_name](
                    **kwargs
                )  # New random problem instantiation
                start = datetime.now()

                best_state, best_fitness, curve = run_optimizer(
                    algo, problem, algo_parameters
                )
                end = datetime.now()

                time_elapsed = end - start

                result_df.iloc[row]["Time Elapsed"] = parse_parameter_value(
                    time_elapsed
                )
                result_df.iloc[row]["Best Fitness"] = best_fitness
                result_df.iloc[row]["Fitness Evaluations"] = EVALUATION_COUNT
                result_df.iloc[row]["Global Optimum"] = global_optimum
                result_df.iloc[row]["Parameter Value"] = param
                result_df.iloc[row]["Iterations"] = len(curve)

                curve_key = f"{key}_{param}_{i}"
                curves[curve_key] = curve.tolist()
                j += 1
                pbar.update(j)
            except ValueError as e:
                print("Math Error")
                error_count += 1
                continue

    return result_df, curves


def run_optimizer(algo, problem, algorithm_params):
    """Utility function that adds the universal hyperparameters into the algorithm before running the optimizer

    Parameters
    ----------

    algo : The mlrose algorithm
    problem : An mlrose.DiscreteOpt problem
    algorithm_params : The algorithm-specific hyperparameters

    Returns
    -------
    """
    # Reset the evaluation count before running any optimization experiment
    global EVALUATION_COUNT
    EVALUATION_COUNT = 0
    all_params = combine_dicts(UNIVERSAL_PARAMETERS, algorithm_params)
    return algo(problem, **all_params)


def extract_best_hyperparameters(gridsearch_results):
    """extract_best_hyperparameters

    Parameters
    ----------

    gridsearch_results : dataframe of gridsearch results

    Returns
    the best hyperparameters as a dictionary
    -------
    """
    params = [col for col in gridsearch_results.columns if col not in METRICS]

    best_row = gridsearch_results["Best Fitness"].idxmax()
    best_params = gridsearch_results[params].iloc[best_row].to_dict()
    for key, value in best_params.items():
        if key == "schedule":  # Special case of conversion
            best_params[key] = ExpDecay(exp_const=value)
        if value >= 1:
            best_params[key] = int(
                value
            )  # This might not be very safe but it works for these cases

    return best_params


def collect_final_results(problem, algo, parameters):
    """measure_reliability

    Measures the reliability of an algorithm based on the variance in performance

    Parameters
    ----------

    problem :
    algo :
    parameters :

    Returns
    -------
    """


# results = run_optimization_experiment("knapsack", GRID)
#
#
# results_knapsack = run_gridsearch_experiments("knapsack", GRID)
# results_flipflop = run_gridsearch_experiments("flipflop", GRID)
# results_fourpeaks = run_gridsearch_experiments("fourpeaks", GRID)
# results_tsp = run_gridsearch_experiments("tsp", GRID)
# results_queens = run_gridsearch_experiments("queens", GRID)


# run_reliability_experiments(
#     "knapsack", OUTPUT_DIR, problem_space=PROBLEM_GRID["knapsack"]
# )

# run_reliability_experiments(
#     "flipflop", OUTPUT_DIR, problem_space=PROBLEM_GRID["flipflop"]
# )

# run_reliability_experiments(
#     "fourpeaks", OUTPUT_DIR, problem_space=PROBLEM_GRID["fourpeaks"]
# )

# run_reliability_experiments(
#     "tsp", OUTPUT_DIR, problem_space=PROBLEM_GRID["tsp"]
# )
# run_reliability_experiments(
#     "queens", OUTPUT_DIR, problem_space=PROBLEM_GRID["queens"]
# )


def run_all(exclude=[],experiment_type='both'):
    assert experiment_type in ['both','reliability','gridsearch'], f"Experiment type {type} not supported"
    problems = set(['knapsack','flipflop','fourpeaks','tsp','queens']) - set(exclude)
    for p in problems:
        print(f"Running Experiment: {p}")
        if experiment_type in ['both','gridsearch']:
            run_gridsearch_experiments(p, GRID)
        
        if experiment_type in ['both','reliability']:
            run_reliability_experiments(p, OUTPUT_DIR, problem_space=PROBLEM_GRID[p])


if __name__ == "__main__":
    run_all(exclude=['tsp','queens','flipflop','knapsack'])

