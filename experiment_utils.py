import os
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from multiprocessing import pool

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
METRICS = ["Time Elapsed", "Best Fitness", "Fitness Evaluations"]
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
        "max_attempts": [1000],
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
    m = np.zeros((n_items, capacity))
    for j in range(capacity):
        pass

    for i in range(1, n_items):
        for j in range(capacity):
            if weights[i] > j:
                m[i, j] = m[i - 1, j]
            else:
                m[i, j] = max(m[i - 1, j], m[i - 1, j - weights[i]] + values[i])

    return m.max()  # The maximum value


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

        global_optimum = calculate_knapsack_global_optimum(
            values, weights, N_items, np.ceil(max_weight_pct * weights.sum())
        )

        knapsack_fitness = count_evaluations(
            mlrose.Knapsack, weights, values, max_weight_pct
        )
        knapsack_fitness_final = mlrose.CustomFitness(knapsack_fitness)
        knapsack = mlrose.DiscreteOpt(length=N_items, fitness_fn=knapsack_fitness_final)

        return knapsack, global_optimum

    return {
        "fourpeaks": fourpeaks_factory,
        "knapsack": knapsack_factory,
        "flipflop": flipflop_factory,
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
    problem_name, output_dir="output", params=None, exclude=[]
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
            gsearch_file = generate_gsearch_filename(
                problem_name, algo_name, OUTPUT_DIR
            )
            param_df = pd.read_csv(gsearch_file, index_col=0)
            algo_params = extract_best_hyperparameters(param_df)
        else:
            algo_params = params[algo_name]

        algo = ALGO_KEYMAP[algo_name]
        results, curves = reliability_test(problem_name, algo, algo_params)

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
        "Found Global Optimum"
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
            print(f"Attempt Number {restart_count+1}")
            _, fitness, curve = run_optimizer(algo, problem, params)

            if fitness > best_fitness:
                best_fitness = fitness

            total_evaluations += EVALUATION_COUNT
            restart_count += 1

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


def reliability_test(problem_name, algo, algo_parameters, problem_grid={}, N_tests=20):
    curves = {}

    result_df = pd.DataFrame(np.zeros((N_tests, len(METRICS))), columns=METRICS)

    for i in tqdm(range(N_tests)):
        np.random.seed(i)  # For replicability

        problem, global_optimum = PROBLEM_MAP[
            problem_name
        ]()  # New random problem instantiation
        start = datetime.now()
        best_fitness = None

        best_state, best_fitness, curve = run_optimizer(algo, problem, algo_parameters)
        end = datetime.now()

        time_elapsed = end - start
        result_df.iloc[i]["Time Elapsed"] = parse_parameter_value(time_elapsed)
        result_df.iloc[i]["Best Fitness"] = best_fitness
        result_df.iloc[i]["Fitness Evaluations"] = EVALUATION_COUNT
        result_df.iloc[i]["Global Optimum"] = global_optimum
        curves[i] = curve.tolist()

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

results = run_gridsearch_experiments("knapsack", GRID)
