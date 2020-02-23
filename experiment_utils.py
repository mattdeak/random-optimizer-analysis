import mlrose
import numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
import pandas as pd
import os
from multiprocessing import pool

MAX_ITERS = 5000
evaluation_count = 0


def count_evaluations(fitness_fn):
    def wrapper(state, *args, **kwargs):
        fitness = fitness_fn(*args, **kwargs)
        global evaluation_count
        evaluation_count += 1
        return fitness.evaluate(state)

    return wrapper


queen_fitness = count_evaluations(mlrose.Queens)
queen_fitness_final = mlrose.CustomFitness(queen_fitness)

fourpeaks_fitness = count_evaluations(mlrose.FourPeaks)
fourpeaks_fitness_final = mlrose.CustomFitness(fourpeaks_fitness)


queens = mlrose.DiscreteOpt(length=8, fitness_fn=queen_fitness_final, maximize=False, max_val=8)
fourpeaks = mlrose.DiscreteOpt(length=100, fitness_fn=fourpeaks_fitness_final)

PROBLEM_MAP = {"queens": queens, 'fourpeaks':fourpeaks}

ALGO_KEYMAP = {
    "rhc": mlrose.random_hill_climb,
    "sa": mlrose.simulated_annealing,
    "ga": mlrose.genetic_alg,
    "mimic": mlrose.mimic,
}


GRID = {
    "rhc": {"max_attempts": [10, 100, 1000], "restarts": [0, 5, 10]},
    "sa": {"schedule": [mlrose.ExpDecay()], "max_attempts": [10, 100, 1000]},
    "ga": {"pop_size": [100, 200, 300], "mutation_prob": [0.05, 0.1, 0.2, 0.4],},
    "mimic": {"pop_size": [100, 200, 300], "keep_pct": [0.1, 0.3, 0.5, 0.7, 0.9]},
}

UNIVERSAL_PARAMETERS = {'curve': True, 'max_iters': MAX_ITERS}

def combine_dicts(dict1, dict2):
    "Returns a dictionary with keys and values combined"
    combined = dict1.copy()
    combined.update(dict2)
    return combined


def run_optimization_experiment(
    problem_name, parameters, output_dir="output", exclude=[]
):

    try:
        problem = PROBLEM_MAP[problem_name]
    except KeyError:
        raise KeyError(f"Problem not yet supported: {problem_name}")

    results = {}

    algos = set(["rhc", "sa", "ga", "mimic"]) - set(exclude)
    for algo_name in algos:
        algo = ALGO_KEYMAP[algo_name]
        params = parameters[algo_name]
        results[algo_name] = grid_search(problem, algo, params)

    for algo_name, df in results.items():
        output_path = (
            f"{os.path.join(output_dir, algo_name)}_{problem_name}_gsresults.csv"
        )
        df.to_csv(output_path)

    return results


def grid_search(problem, algo, parameter_grid, cores=-1):

    param_list = list(ParameterGrid(parameter_grid))

    N_keys = len(parameter_grid.keys())
    N_entries = len(param_list)

    columns = list(parameter_grid.keys()) + [
        "Time Elapsed",
        "Best Fitness",
        "Fitness Evaluations",
    ]

    N_cols = len(columns)
    result_array = np.zeros((N_entries, N_cols))

    result_df = pd.DataFrame(result_array, columns=columns)

    results = defaultdict(dict)

    # TODO: Multiprocess if it takes too long
    for i, params in enumerate(param_list):
        global evaluation_count
        evaluation_count = 0

        start = datetime.now()
        best_state, best_fitness, curve = run_optimizer(algo, problem, params)
        print(curve)
        end = datetime.now()

        time_elapsed = end - start
        result_df.iloc[i]["Time Elapsed"] = time_elapsed

        print(time_elapsed)
        result_df.iloc[i]["Best Fitness"] = best_fitness

        print(f"Evaluation Count: {evaluation_count}")
        result_df.iloc[i]["Fitness Evaluations"] = evaluation_count

        for key, value in params.items():
            result_df.iloc[i][key] = value

    return result_df

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



results = run_optimization_experiment("fourpeaks", GRID)
