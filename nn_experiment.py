import mlrose
import pandas as pd

from data_utils import load_intention
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from mlrose import ExpDecay
import datetime
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
import numpy as np

RANDOM_STATE = 5
FOLDS = 5
TRAIN_PCT = 0.8
METRICS = ["Train_Acc", "Train_F1", "Val_Acc", "Val_F1", "Train_Time", "Iterations"]

X, y = load_intention()


GRID = {
    "random_hill_climb": {"restarts": [10], "learning_rate": [0.0001, 0.001, 0.01]},
    "simulated_annealing": {"learning_rate": [0.0001, 0.001, 0.01],
        "schedule": [
            ExpDecay(exp_const=0.001),
            ExpDecay(exp_const=0.005),
            ExpDecay(exp_const=0.01),
            ExpDecay(exp_const=0.05),
            ExpDecay(exp_const=0.1),
        ],
    },
    "genetic_alg": {"learning_rate": [0.0001, 0.001, 0.01], "mutation_prob": [0.05, 0.1, 0.2, 0.4]},
}

NN_PARAMS = {
    "hidden_nodes": [64, 64],
    "clip_max": 5,
    "random_state": RANDOM_STATE,
    "bias": True,
    "curve": True,
}

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X.values, y.values, shuffle=True, random_state=RANDOM_STATE, train_size=TRAIN_PCT
)


def grid_search_nn(algo):
    """Returns Grid Search Results on Neural Network

    Parameters
    ----------

    algo : name of algorithm to use as nn optimizer

    Returns results dataframe
    -------
    """

    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

    parameters = list(ParameterGrid(GRID[algo]))
    n_params = len(parameters)
    param_names = list(parameters[0].keys())
    columns = METRICS + param_names + ["Fold"]

    n_cols = len(columns)

    results_df = pd.DataFrame(np.zeros((n_params * FOLDS, n_cols)), columns=columns)

    for i, params in enumerate(parameters):
        print(f"Testing params: {params}")

        nn = mlrose.NeuralNetwork(algorithm=algo, **params, **NN_PARAMS)

        for j, (train_ix, val_ix) in enumerate(kfold.split(Xtrain)):
            print(f"Running Fold: {j}")
            results = {}
            train_fold_x = Xtrain[train_ix, :]
            val_fold_x = Xtrain[val_ix, :]

            train_fold_y = ytrain[train_ix]
            val_fold_y = ytrain[val_ix]

            start_train = datetime.datetime.now()
            nn.fit(train_fold_x, train_fold_y)
            end_train = datetime.datetime.now()

            train_preds = nn.predict(train_fold_x)
            val_preds = nn.predict(val_fold_x)

            # Collect Metrics
            train_time = (end_train - start_train).total_seconds()
            train_accuracy = accuracy_score(train_fold_y, train_preds)
            val_accuracy = accuracy_score(val_fold_y, val_preds)
            train_f1_score = f1_score(train_fold_y, train_preds)
            val_f1_score = f1_score(val_fold_y, val_preds)
            iterations = len(nn.fitness_curve)

            results[f"Train_Acc"] = train_accuracy
            results[f"Val_Acc"] = val_accuracy
            results[f"Train_F1"] = train_f1_score
            results[f"Val_F1"] = val_f1_score
            results[f"Train_Time"] = train_time
            results[f"Iterations"] = iterations
            results[f"Fold"] = j

            for key, value in params.items():
                if isinstance(value, ExpDecay):
                   results[key] = value.exp_const
                else:
                    results[key] = value

            results_df.iloc[i * FOLDS + j] = results

    return results_df

def collect_grid_search_data(algo, output_path):
    """Writes grid search results to csv file

    Parameters
    ----------

    algo : Name of algorithm to use
    output_path : Output file

    Returns results
    -------
    """
    result = grid_search_nn(algo)
    result.to_csv(output_path)
    return result


if __name__ == "__main__":
    # rhc_results = collect_grid_search_data('random_hill_climb', 'output/rhc_nn_gsresults.csv')
    # sa_results = collect_grid_search_data('simulated_annealing', 'output/sa_nn_gsresults.csv')
    # ga_results = collect_grid_search_data('genetic_alg', 'output/ga_nn_gsresults.csv')
    pass


