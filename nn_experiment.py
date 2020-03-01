import mlrose
import os
import pandas as pd

from data_utils import load_intention
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from mlrose import ExpDecay
import datetime
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn.model_selection import learning_curve

RANDOM_STATE = 5
FOLDS = 5
TRAIN_PCT = 0.8
METRICS = ["Train_Acc", "Train_F1", "Val_Acc", "Val_F1", "Train_Time", "Iterations"]

X, y = load_intention()


GRID = {
    "random_hill_climb": {"restarts": [10], "learning_rate": [0.0001, 0.001, 0.01]},
    "simulated_annealing": {
        "learning_rate": [0.0001, 0.001, 0.01],
        "schedule": [
            ExpDecay(exp_const=0.001),
            ExpDecay(exp_const=0.005),
            ExpDecay(exp_const=0.01),
            ExpDecay(exp_const=0.05),
            ExpDecay(exp_const=0.1),
        ],
    },
    "genetic_alg": {
        "learning_rate": [0.0001, 0.001, 0.01],
        "mutation_prob": [0.05, 0.1, 0.2, 0.4],
    },
}

NN_PARAMS = {
    "hidden_nodes": [64, 64],
    "clip_max": 5,
    "random_state": RANDOM_STATE,
    "bias": True,
    "curve": True,
}

best_nn_params = {"learning_rate": 0.0001}

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
            results[f"Final_Loss"] = -nn.fitness_curve_[-1]
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


def collect_nn_results(
    X,
    y,
    algo,
    output_dir,
    runs=5,
    max_iters=500,
    extract_params=True,
    collect_lc=False,
):
    algo_map = {
        "ga": "genetic_alg",
        "rhc": "random_hill_climb",
        "sa": "simulated_annealing",
        "gs": "gradient_descent",
    }

    if extract_params:
        if algo == "gs":  # Use mlpclassifier
            # Use MLP Classifier to keep consistent with old results. Should be identical to mlrose except supports ADAM optimizer and is more efficient.
            nn = MLPClassifier(
                hidden_layer_sizes=[64, 64],
                learning_rate_init=0.0001,
                max_iter=max_iters,
            )
        else:
            params = extract_best_hyperparameters(algo, output_dir)
            nn = mlrose.NeuralNetwork(
                **NN_PARAMS,
                **params,
                algorithm=algo_map[algo],
                max_iters=max_iters,
                early_stopping=True,
            )
    else:
        if algo == "gs":  # Use mlpclassifier
            # Use MLP Classifier to keep consistent with old results. Should be identical to mlrose except supports ADAM optimizer and is more efficient.
            nn = MLPClassifier(
                hidden_layer_sizes=[64, 64],
                learning_rate_init=0.0001,
                max_iter=max_iters,
            )

        else:
            nn = mlrose.NeuralNetwork(
                **NN_PARAMS,
                algorithm=algo_map[algo],
                max_iters=max_iters,
                early_stopping=True,
            )

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X.values,
        y.values,
        shuffle=True,
        random_state=RANDOM_STATE,
        train_size=TRAIN_PCT,
    )
    # Collect train and test scores
    start = datetime.datetime.now()
    nn.fit(Xtrain, ytrain)
    end = datetime.datetime.now()

    train_time = (end - start).total_seconds()

    train_preds = nn.predict(Xtrain)
    test_preds = nn.predict(Xtest)

    train_acc = accuracy_score(ytrain, train_preds)
    train_f1 = f1_score(ytrain, train_preds)

    test_acc = accuracy_score(ytest, test_preds)
    test_f1 = f1_score(ytest, test_preds)

    # Iterations Until Best Fit
    if algo == "gs":
        iterations = len(nn.loss_curve_)
        loss = nn.loss_
    else:
        iterations = len(nn.fitness_curve)
        loss = -nn.fitness_curve[-1]

    results_data = pd.Series(
        {
            "Train_Acc": train_acc,
            "Test_Acc": test_acc,
            "Train_F1": train_f1,
            "Test_F1": test_f1,
            "Train_Time": train_time,
            "Iterations": iterations,
            "Final_Loss": loss
        }
    )
    results_data.to_csv(os.path.join(output_dir, f"{algo}_nn_results.csv"))

    # Collect Learning Curves
    if collect_lc:
        curves = []
        for i in range(runs):
            Xtrain, Xtest, ytrain, ytest = train_test_split(
                X.values, y.values, shuffle=True, random_state=i, train_size=TRAIN_PCT
            )
            nn.fit(Xtrain, ytrain)
            if algo == "gs":
                losses = nn.loss_curve_
            else:
                losses = -nn.fitness_curve
            curves.append(losses)

        # Forward fill best fitness to 500 iterations
        for i, curve in enumerate(curves):
            if len(curve) < max_iters:
                new_curve = np.zeros(max_iters)
                new_curve[: len(curve)] = curve
                new_curve[len(curve) :] = curve[-1]
                curves[i] = new_curve

        # Plot Learning Curve Data
        arr_curves = np.array(curves)

        filename = os.path.join(output_dir, f"{algo}_nn_curves.npy")
        np.save(filename, arr_curves.T)


def extract_best_hyperparameters(algo, output_dir):
    if algo == "gs":
        return best_nn_params

    filename = os.path.join(output_dir, f"{algo}_nn_gsresults.csv")
    df = pd.read_csv(filename, index_col=0)

    params = [col for col in df.columns if col not in METRICS + ["Fold"]]
    grouped = df.groupby(params)
    mean_groups = grouped.mean().reset_index()
    max_ix = mean_groups["Val_F1"].argmax()

    best_params = mean_groups.loc[max_ix, params].to_dict()

    for key, value in best_params.items():
        if key == "schedule":
            best_params[key] = ExpDecay(exp_const=value)
        elif value.is_integer():
            best_params[key] = int(value)

    print(best_params)
    return best_params


if __name__ == "__main__":
    # rhc_results = collect_grid_search_data('random_hill_climb', 'output/rhc_nn_gsresults.csv')
    # sa_results = collect_grid_search_data('simulated_annealing', 'output/sa_nn_gsresults.csv')
    # ga_results = collect_grid_search_data('genetic_alg', 'output/ga_nn_gsresults.csv')
    # print("Extracting Gradient Descent LC")
    collect_nn_results(X, y, "gs", "output", extract_params=False, max_iters=2000)
    print("Extracting Genetic Alg LC")
    collect_nn_results(X, y, "ga", "output", extract_params=True, max_iters=2000)
    print("Extracting Random Hill Climb LC")
    collect_nn_results(X, y, "rhc", "output", extract_params=True, max_iters=2000)
    print("Extracting Simulated Annealling LC")
    collect_nn_results(X, y, "sa", "output", extract_params=True, max_iters=2000)
