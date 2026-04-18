import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, make_scorer, recall_score, precision_score, f1_score
)
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.svm import SVC

import config
from config import logger

"""
Create embeddings of the control group, and compare it with the embeddings of the diagnosed group.
The text-davinci-003 model will then evaluate it.
"""


# Turning the embeddings into a NumPy array, which will provide more flexibility in how to use it.
# It will also flatten the dimension to 1-D, which is the required format for many subsequent operations.
def embeddings_to_array(embeddings_file):
    df = pd.read_csv(embeddings_file)
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    logger.debug(df.head())
    return df


def scoring_metrics():
    return {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1_score': make_scorer(f1_score, average='macro')
    }


def build_cv(n_splits):
    return StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)


def build_grid_search(model, name, cv):
    lr_param_grid, rf_param_grid, svc_param_grid = param_grids()
    param_grid = None
    if name == 'SVC':
        param_grid = svc_param_grid
    elif name == 'LR':
        param_grid = lr_param_grid
    elif name == 'RF':
        param_grid = rf_param_grid

    return GridSearchCV(estimator=clone(model),
                        param_grid=param_grid,
                        cv=cv,
                        n_jobs=-1,
                        error_score=0.0,
                        scoring='accuracy')


def cross_validation(estimator, _X, _y, _cv):
    """ Function to perform K-Fold Cross-Validation
    We do this to see which model proves better at predicting the test set points.
    But once we have used cross-validation to evaluate the performance,
    we train that model on all the data. We don't use the actual model
    instances we trained during cross-validation for our final predictive model.
     Parameters
     ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    _X: array
         This is the matrix of features.
    _y: array
         This is the target variable.
    _cv: int, default=5
        Determines the number of folds for cross-validation.
     Returns
     -------
     The function returns a dictionary containing the metrics 'accuracy', 'precision',
     'recall', 'f1' for both training set and validation set.
    """

    scores = cross_validate(estimator=estimator,
                            X=_X,
                            y=_y,
                            cv=_cv,
                            scoring=scoring_metrics(),
                            return_train_score=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    result = {}

    for metric in metrics:
        train_scores = scores[f'train_{metric}']
        train_scores_mean = round(train_scores.mean(), 3)
        train_scores_std = round(train_scores.std(), 3)

        test_scores = scores[f'test_{metric}']
        test_scores_mean = round(test_scores.mean(), 3)
        test_scores_std = round(test_scores.std(), 3)

        result[f'train_{metric}'] = train_scores
        result[f'train_{metric}_mean'] = train_scores_mean
        result[f'train_{metric}_std'] = train_scores_std

        result[f'test_{metric}'] = test_scores
        result[f'test_{metric}_mean'] = test_scores_mean
        result[f'test_{metric}_std'] = test_scores_std

    return result


def classify_embedding(train_data, test_data, _n_splits):
    """
    Perform classification using linguistic features (embeddings) from transcribed speech.

    Args:
        train_data (DataFrame): Training data containing 'diagnosis' labels and 'embedding' features.
                                Split into train and validation sets.
        test_data (DataFrame): Test data containing 'embedding' features.
        _n_splits (int): Number of folds for cross-validation.

    Returns:
        None

    This function trains and evaluates multiple machine learning models using the provided embeddings.
    It performs the following steps:
    - Loads training and test data.
    - Calculates baseline performance using a dummy stratified classifier.
    - Initializes machine learning models.
    - Creates cross-validation K-folds.
    - Iterates through each model:
        Model checking:
        - Performs hyperparameter optimization using cross-validation.
        - Records model performance on the training data using cross-validation.
        - Visualizes cross-validation results.
        Model building:
        - Trains the model on the entire training set with the best hyperparameters.
        - Predicts labels on the test data using the trained model.
        - Stores predictions in a CSV file.
        - Evaluates performance on the test data by comparing it to real diagnoses.
        - Records model sizes before and after training.
    - Saves performance results to a CSV file.
    - Saves model sizes to a CSV file.
    """
    logger.info("Initiating classification with GPT-3 text embeddings...")

    # Define the dependent variable that needs to be predicted (labels)
    y_train = train_data['diagnosis'].values
    # Define the independent variable
    X_train = train_data['embedding'].to_list()
    # Test data which is only used after training the model with the train data
    X_test = test_data['embedding'].to_list()

    outer_cv = build_cv(_n_splits)
    inner_cv = build_cv(min(5, _n_splits))

    baseline_scores = cross_validation(DummyClassifier(strategy='stratified', random_state=42),
                                       X_train,
                                       y_train,
                                       outer_cv)
    logger.debug(f"Baseline dummy classifier scores: {baseline_scores}")

    # Create models
    models = [SVC(probability=True, random_state=42),
              LogisticRegression(random_state=42),
              RandomForestClassifier(random_state=42)]
    names = ['SVC', 'LR', 'RF']

    # Prepare dataframe for results
    results_df = pd.DataFrame(columns=['Set', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    models_size_df = pd.DataFrame(columns=['Model', 'Size'])

    logger.info("Beginning to train models using GPT embeddings...")
    results_df = results_to_df('Dummy', baseline_scores, results_df)

    # Collect total size of all models
    total_models_size = 0

    for model, name in zip(models, names):
        logger.info(f"Initiating {name}...")

        ### Model checking
        scores = cross_validation(build_grid_search(model, name, inner_cv), X_train, y_train, outer_cv)
        results_df = results_to_df(name, scores, results_df)
        best_params = hyperparameter_optimization(X_train, y_train, inner_cv, model, name)
        logger.info(f"Best parameters for {name}: {best_params}")
        model.set_params(**best_params)

        # Visualize folds for different metrics in plots
        visualize_results(_n_splits, name, scores, (config.embedding_results_dir / "plots").resolve())

        ### Model building
        # Get size of the serialized model in bytes before training
        model_size = len(pickle.dumps(model, -1))
        logger.debug(f"Model size of {name} before training: {model_size} bytes.")

        # Train each model on the entire training set with best hyperparameters
        model.fit(X_train, y_train)

        # Get size of the serialized model in bytes after training
        model_size = len(pickle.dumps(model, -1))
        logger.debug(f"Model size of {name} after training: {model_size} bytes.")
        total_models_size += model_size

        # Add trained model size to DataFrame
        models_size_df = pd.concat([models_size_df, pd.DataFrame([{'Model': name,
                                                                   'Size': f"{model_size} B",
                                                                   }])], ignore_index=True)

        # Load the empty task1 results CSV file
        model_test_results = pd.read_csv(config.empty_test_results_file)

        model_predictions = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            model_probabilities = model.predict_proba(X_test)
            positive_class_index = int(np.where(model.classes_ == 1)[0][0])
            positive_probabilities = model_probabilities[:, positive_class_index]
            predictions_with_confidence = ["{} ({:.2f})".format(int(pred), prob) for pred, prob in
                                           zip(model_predictions, positive_probabilities)]
            logger.info(f"{name} probabilities: {model_probabilities}")
            logger.info(f"{predictions_with_confidence}")

        # Create a dictionary to store the filename-prediction value pairs
        filename_to_prediction = {}

        # Iterate through the filenames and model predictions arrays simultaneously
        for filename, prediction in zip(test_data['addressfname'], model_predictions):
            # Reverse binary classification
            filename_to_prediction[filename] = 'ProbableAD' if prediction == 1 else 'Control'

        # Fill the 'Prediction' column using the dictionary
        model_test_results['Prediction'] = model_test_results['ID'].map(filename_to_prediction)

        # Save the updated DataFrame in a new CSV file
        model_test_results_csv = (config.embedding_results_dir / f'task1_{name}.csv').resolve()
        model_test_results.to_csv(model_test_results_csv, index=False)
        logger.info(f"Writing {model_test_results_csv}...")

        # Evaluate performance on test data
        test_metrics = evaluate_similarity(name, model_test_results)
        results_df = test_results_to_df(name, test_metrics, results_df)

    logger.info("Training using GPT embeddings done.")

    # Adjust resulting dataframe
    results_df['Set'] = pd.Categorical(results_df['Set'],
                                       categories=['Train', 'Validation', 'Test'],
                                       ordered=True)
    results_df = results_df.sort_values(by=['Model', 'Set']).reset_index(drop=True)

    # Save results to csv
    embedding_results_file = (config.embedding_results_dir / 'embedding_results.csv').resolve()
    results_df.to_csv(embedding_results_file)
    logger.info(f"Writing {embedding_results_file}...")

    # Add total size to models_size
    logger.debug(f"Total size of all models: {total_models_size}.")
    models_size_df = pd.concat([models_size_df, pd.DataFrame([{'Model': 'Total',
                                                               'Size': f'{total_models_size} B',
                                                               }])], ignore_index=True)
    # Save results to csv

    models_size_df.to_csv(config.models_size_file)
    logger.info(f"Writing {config.models_size_file}...")

    logger.info("Classification with GPT-3 text embeddings done.")


def evaluate_similarity(name, model_test_results):
    test_results_task1 = pd.read_csv(config.test_results_task1)
    evaluation_df = pd.merge(test_results_task1,
                             model_test_results[['ID', 'Prediction']],
                             on='ID',
                             how='left',
                             validate='one_to_one')

    if evaluation_df['Prediction'].isna().any():
        missing_ids = evaluation_df.loc[evaluation_df['Prediction'].isna(), 'ID'].tolist()
        raise ValueError(f"Missing predictions for test IDs: {missing_ids}")

    real_diagnoses = evaluation_df['Dx']
    predicted_diagnoses = evaluation_df['Prediction']

    test_metrics = {
        'accuracy': round(accuracy_score(real_diagnoses, predicted_diagnoses), 3),
        'precision': round(precision_score(real_diagnoses, predicted_diagnoses, average='weighted'), 3),
        'recall': round(recall_score(real_diagnoses, predicted_diagnoses, average='weighted'), 3),
        'f1_score': round(f1_score(real_diagnoses, predicted_diagnoses, average='macro'), 3)
    }
    logger.info(f"Test accuracy for model {name}: {test_metrics['accuracy']:.3f}.")

    return test_metrics


# Tune hyperparameters with GridSearchCV
def hyperparameter_optimization(X_train, y_train, cv, model, name):
    grid_search = build_grid_search(model, name, cv)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params


#                                                            }])], ignore_index=True)
# Save results to csv
# models_size_file = (config.embedding_results_dir / 'embedding_models_size.csv').resolve()
# models_size_df.to_csv(models_size_file)
# logger.info(f"Writing {models_size_file}...")

# logger.info("Classification with GPT-3 text embeddings done.")


def param_grids():
    svc_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    lr_param_grid = [
        {'penalty': ['l1', 'l2'],
         'C': np.logspace(-4, 4, 20),
         'solver': ['liblinear'],
         'max_iter': [100, 200, 500, 1000]},
        {'penalty': ['l2'],
         'C': np.logspace(-4, 4, 20),
         'solver': ['lbfgs'],
         'max_iter': [200, 500, 1000]},
    ]
    rf_param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9],
    }
    return lr_param_grid, rf_param_grid, svc_param_grid


def visualize_results(_n_splits, name, results, save_dir):
    plot_accuracy_path = (save_dir / f'plot_accuracy_{name}.png').resolve()
    plot_precision_path = (save_dir / f'plot_precision_{name}.png').resolve()
    plot_recall_path = (save_dir / f'plot_recall_{name}.png').resolve()
    plot_f1_path = (save_dir / f'plot_f1_{name}.png').resolve()
    # Plot Accuracy Result
    plot_result(name,
                "Accuracy",
                f"Accuracy scores in {_n_splits} Folds",
                results["train_accuracy"],
                results["test_accuracy"],
                plot_accuracy_path)
    # Plot Precision Result
    plot_result(name,
                "Precision",
                f"Precision scores in {_n_splits} Folds",
                results["train_precision"],
                results["test_precision"],
                plot_precision_path)
    # Plot Recall Result
    plot_result(name,
                "Recall",
                f"Recall scores in {_n_splits} Folds",
                results["train_recall"],
                results["test_recall"],
                plot_recall_path)
    # Plot F1-Score Result
    plot_result(name,
                "F1",
                f"F1 Scores in {_n_splits} Folds",
                results["train_f1_score"],
                results["test_f1_score"],
                plot_f1_path)


def results_to_df(name, scores, results_df):
    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Train',
                                                       'Model': name,
                                                       'Accuracy': f"{scores['train_accuracy_mean']} "
                                                                   f"({scores['train_accuracy_std']})",
                                                       'Precision': f"{scores['train_precision_mean']} "
                                                                    f"({scores['train_precision_std']})",
                                                       'Recall': f"{scores['train_recall_mean']} "
                                                                 f"({scores['train_recall_std']})",
                                                       'F1': f"{scores['train_f1_score_mean']} "
                                                             f"({scores['train_f1_score_std']})",
                                                       }])], ignore_index=True)

    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Validation',
                                                       'Model': name,
                                                       'Accuracy': f"{scores['test_accuracy_mean']} "
                                                                   f"({scores['test_accuracy_std']})",
                                                       'Precision': f"{scores['test_precision_mean']} "
                                                                    f"({scores['test_precision_std']})",
                                                       'Recall': f"{scores['test_recall_mean']} "
                                                                 f"({scores['test_recall_std']})",
                                                       'F1': f"{scores['test_f1_score_mean']} "
                                                             f"({scores['test_f1_score_std']})"
                                                       }])], ignore_index=True)
    return results_df


def test_results_to_df(name, scores, results_df):
    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Test',
                                                       'Model': name,
                                                       'Accuracy': scores['accuracy'],
                                                       'Precision': scores['precision'],
                                                       'Recall': scores['recall'],
                                                       'F1': scores['f1_score']
                                                       }])], ignore_index=True)
    return results_df


# Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data, savefig_path=None):
    """Function to plot a grouped bar chart showing the training and validation
      results of the ML model in each fold after applying K-fold cross-validation.
     Parameters
     ----------
     x_label: str,
        Name of the algorithm used for training e.g 'Decision Tree'

     y_label: str,
        Name of metric being visualized e.g 'Accuracy'
     plot_title: str,
        This is the title of the plot e.g 'Accuracy Plot'

     train_data: list, array
        This is the list containing either training precision, accuracy, or f1 score.

     val_data: list, array
        This is the list containing either validation precision, accuracy, or f1 score.
     savefig_path: str
        Save figures to this path if not empty (by default)
     Returns
     -------
     The function returns a Grouped Barchart showing the training and validation result
     in each fold.
    """

    # Set size of plot
    fig = plt.figure(figsize=(12, 6))
    labels = [f"Fold {fold_idx}" for fold_idx in range(1, len(train_data) + 1)]
    X_axis = np.arange(len(labels))
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, train_data, 0.4, color='blue', label='Training')
    plt.bar(X_axis + 0.2, val_data, 0.4, color='red', label='Validation')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    if savefig_path is not None:
        fig.savefig(savefig_path, dpi=fig.dpi)
