import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, make_scorer, recall_score, precision_score, f1_score
)
from sklearn.model_selection import (
    KFold, train_test_split, GridSearchCV, cross_validate
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


def cross_validation(name, model, _X, _y, _cv):
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

    # Define custom scoring metrics
    _scoring = {
        'accuracy': make_scorer(accuracy_score),  # How many predictions out of the whole were correct?
        'precision': make_scorer(precision_score, average='weighted'),  # How many out of the predicted
        # positives were actually positive?
        'recall': make_scorer(recall_score, average='weighted'),  # How many positive samples are captured
        # by the positive predictions?
        'f1_score': make_scorer(f1_score, average='macro')  # How balanced is the tradeoff between precision and recall?
    }

    scores = cross_validate(estimator=model,
                            X=_X,
                            y=_y,
                            cv=_cv,
                            scoring=_scoring,
                            return_train_score=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    result = {}

    for metric in metrics:
        train_scores = scores[f'test_{metric}']
        train_scores_mean = round(train_scores.mean(), 3)
        train_scores_std = round(train_scores.std(), 3)

        test_scores = scores[f'test_{metric}']
        test_scores_mean = round(test_scores.mean(), 3)

        result[f'train_{metric}'] = train_scores
        result[f'train_{metric}_mean'] = train_scores_mean
        result[f'train_{metric}_std'] = train_scores_std

        result[f'test_{metric}'] = test_scores
        result[f'test_{metric}_mean'] = test_scores_mean

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

    baseline_score = dummy_stratified_clf(X_train, y_train)
    logger.debug(f"Baseline performance of the dummy classifier: {baseline_score}")

    # Create models
    models = [SVC(), LogisticRegression(), RandomForestClassifier()]
    names = ['SVC', 'LR', 'RF']

    # Split the dataset into k equal partitions (each partition is divided in train and validation data)
    cv = KFold(n_splits=_n_splits, random_state=42, shuffle=True)

    # Prepare dataframe for results
    results_df = pd.DataFrame(columns=['Set', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    models_size_df = pd.DataFrame(columns=['Model', 'Size'])

    logger.info("Beginning to train models using GPT embeddings...")

    # Collect total size of all models
    total_models_size = 0

    for model, name in zip(models, names):
        logger.info(f"Initiating {name}...")

        ### Model checking
        best_params = hyperparameter_optimization(X_train, y_train, cv, model, name)
        model.set_params(**best_params)
        scores = cross_validation(name, model, X_train, y_train, cv)
        results_df = results_to_df(name, scores, results_df)

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

        # Predict label on test data with trained model
        model_predictions = model.predict(X_test)

        # Create a dictionary to store the filename-prediction value pairs
        filename_to_prediction = {}

        # Iterate through the filenames and model predictions arrays simultaneously
        for filename, prediction in zip(test_data['addressfname'], model_predictions):
            # Reverse binary classification
            filename_to_prediction[filename] = 'ProbableAD' if prediction == 1 else 'Control'

        # Fill the 'Prediction' column using the dictionary
        model_test_results['Prediction'] = model_test_results['ID'].map(filename_to_prediction)

        # Save the updated DataFrame in a new CSV file
        model_test_results.to_csv((config.embedding_results_dir / "f'task1_{name}.csv'").resolve(), index=False)

        # Evaluate performance on test data
        evaluate_similarity(model_test_results)

    logger.info("Training using GPT embeddings done.")

    # Adjust resulting dataframe
    results_df = results_df.sort_values(by='Set', ascending=False)
    results_df = results_df.reset_index(drop=True)

    # Add baseline score to dataframe
    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Test',
                                                       'Model': 'Dummy',
                                                       'Accuracy': baseline_score,
                                                       }])], ignore_index=True)

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


def evaluate_similarity(model_test_results):
    # Actual diagnosed data
    test_results_task1 = pd.read_csv(config.test_results_task1)
    real_diagnoses = test_results_task1['Dx']
    predicted_diagnoses = model_test_results['Prediction']
    # Calculate the number of matching values
    matching_values = sum(real_diagnoses == predicted_diagnoses)
    # Calculate the total number of values
    total_values = len(real_diagnoses)
    # Calculate the percentage of matching values
    similarity_percentage = (matching_values / total_values) * 100
    print(f"The similarity between the real and predicted diagnoses is {similarity_percentage:.2f}%.")


# Tune hyperparameters with GridSearchCV
def hyperparameter_optimization(X_train, y_train, cv, model, name):
    # Get the parameter grids
    lr_param_grid, rf_param_grid, svc_param_grid = param_grids()
    grid_search = None
    if name == 'SVC':
        grid_search = GridSearchCV(estimator=model, param_grid=svc_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
    elif name == 'LR':
        grid_search = GridSearchCV(estimator=model, param_grid=lr_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
    elif name == 'RF':
        grid_search = GridSearchCV(estimator=model, param_grid=rf_param_grid, cv=cv, n_jobs=-1, error_score=0.0)

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
    plot_f1_path = (save_dir / f'plot_precision_{name}.png').resolve()
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
                                                       'Accuracy': f"{scores['train_accuracy_mean']}"
                                                                   f"({scores['train_accuracy_std']})",
                                                       'Precision': f"{scores['train_precision_mean']} "
                                                                    f"({scores['train_precision_std']})",
                                                       'Recall': f"{scores['train_recall_mean']} "
                                                                 f"({scores['train_recall_std']})",
                                                       'F1': f"{scores['train_f1_score_mean']} "
                                                             f"({scores['train_f1_score_std']})",
                                                       }])], ignore_index=True)

    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Test',
                                                       'Model': name,
                                                       'Accuracy': scores['test_accuracy_mean'],
                                                       'Precision': scores['test_precision_mean'],
                                                       'Recall': scores['test_recall_mean'],
                                                       'F1': scores['test_f1_score_mean']
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
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold", "7th Fold", "8th Fold",
              "9th Fold", "10th Fold"]
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


def dummy_stratified_clf(X, y):
    """
    DummyClassifier makes predictions that ignore the input features.

    This classifier serves as a simple baseline to compare against other more complex classifiers.
    It gives us a measure of “baseline” performance — i.e. the success rate one should expect to achieve even
    if simply guessing.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    stratified_clf = DummyClassifier(strategy='stratified').fit(X_train, y_train)

    score = round(stratified_clf.score(X_test, y_test), 3)

    return score


# AD classification using acoustic features from OpenSMILE
def classify_acoustic(acoustic_features_csv, transcription_csv, _n_splits):
    # TODO: Adjust dataset
    dataset = None
    # Define the dependent variable that needs to be predicted (labels)
    y = dataset['diagnosis'].values
    # Define the independent variable
    X = acoustic_features_csv

    # Create models
    models = [SVC(), LogisticRegression(), RandomForestClassifier()]
    names = ['SVC', 'LR', 'RF']

    # Split the dataset into k equal partitions
    cv = KFold(n_splits=_n_splits, random_state=42, shuffle=True)

    # Prepare dataframe for results
    results_df = pd.DataFrame(columns=['Set', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1'])

    # Create the parameter grid
    lr_param_grid, rf_param_grid, svc_param_grid = param_grids()

    logger.info("Beginning to train models using acoustic embeddings...")

    for model, name in zip(models, names):
        # TODO: Ask Tobi: Standard-/MinMaxScaler needed to normalize data?

        # Tune hyperparameters with GridSearchCV
        best_params = None
        if name == 'SVC':
            grid_search = GridSearchCV(estimator=model, param_grid=svc_param_grid, cv=cv, n_jobs=-1,
                                       error_score='raise')
            grid_search.fit(X, y)
            best_params = grid_search.best_params_
        elif name == 'LR':
            grid_search = GridSearchCV(estimator=model, param_grid=lr_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
            grid_search.fit(X, y)
            best_params = grid_search.best_params_
        elif name == 'RF':
            grid_search = GridSearchCV(estimator=model, param_grid=rf_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
            grid_search.fit(X, y)
            best_params = grid_search.best_params_
        logger.debug(f"Best params for model {name}: {best_params}")
        model.set_params(**best_params)

        # Perform cross-validation with custom scoring metrics and best params
        results = cross_validation(name, model, X, y, cv)
        logger.debug(results)
        results_df = results_to_df(name, results, results_df)

        visualize_results(_n_splits, name, results, config.acoustic_results_dir)

    logger.info("Training using acoustic features done.")

    # Adjust resulting dataframe
    results_df = results_df.sort_values(by='Set', ascending=False)
    results_df = results_df.reset_index(drop=True)

    # Save results to csv
    results_df.to_csv(config.acoustic_results_file)

    logger.info(f"Writing {config.acoustic_results_file}...")
