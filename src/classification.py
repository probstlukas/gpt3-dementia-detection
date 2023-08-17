import config
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    KFold, train_test_split, GridSearchCV, cross_validate
)
from sklearn.metrics import (
    accuracy_score, make_scorer, recall_score, precision_score, f1_score
)
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier

# Configure logging to display messages in the terminal
logging.basicConfig(level=logging.INFO)
# Create a logger instance for this file
log = logging.getLogger("Classification")

"""
Create embeddings of the control group, and compare it with the embeddings of the diagnosed group.
The text-davinci-003 model will then evaluate it.
"""


# Turning the embeddings into a NumPy array, which will provide more flexibility in how to use it.
# It will also flatten the dimension to 1-D, which is the required format for many subsequent operations.
def embeddings_to_array():
    df = pd.read_csv(config.embeddings_path, index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    df.head()
    return df


# K-Fold Cross-Validation
def cross_validation(model, _X, _y, _cv):
    """ Function to perform 5 Folds Cross-Validation
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

    train_accuracy = scores['test_accuracy']
    train_accuracy_mean = round(train_accuracy.mean(), 3)
    train_accuracy_std = round(train_accuracy.std(), 3)
    train_precision = scores['test_precision']
    train_precision_mean = round(train_precision.mean(), 3)
    train_precision_std = round(train_precision.std(), 3)
    train_recall = scores['test_recall']
    train_recall_mean = round(train_recall.mean(), 3)
    train_recall_std = round(train_recall.std(), 3)
    train_f1 = scores['test_f1_score']
    train_f1_mean = round(train_f1.mean(), 3)
    train_f1_std = round(train_f1.std(), 3)

    test_accuracy = scores['test_accuracy']
    test_accuracy_mean = round(test_accuracy.mean(), 3)
    test_precision = scores['test_precision']
    test_precision_mean = round(test_precision.mean(), 3)
    test_recall = scores['test_recall']
    test_recall_mean = round(test_recall.mean(), 3)
    test_f1 = scores['test_f1_score']
    test_f1_mean = round(test_f1.mean(), 3)

    return {'train_accuracy': train_accuracy,
            'train_accuracy_mean': train_accuracy_mean,
            'train_accuracy_std': train_accuracy_std,
            'train_precision': train_precision,
            'train_precision_mean': train_precision_mean,
            'train_precision_std': train_precision_std,
            'train_recall': train_recall,
            'train_recall_mean': train_recall_mean,
            'train_recall_std': train_recall_std,
            'train_f1': train_f1,
            'train_f1_mean': train_f1_mean,
            'train_f1_std': train_f1_std,

            'test_accuracy': test_accuracy,
            'test_accuracy_mean': test_accuracy_mean,
            'test_precision': test_precision,
            'test_precision_mean': test_precision_mean,
            'test_recall': test_recall,
            'test_recall_mean': test_recall_mean,
            'test_f1': test_f1,
            'test_f1_mean': test_f1_mean,
            }


# AD classification using linguistic features (embeddings) from transcribed speech
def classify_embedding(dataset, _n_splits):
    dataset = data_preprocessing(dataset)

    # Define the dependent variable that needs to be predicted (labels)
    y = dataset['dx'].values
    # Define the independent variable
    X = list(dataset['embeddings'].values)

    baseline_score = dummy_stratified_clf(X, y)
    log.info("Baseline performance: ", baseline_score)

    # Create models
    models = [SVC(), LogisticRegression(), RandomForestClassifier()]
    names = ['SVC', 'LR', 'RF']

    # Split the dataset into k equal partitions
    cv = KFold(n_splits=_n_splits, random_state=42, shuffle=True)

    # Prepare dataframe for results
    results_df = pd.DataFrame(columns=['Set', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1'])

    # Create the parameter grid
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

    log.info("Beginning to train models using GPT embeddings...")

    for model, name in zip(models, names):
        # It's not necessary to scale the GPT embeddings before using them.
        # They are already normalised and are in the vector space with a certain distribution.

        # Tune hyperparameters with GridSearchCV
        best_params = None
        if name == 'SVC':
            grid_search = GridSearchCV(estimator=model, param_grid=svc_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
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
        model.set_params(**best_params)

        # Perform cross-validation with custom scoring metrics and best params
        results = cross_validation(model, X, y, cv)
        results_df = results_to_df(name, results, results_df)

        visualize_results(_n_splits, name, results, config.embedding_results_dir)

    log.info("Training using GPT embeddings done.")

    # Adjust resulting dataframe
    results_df = results_df.sort_values(by='Set', ascending=False)
    results_df = results_df.reset_index(drop=True)

    # Save results to csv
    embedding_results_file = (config.embedding_results_dir / 'embedding_results.csv').resolve()
    results_df.to_csv(embedding_results_file)

    log.info(f"Writing {embedding_results_file}...")


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
                results["train_f1"],
                results["test_f1"],
                plot_f1_path)


def results_to_df(name, results, results_df):
    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Train',
                                                       'Model': name,
                                                       'Accuracy': f"{results['train_accuracy_mean']}"
                                                                   f"({results['train_accuracy_std']})",
                                                       'Precision': f"{results['train_precision_mean']} "
                                                                    f"({results['train_precision_std']})",
                                                       'Recall': f"{results['train_recall_mean']} "
                                                                 f"({results['train_recall_std']})",
                                                       'F1': f"{results['train_f1_mean']} "
                                                             f"({results['train_f1_std']})",
                                                       }])], ignore_index=True)

    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Test',
                                                       'Model': name,
                                                       'Accuracy': results['test_accuracy_mean'],
                                                       'Precision': results['test_precision_mean'],
                                                       'Recall': results['test_recall_mean'],
                                                       'F1': results['test_f1_mean']
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


def data_preprocessing(df):
    """ Preprocessing of the data """
    # Drop rows with NaN values in "embedding" or "dx" columns
    # TODO: There shouldn't be any empty entries in the first place
    df.dropna(subset=['embeddings', 'dx'], inplace=True)
    # Transform into binary classification
    df['dx'] = [1 if b == 'ad' else 0 for b in df.dx]
    # How many data points for each class?
    # print(df.dx.value_counts())
    # Understand the data
    # sns.countplot(x='dx', data=df)  # 1 - diagnosed   0 - control group

    ### Balance data by down-sampling majority class
    # Separate majority and minority classes
    df_majority = df[df.dx == 1]  # 87 ad datapoints
    df_minority = df[df.dx == 0]  # 79 cn datapoints
    # print(len(df_minority))
    # Undersample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority),  # to match minority class
                                       random_state=42)  # reproducible results

    # Combine undersampled majority class with minority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    # Display new class counts
    # print(df_downsampled.dx.value_counts())
    # sns.countplot(x='dx', data=df_downsampled)  # 1 - diagnosed   0 - control group
    plt.show()
    return df_downsampled


def dummy_stratified_clf(X, y):
    """
    DummyClassifier makes predictions that ignore the input features.

    This classifier serves as a simple baseline to compare against other more complex classifiers.
    It gives us a measure of “baseline” performance — i.e. the success rate one should expect to achieve even
    if simply guessing.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    stratified_clf = DummyClassifier(strategy='stratified').fit(X_train, y_train)

    return stratified_clf.score(X_test, y_test)



# AD classification using acoustic features from OpenSMILE
def classify_acoustic():
    dataset = data_preprocessing(dataset)

    file_label_dict = dict(zip(dataset["adressfname"], dataset["dx"]))

    # Create an instance of the openSMILE feature extractor
    smile_lowlevel = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    smile_functionals = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # Fetch the audio files
    audio_files = fetch_audio_files(config.diagnosis_train_data)

    # Create lists to hold the filenames, labels, and feature vectors
    filenames = []
    labels = []
    feature_vectors_lowlevel = []
    feature_vectors_functionals = []

    # Iterate through audio files and extract features
    for audio_file in audio_files:
        filename = Path(audio_file).stem
        label = file_label_dict.get(filename)

        # Check if the label is not None (i.e., the filename was found in the DataFrame)
        if label is not None:
            filenames.append(filename)
            labels.append(label)

            # Extract Low-Level Descriptors using openSMILE
            feature_vector_lowlevel = smile_lowlevel.process_file(audio_file)
            feature_vectors_lowlevel.append(feature_vector_lowlevel)

            # Extract Functionals features using openSMILE
            feature_vector_functionals = smile_functionals.process_file(audio_file)
            feature_vectors_functionals.append(feature_vector_functionals)

    # Combine Low-Level Descriptors and Functionals features
    feature_vectors_combined = [lowlevel + functionals for lowlevel, functionals in
                                zip(feature_vectors_lowlevel, feature_vectors_functionals)]

    # Define the dependent variable that needs to be predicted (labels)
    y = df['dx'].values
    # Define the independent variable
    X = feature_vectors_combined

    # Create models
    models = [SVC(), LogisticRegression(), RandomForestClassifier()]
    names = ['SVC', 'LR', 'RF']

    # Split the dataset into k equal partitions
    cv = KFold(n_splits=_n_splits, random_state=42, shuffle=True)

    # Prepare dataframe for results
    results_df = pd.DataFrame(columns=['Set', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1'])

    # Create the parameter grid
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

    log.info("Beginning to train models using acoustic embeddings...")

    for model, name in zip(models, names):
        # TODO: scaling?

        # Tune hyperparameters with GridSearchCV
        if name == 'SVC':
            grid_search = GridSearchCV(estimator=model, param_grid=svc_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
        elif name == 'LR':
            grid_search = GridSearchCV(estimator=model, param_grid=lr_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
        elif name == 'RF':
            grid_search = GridSearchCV(estimator=model, param_grid=rf_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        model.set_params(**best_params)

        # Perform cross-validation with custom scoring metrics and best params
        results = cross_validation(model, X, y, cv)
        results_df = results_to_df(name, results, results_df)

        visualize_results(_n_splits, name, results, config.acoustic_results_dir)

    log.info("Training using acoustic features done.")

    # Adjust resulting dataframe
    results_df = results_df.sort_values(by='Set', ascending=False)
    results_df = results_df.reset_index(drop=True)

    # Save results to csv
    acoustic_results_file = (config.acoustic_results_dir / 'acoustic_results.csv').resolve()
    results_df.to_csv(acoustic_results_file)

    log.info(f"Writing {acoustic_results_file}...")
