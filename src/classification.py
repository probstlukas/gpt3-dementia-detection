import config
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from openai.embeddings_utils import plot_multiclass_precision_recall
from matplotlib import pyplot as plt

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


# Helper function for specific classify functions
def classify(df):
    # Drop rows with NaN values in "embedding" or "dx" columns
    # TODO: There shouldn't be any empty entries in the first place
    df.dropna(subset=["embeddings", "dx"], inplace=True)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        list(df.embeddings.values), df.dx, test_size=0.3, random_state=42  # 70% training and 30% test
    )

    # Classify using different classifiers
    classify_svc(X_train, X_test, y_train, y_test)
    classify_lr(X_train, X_test, y_train, y_test)
    classify_rf(X_train, X_test, y_train, y_test)
    print("Classification done.")


def classify_svc(X_train, X_test, y_train, y_test):
    # Create a Classifier
    # TODO: Experiment with different kernels
    clf = SVC(kernel='linear')

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(config.svc_report_path, index=False)
    print(f"Writing {config.svc_report_path}...")


def classify_lr(X_train, X_test, y_train, y_test):
    # Create a Classifier
    clf = LogisticRegression(solver='liblinear', random_state=0)

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # TODO: Confusion matrix may be helpful as well
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(config.lr_report_path, index=False)
    print(f"Writing {config.lr_report_path}...")


def classify_rf(X_train, X_test, y_train, y_test):
    # Create a Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for the test dataset
    y_pred = clf.predict(X_test)

    # Get the class probabilities for the test dataset
    y_prob = clf.predict_proba(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(config.rf_report_path)
    print(f"Writing {config.rf_report_path}...")

    #plot_multiclass_precision_recall(y_prob, y_test, ['ad', 'cn'], clf)
    #plt.show()
