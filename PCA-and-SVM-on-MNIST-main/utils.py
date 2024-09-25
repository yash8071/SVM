import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    x_tr_min = X_train.min()
    x_tr_max = X_train.max()
    X_train = 2*((X_train-x_tr_min)/(x_tr_max-x_tr_min)) - 1
    x_te_min = X_test.min()
    x_te_max = X_test.max()
    X_test = 2*((X_test-x_te_min)/(x_te_max-x_te_min)) - 1
    return X_train,X_test
    raise NotImplementedError


def plot_metrics(metrics) -> None:
    k = []
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    for i in range(len(metrics)):
        k.append(metrics[i][0])
        accuracy.append(metrics[i][1])
        precision.append(metrics[i][2])
        recall.append(metrics[i][3])
        f1_score.append(metrics[i][4])
    plt.plot(k,accuracy)
    plt.xlabel("Components")
    plt.ylabel("Accuracy")
    plt.title("Components vs Accuracy")
    plt.savefig("k_accuracy.jpg")
    plt.show()
    plt.plot(k,precision)
    plt.xlabel("Components")
    plt.ylabel("Precision")
    plt.title("Components vs Precision")
    plt.savefig("k_precision.jpg")
    plt.show()
    plt.plot(k,recall)
    plt.xlabel("Components")
    plt.ylabel("Recall")
    plt.title("components vs Recall")
    plt.savefig("k_recall.jpg")
    plt.show()
    plt.plot(k,f1_score)
    plt.xlabel("Components")
    plt.ylabel("F1_score")
    plt.title("Components vs F1_score")
    plt.savefig("k_f1_score.jpg")
    plt.show()
    # raise NotImplementedError