from utils import get_data, plot_metrics, normalize
from model import MultiClassSVM, PCA
from typing import Tuple


def get_hyperparameters() -> Tuple[float, int, float]:
    # get the hyperparameters
    learning_rate = 0.0001  #0.001 less acc(84%)
    num_iters = 270000  # 240000
    C = 10  #10
    return learning_rate,num_iters,C
    raise NotImplementedError


def main() -> None:
    # hyperparameters
    learning_rate, num_iters, C = get_hyperparameters()

    # get data
    X_train, X_test, y_train, y_test = get_data()

    # normalize the data
    X_train, X_test = normalize(X_train, X_test)

    metrics = []
    for k in [5, 10, 20, 50, 100, 200, 500]:    # 
        # reduce the dimensionality of the data
        pca = PCA(n_components=k)
        X_train_emb = pca.fit_transform(X_train)
        X_test_emb = pca.transform(X_test)

        # create a model
        svm = MultiClassSVM(num_classes=10)

        # fit the model
        svm.fit(
            X_train_emb, y_train, C=C,
            learning_rate=learning_rate,
            num_iters=num_iters,
        )

        # evaluate the model
        accuracy = svm.accuracy_score(X_test_emb, y_test)
        precision = svm.precision_score(X_test_emb, y_test)
        recall = svm.recall_score(X_test_emb, y_test)
        f1_score = svm.f1_score(X_test_emb, y_test)

        metrics.append((k, accuracy, precision, recall, f1_score))

        print(f'k={k}, accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_score}')

    # plot and save the results
    plot_metrics(metrics)


if __name__ == '__main__':
    main()
