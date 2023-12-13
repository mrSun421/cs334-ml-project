import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, metrics, neighbors
import helper_functions


def main():
    data_np, label_np = helper_functions.dataframe_to_numpy_label_and_features(
        "./Hourly_Dataset_2023_processed.csv")

    print("Splitting to Train-Test Split...")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data_np, label_np, test_size=0.2)

    print("Finding Best Hyperparameters for KNN Regressor...")
    model = neighbors.KNeighborsRegressor()
    # Coarse then fine search
    model_search = model_selection.RandomizedSearchCV(estimator=model,
                                                      param_distributions={'n_neighbors': np.arange(1, 100),
                                                                           'p': [1, 2],
                                                                           'weights': ['uniform', 'distance']}, verbose=4)
    model_search = model_search.fit(x_train, y_train)
    print(f"Best Parameters: {model_search.best_params_=}")
    y_hat = model_search.predict(X=x_test)

    print("Calculating MSE (KNN)...")
    mean_squared_error = metrics.mean_squared_error(
        y_true=y_test, y_pred=y_hat)
    print(f"Mean Squared Error: {mean_squared_error}")

    print("plotting predicted and actual labels....")
    plt.scatter(np.arange(y_hat.shape[0]), y_hat)
    plt.show()
    plt.scatter(np.arange(y_test.shape[0]), y_test)
    plt.show()


if __name__ == "__main__":
    main()
