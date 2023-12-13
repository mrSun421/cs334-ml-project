import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree, svm, linear_model, model_selection, metrics, neighbors, ensemble
import helper_functions


def main():
    data_np, label_np = helper_functions.dataframe_to_numpy_label_and_features(
        "./Hourly_Dataset_2023_processed.csv")

    print("Splitting to Train-Test Split...")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data_np, label_np, test_size=0.2)

    # Reduce iterations
    print("Training SVR with linear kernel:")
    x_train_SVR = x_train[:int(x_train.shape[0] * 0.1)][:]
    y_train_SVR = y_train[:int(y_train.len() * 0.1)][:]
    model = svm.SVR(kernel='linear')
    model = model.fit(x_train_SVR, y_train_SVR)
    y_hat = model.predict(x_test)
    print("Calculating MSE (SVR Linear):")
    mean_squared_error = metrics.mean_squared_error(
        y_true=y_test, y_pred=y_hat)
    print(f"Linear SVR MSE: {mean_squared_error}")

    print("Training SVR with linear kernel...")
    model = svm.SVR(kernel='poly')
    model = model.fit(x_train_SVR, y_train_SVR)
    y_hat = model.predict(x_test)
    mean_squared_error = metrics.mean_squared_error(
        y_true=y_test, y_pred=y_hat)
    print(f"Poly SVR MSE: {mean_squared_error}")


if __name__ == "__main__":
    main()
