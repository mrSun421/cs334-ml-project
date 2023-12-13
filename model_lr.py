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

    print("Fitting Multi-variate Linear Regression...")
    model = linear_model.LinearRegression()
    model = model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    mean_squared_error = metrics.mean_squared_error(
        y_true=y_test, y_pred=y_hat)
    r_2 = metrics.r2_score(y_true=y_test, y_pred=y_hat)
    print(f"Mean Squared Error:{mean_squared_error}")
    print(f"R^2:{r_2}")

    print("Plotting predicted and actual labels....")
    plt.scatter(y_test, y_hat)
    plt.show()


if __name__ == "__main__":
    main()
