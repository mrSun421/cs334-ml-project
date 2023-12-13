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

    print("Finding Best Hyperparameters for Decision Tree...")
    model = tree.DecisionTreeRegressor(max_depth=10)
    model_search = model_selection.RandomizedSearchCV(estimator=model,
                                                      param_distributions={"max_depth": np.arange(1, 10),
                                                                           "min_samples_leaf": np.arange(1, 20)}, verbose=4)
    model_search = model_search.fit(x_train, y_train)
    print(f"Best Parameters: {model_search.best_params_=}")
    dt_params = model_search.best_params_
    y_hat = model_search.predict(X=x_test)

    print("Calculating MSE (DT)...")
    # Regression task, so MSE is a reasonable choice
    mean_squared_error = metrics.mean_squared_error(
        y_true=y_test, y_pred=y_hat)
    print(f"Mean Squared Error = {mean_squared_error}")

    print("Printing Decision Tree....")
    fig = plt.figure(figsize=(25, 20))
    tree.plot_tree(model_search)
    fig.savefig("decistion_tree.png")

    print("plotting predicted and actual labels....")
    plt.scatter(np.arange(y_hat.shape[0]), y_hat)
    plt.show()
    plt.scatter(np.arange(y_test.shape[0]), y_test)
    plt.show()


if __name__ == "__main__":
    main()
