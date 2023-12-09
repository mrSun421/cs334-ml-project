import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree, svm, linear_model, model_selection, metrics, neighbors, ensemble

def main():
    data = pd.read_csv("./Dataset_2023_processed.csv")


    print("Converting Data to Numpy...")
    label = data[data.columns[0]].copy()
    label_np = label.to_numpy()
    data = data.drop([data.columns[0]], axis=1)
    data_np = data.to_numpy()


    print("Splitting to Train-Test Split...")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data_np, label_np, test_size=0.2)


    print("Fitting Multi-variate Linear Regression...")
    model = linear_model.LinearRegression()
    model = model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    mean_squared_error = metrics.mean_squared_error(y_true=y_test, y_pred=y_hat)
    r_2 = metrics.r2_score(y_true=y_test, y_pred=y_hat)
    print(f"Mean Squared Error:{mean_squared_error}")
    print(f"R^2:{r_2}")


    print("Plotting predicted and actual labels....")
    plt.scatter(np.arange(y_hat.shape[0]), y_hat)
    plt.show()
    plt.scatter(np.arange(y_test.shape[0]), y_test)
    plt.show()


    print("Finding Best Hyperparameters for KNN Regressor...")
    model = neighbors.KNeighborsRegressor()
    model_search = model_selection.RandomizedSearchCV(estimator=model,
                                                      param_distributions={'n_neighbors': np.arange(1, 100),
                                                                           'p': [1, 2],
                                                                           'weights': ['uniform', 'distance']})
    model_search = model_search.fit(x_train, y_train)
    print(f"Best Parameters: {model_search.best_params_=}")
    y_hat = model_search.predict(X=x_test)


    print("Calculating MSE (KNN)...")
    mean_squared_error  = metrics.mean_squared_error(y_true=y_test, y_pred=y_hat)
    print(f"Mean Squared Error: {mean_squared_error}")


    print("plotting predicted and actual labels....")
    plt.scatter(np.arange(y_hat.shape[0]), y_hat)
    plt.show()
    plt.scatter(np.arange(y_test.shape[0]), y_test)
    plt.show()


    print("Finding Best Hyperparameters for Decision Tree...")
    model = tree.DecisionTreeRegressor(max_depth=10)
    model_search = model_selection.RandomizedSearchCV(estimator=model,
                                                      param_distributions={"max_depth": np.arange(1, 10),
                                                                           "min_samples_leaf": np.arange(1, 20)})
    model_search = model_search.fit(x_train, y_train)
    print(f"Best Parameters: {model_search.best_params_=}")
    dt_params = model_search.best_params_
    y_hat = model_search.predict(X=x_test)


    print("Calculating MSE (DT)...")
    # Regression task, so MSE is a reasonable choice
    mean_squared_error = metrics.mean_squared_error(y_true=y_test, y_pred=y_hat)
    print(f"Mean Squared Error = {mean_squared_error}")

    print("Printing Decision Tree....")
    fig = plt.figure(figsize=(25,20))
    tree.plot_tree(model_search)
    fig.savefig("decistion_tree.png")


    print("plotting predicted and actual labels....")
    plt.scatter(np.arange(y_hat.shape[0]), y_hat)
    plt.show()
    plt.scatter(np.arange(y_test.shape[0]), y_test)
    plt.show()

    print("Training SVR with linear kernel:")
    x_train_SVR = x_train[:int(x_train.shape[0] * 0.1)][:]
    y_train_SVR = y_train[:int(y_train.len() * 0.1)][:]
    model = svm.SVR(kernel='linear')
    model = model.fit(x_train_SVR, y_train_SVR)
    y_hat = model.predict(x_test)
    print("Calculating MSE (SVR Linear):")
    mean_squared_error = metrics.mean_squared_error(y_true=y_test, y_pred=y_hat)
    print(f"Linear SVR MSE: {mean_squared_error}")


    print("Training SVR with linear kernel...")
    model = svm.SVR(kernel='poly')
    model = model.fit(x_train_SVR, y_train_SVR)
    y_hat = model.predict(x_test)
    mean_squared_error = metrics.mean_squared_error(y_true=y_test, y_pred=y_hat)
    print(f"Poly SVR MSE: {mean_squared_error}")


    """
    print("Finding Best Hyperparameters for Support Vector Machine...")
    # replace with different models and parameters to your choice!
    clf = model_selection.RandomizedSearchCV(estimator=model, 
                                             param_distributions={'kernel':['linear',
                                                                            'poly'],
                                                                  'degree':np.arange(0,3)})
    search = clf.fit(xTrain, yTrain)
    print(f"{search.best_params_=}")
    yHat = search.predict(X=xTest)

    print("Calculating MSE (SVM)...")
    # Regression task, so MSE is a reasonable choice
    meanSquaredError = metrics.mean_squared_error(y_true=yTest, y_pred=yHat)
    print(f"{meanSquaredError}")
    """

    print("Training Random Forest...")
    model = ensemble.RandomForestRegressor(max_depth=dt_params['max_depth'],
                                           min_samples_leaf=dt_params['min_samples_leaf'])
    model = model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    mean_squared_error = metrics.mean_squared_error(y_true=y_test, y_pred=y_hat)
    print(f"Mean Squared Error:{mean_squared_error}")


    print("Plotting predicted and actual labels....")
    plt.scatter(np.arange(y_hat.shape[0]), y_hat)
    plt.show()
    plt.scatter(np.arange(y_test.shape[0]), y_test)
    plt.show()

if __name__ == "__main__":
    main()
