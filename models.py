import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree, svm, linear_model, model_selection, metrics, neighbors

def main():
    data = pd.read_csv("./Dataset_2023_processed.csv")

    print("Converting Data to Numpy...")
    # convert the data into numpy arrays to make sure index data doesn't corrupt it
    # NOTE: data[0] == data['PERCIP'] form earlier because of changes to preprocessing col name lost
    label = data[data.columns[0]].copy()
    label_np = label.to_numpy()
    data = data.drop([data.columns[0]], axis=1)
    data_np = data.to_numpy()

    print("Splitting to Train-Test Split...")
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(
        data_np, label_np, test_size=0.2)

    print("linear regression")
    model = linear_model.LinearRegression()
    model = model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    meanSquaredError = metrics.mean_squared_error(y_true=yTest, y_pred=yHat)
    r_2 = metrics.r2_score(y_true=yTest, y_pred=yHat)
    print(f"{r_2}")



    """
    model = neighbors.KNeighborsRegressor(n_neighbors=30)
    print("Finding Best Hyperparameters for KNN Regressor...")
    model_search = model_selection.GridSearchCV(estimator=model,
                                                      param_grid={
                                                          'n_neighbors':np.arange(1, 3)})
    model_search = model_search.fit(xTrain, yTrain)
    print(f"{search.best_params_=}")
    yHat = model_search.predict(X=xTest)
    """

    """
    model = model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    """

    """
    print("Calculating MSE (KNN)...")
    # Regression task, so MSE is a reasonable choice
    meanSquaredError = metrics.mean_squared_error(y_true=yTest, y_pred=yHat)
    print(f"{meanSquaredError}")

    model = tree.DecisionTreeRegressor(max_depth=10)
    """
    """
    # replace with different models and parameters to your choice!
    print("Finding Best Hyperparameters for Decision Tree...")
    model_search = model_selection.RandomizedSearchCV(estimator=model, param_distributions={
        "max_depth": np.arange(1, 10), "min_samples_leaf": np.arange(1, 20)})
    model_search = model_search.fit(xTrain, yTrain)
    print(f"{search.best_params_=}")
    yHat = model_search.predict(X=xTest)
    """

    """
    model = model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    """
    """
    print("Calculating MSE (DT)...")
    # Regression task, so MSE is a reasonable choice
    meanSquaredError = metrics.mean_squared_error(y_true=yTest, y_pred=yHat)
    print(f"{meanSquaredError}")
    """
    """
    print("Printing Decision Tree....")
    fig = plt.figure(figsize=(25,20))
    plot = tree.plot_tree(model)
    fig.savefig("decistion_tree.png")
    """

    """
    print("Finding best Hyperparameters for Logistic Regression....")
    model = linear_model.LogisticRegression()
    clf = model_selection.RandomizedSearchCV(estimator=model,
                                             param_distributions={'penalty':('l2',
                                                                             'elastinet'),
                                                                  'fit_intercept':(True, False),
                                                                  'max_iter':np.arange(100,500)})
    search = clf.fit(xTrain, yTrain)
    print(f"{search.best_params_=}")
    yHat = search.predict(X=xTest)                                                              
    """

    """
    model = svm.SVR(kernel='linear')
    model = model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    meanSquaredError = metrics.mean_squared_error(y_true=yTest, y_pred=yHat)
    print(f"Linear SVR MSE: {meanSquaredError}")

    model = svm.SVR(kernel='poly')
    model = model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    meanSquaredError = metrics.mean_squared_error(y_true=yTest, y_pred=yHat)
    print(f"Poly SVR MSE: {meanSquaredError}")
    """

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

if __name__ == "__main__":
    main()
