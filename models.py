import numpy as np
import pandas as pd
from sklearn import tree, svm, linear_model, model_selection, metrics

# def model_training_eval(model, xTrain, xTest, yTrain, yTest):
#     model.fit(xTrain, yTrain)
#     yHat = model.predict(xTest)
#     metrics.accuracy_score(y_true=yTest, y_pred=yHat)


def main():
    data = pd.read_csv("./Dataset_2023_processed.csv")

<<<<<<< HEAD
    label = data['PERCIP'].copy()
    data = data.drop(['PERCIP'], axis=1)
    data = model_selection.train_test_split(data, label, test_size=0.2)


=======
    print("Converting Data to Numpy...")
    # convert the data into numpy arrays to make sure index data doesn't corrupt it
    label = data['PERCIP'].copy()
    label_np = label.to_numpy()
    data = data.drop(['PERCIP'], axis=1)
    data_np = data.to_numpy()

    print("Splitting to Train-Test Split...")
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(
        data_np, label_np, test_size=0.2)

    print("Finding Best Hyperparameters for model...")
    # replace with different models and parameters to your choice!
    model = tree.DecisionTreeRegressor()
    clf = model_selection.RandomizedSearchCV(estimator=model, param_distributions={
        "max_depth": np.arange(1, 10), "min_samples_leaf": np.arange(1, 20)})
    search = clf.fit(xTrain, yTrain)
    print(f"{search.best_params_=}")
    yHat = search.predict(X=xTest)

    print("Calculating MSE...")
    # Regression task, so MSE is a reasonable choice
    meanSquaredError = metrics.mean_squared_error(y_true=yTest, y_pred=yHat)
    print(f"{meanSquaredError}")
>>>>>>> origin/JoshBranch


if __name__ == "__main__":
    main()
