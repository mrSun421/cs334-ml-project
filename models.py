import numpy as np
import pandas as pd
from sklearn import tree, svm, linear_model, model_selection, metrics

# def model_training_eval(model, xTrain, xTest, yTrain, yTest):
#     model.fit(xTrain, yTrain)
#     yHat = model.predict(xTest)
#     metrics.accuracy_score(y_true=yTest, y_pred=yHat)
    

def main():
    data = pd.read_csv("./Dataset_2023_processed.csv")

    label = data['PERCIP'].copy()
    data = data.drop(['PERCIP'], axis=1)
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(data, label, test_size=0.2)
    model = tree.DecisionTreeRegressor()
    model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    meanSquaredError = metrics.mean_squared_error(y_true=yTest, y_pred=yHat)
    print(meanSquaredError)





if __name__ == "__main__":
    main()