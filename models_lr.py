import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, linear_model, metrics

def main():
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
