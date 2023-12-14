import numpy as np
from sklearn import model_selection, metrics, neighbors 

def main():
    print("Finding Best Hyperparameters for KNN Regressor...")
    model = neighbors.KNeighborsRegressor(n_jobs=100)
    # Coarse then fine search
    k_coarse_values = np.arange(1, 100, 10)
    model_search = model_selection.GridSearchCV(estimator=model,
                                                param_grid={'n_neighbors':k_coarse_values})
    model_search = model_search.fit(x_train, y_train)
    print(f"Coarse Search Result: {model_search.best_params_=}")

    k_fine_values = np.arange(model_search.best_params_['n_neighbors'] - 5, 10)
    model_search = model_selection.GridSearchCV(estimator=model,
                                                param_grid={'n_neighbors':k_fine_values})
    model_search = model_search.fit(x_train, y_train)
    print(f"Fine Search Result: {model_search.best_params_=}")

    model = neighbors.KNeighborsRegressor(n_neighbors=model_search.best_params_['n_neighbors'],
                                          n_jobs=100)
    model_search = model_selection.RandomizedSearchCV(estimator=model,
                                                param_distributions={'weights':['uniform','distance'],
                                                                     'algorithm':['auto',
                                                                                  'ball_tree',
                                                                                  'kd_tree',
                                                                                  'brute'],
                                                                     'p':[1, 2]})
    model_search = model_search.fit(x_train, y_train)
    print(f"Other Parameter Results: {model_search.best_params_=}")
    # We can also tune leafsize

    print("Calculating MSE (KNN)...")
    y_hat = model_search.predict(X=x_test)
    mean_squared_error = metrics.mean_squared_error(
    y_true=y_test, y_pred=y_hat)
    print(f"Mean Squared Error: {mean_squared_error}")

