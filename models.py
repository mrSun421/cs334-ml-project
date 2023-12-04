import numpy as np
import pandas as pd
from sklearn import tree, svm, linear_model, model_selection, metrics

def model_training_eval(model, data):
    model.fit(data[0], data[2])

    metrics.accuracy_score(data[3], model.predict(data[1]))
    

def main():
    data = pd.read_csv("./Dataset_2023_processed.csv")

    label = data['P_CALC'].copy()
    data = data.drop(['P_CALC'], axis=1)
    data = model_selection.train_test_split(data, label, test_size=0.2)




if __name__ == "__main__":
    main()