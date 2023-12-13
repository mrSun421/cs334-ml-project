import pandas as pd


def dataframe_to_numpy_label_and_features(path):
    data = pd.read_csv("./Hourly_Dataset_2023_processed.csv")

    print("Converting Data to Numpy...")
    label = data[data.columns[0]].copy()
    label_np = label.to_numpy()
    data = data.drop([data.columns[0]], axis=1)
    data_np = data.to_numpy()
    return data_np, label_np
