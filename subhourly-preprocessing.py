import sklearn.preprocessing as prepros
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = pd.read_csv("./Subhourly_Dataset_2023.csv", index_col=False)

    correlation_cutoff = 0.5
    data = data.drop(columns=[
        "SOIL_MOISTURE_5",
        "SOIL_TEMPERATURE_5",
    ])

    # Use flags to elimiate bad data
    # Drop any Missing data: From documentation
    logical_filter = ((data["SR_FLAG"] > 0)
                      | (data["ST_FLAG"] > 0)
                      | (data["RH_FLAG"] > 0)
                      | (data["WET_FLAG"] > 0)
                      | (data["WIND_FLAG"] > 0)
                      | (data["ST_TYPE"] != "C")
                      | (data["PRECIPITATION"] < -9998)
                      )
    data = data.drop(data[logical_filter].index)

    # Reduce data size to make training more managable
    # data = data.iloc[70000:]

    # Extract time data
    data["UTC_TIME"] = data["UTC_TIME"].astype(str)
    data["UTC_TIME"] = data["UTC_TIME"].map(lambda x: x.zfill(4))
    data["UTC"] = data["UTC_DATE"].astype(str) + data["UTC_TIME"]
    data["UTC"] = pd.to_datetime(data["UTC"], format="%Y%m%d%H%M")
    data["YEAR"] = data["UTC"].dt.year
    data["MONTH"] = data["UTC"].dt.month
    data["DAY"] = data["UTC"].dt.day
    data["HOUR"] = data["UTC"].dt.hour
    data["MINUTE"] = data["UTC"].dt.minute

    data = data.drop(columns="UTC")
    # Drop Flags, Dates, Station Identifiers, and Dummy columns
    data = data.drop(columns=['WBANNO',
                              'UTC_DATE',
                              'UTC_TIME',
                              'LST_DATE',
                              'LST_TIME',
                              'CRX_VN',
                              "SR_FLAG",
                              "ST_FLAG",
                              "RH_FLAG",
                              "WET_FLAG",
                              "WIND_FLAG",
                              "ST_TYPE",
                              ])

    # Normalize Data
    min_max_scaler = prepros.MinMaxScaler()
    data[data.columns] = min_max_scaler.fit_transform(data[data.columns])
    data = data.dropna()

   # Calcluate and Plot Correlation Matrix
    cor = data.corr().abs()
    cor = cor.where(np.tril(np.ones(cor.shape)).astype(bool))
    sb.heatmap(cor, annot=True, cmap=plt.cm.Blues)
    plt.show()

    # Drop Correlated Features
    '''
    to_drop = [column for column in cor.columns if any(cor[column] > correlation_cutoff)]
    data = data.drop(to_drop, axis=1)
    '''

    data.to_csv("./Hourly_Dataset_2023_Processed.csv", index=False)


if __name__ == "__main__":
    main()
