import sklearn.preprocessing as prepros
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = pd.read_csv("./Hourly_Dataset_2023.csv")

    correlation_cutoff = 0.5

    # Use flags to elimiate bad data
    data = data.drop(data[data['SOLARAD_FLAG'] > 0].index)
    data = data.drop(data[data['SOLARAD_MAX_FLAG'] > 0].index)
    data = data.drop(data[data['SOLARAD_MIN_FLAG'] > 0].index)
    data = data.drop(data[data['SUR_TEMP_FLAG'] > 0].index)
    data = data.drop(data[data['SUR_TEMP_MAX_FLAG'] > 0].index)
    data = data.drop(data[data['SUR_TEMP_MIN_FLAG'] > 0].index)

    # Drop -9999 labels
    data = data.drop(data[data['P_CALC'] < 0].index)

    # Reduce data size to make training more managable
    # data = data.iloc[70000:]

    # Drop Flags, Dates, Station Identifiers, and Dummy columns
    data = data.drop(data.columns[[0, 1]], axis=1)
    data = data.drop(columns=['WBANNO',
                              'UTC_DATE',
                              'UTC_TIME',
                              'LST_DATE',
                              'LST_TIME',
                              'CRX_VN',
                              'SOLARAD_FLAG',
                              'SOLARAD_MAX_FLAG',
                              'SOLARAD_MIN_FLAG',
                              'SUR_TEMP_TYPE',
                              'SUR_TEMP_FLAG',
                              'SUR_TEMP_MAX_FLAG',
                              'SUR_TEMP_MIN_FLAG',
                              'RH_HR_AVG_FLAG'
                              ])

    # Remove Label for Pearson Correlation
    label = data['P_CALC'].copy()
    data = data.drop(columns=['P_CALC'])

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

    # Normalize Data
    data_values = data.to_numpy()
    min_max_scaler = prepros.MinMaxScaler()
    data_values = min_max_scaler.fit_transform(data_values)
    data = pd.DataFrame(data_values)
    data.insert(0, "PERCIP", label, True)
    data = data.dropna()

    data.to_csv("./Dataset_2023_Processed.csv", index=False)


if __name__ == "__main__":
    main()
