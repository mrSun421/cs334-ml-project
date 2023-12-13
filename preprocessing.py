import sklearn.preprocessing as prepros
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

def main():
    LCD_data = pd.read_csv("./Dataset_Brunswick_LCD.csv", dtype=object)
    Hourly_data = pd.read_csv("./Dataset_Brunswick_Hourly.csv", dtype=object)

    """
    # Variables
    correlation_cutoff = 0.5
    """

    # Date Extraction
    LCD_data["DATE"] = pd.to_datetime(LCD_data["DATE"])
    LCD_data["YEAR"] = LCD_data["DATE"].dt.year
    LCD_data["MONTH"] = LCD_data["DATE"].dt.month
    LCD_data["DAY"] = LCD_data["DATE"].dt.day
    LCD_data["HOUR"] = LCD_data["DATE"].dt.hour
    LCD_data["MINUTE"] = LCD_data["DATE"].dt.minute
    LCD_data = LCD_data[LCD_data["MINUTE"] == 15]
    LCD_data = LCD_data.drop(columns=["DATE", "MINUTE"])

    """
    # Reduce data size to make training more managable
    data = data.iloc[70000:]
    """

    """
    # Calcluate and Plot Correlation Matrix
    cor = data.corr().abs()
    cor = cor.where(np.tril(np.ones(cor.shape)).astype(bool))
    sb.heatmap(cor, annot=True, cmap=plt.cm.Blues)
    plt.show()
    """

    '''
    # Drop Correlated Features
    to_drop = [column for column in cor.columns if any(cor[column] > correlation_cutoff)]
    data = data.drop(to_drop, axis=1)
    '''

    """
    # Normalize Data
    data_values = data.to_numpy()
    min_max_scaler = prepros.MinMaxScaler()
    data_values = min_max_scaler.fit_transform(data_values)
    data = pd.DataFrame(data_values)
    data.insert(0, "PERCIP", label, True)
    data = data.dropna()
    """

    data.to_csv("./Dataset_Dekalb_Processed.csv", index=False)

if __name__ == "__main__":
    main()
