import sklearn.preprocessing as prepros
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


def main():
    LCD_data = pd.read_csv("./Dataset_Brunswick_LCD.csv", dtype=object)
    Hourly_data = pd.read_csv("./Dataset_Brunswick_Hourly.csv", dtype=object)

    
    """
    # Reduce data size to make training more managable
    data = data.iloc[70000:]
    """
    # data.to_csv("./Dataset_Dekalb_Processed.csv", index=False)

if __name__ == "__main__":
    main()
