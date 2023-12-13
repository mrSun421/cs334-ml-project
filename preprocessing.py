import sklearn.preprocessing as prepros
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


def main():
    LCD_data = pd.read_csv("./Dataset_Brunswick_LCD.csv", dtype=object)
    Hourly_data = pd.read_csv("./Dataset_Brunswick_Hourly.csv", dtype=object)

    # Date Extraction
    LCD_data["DATE"] = pd.to_datetime(LCD_data["DATE"])
    LCD_data["YEAR"] = LCD_data["DATE"].dt.year
    LCD_data["MONTH"] = LCD_data["DATE"].dt.month
    LCD_data["DAY"] = LCD_data["DATE"].dt.day
    LCD_data["HOUR"] = LCD_data["DATE"].dt.hour
    LCD_data["MINUTE"] = LCD_data["DATE"].dt.minute
    LCD_data = LCD_data[LCD_data["MINUTE"] == 15]
    LCD_data = LCD_data.drop(columns=["DATE", "MINUTE"])
    _ = 1

    # data.to_csv("./Dataset_Dekalb_Processed.csv", index=False)


if __name__ == "__main__":
    main()
