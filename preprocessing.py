import sklearn.preprocessing as prepros
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


def main():
    LCD_data = pd.read_csv("./Dataset_Brunswick_LCD.csv", dtype=object)
    Hourly_data = pd.read_csv("./Dataset_Brunswick_Hourly.csv", dtype=object)

    # Determine Bad Data in Hourly Dataset Based on Flag and -9999
    flag_list = ['SOLARAD_FLAG',
                 'SOLARAD_FLAG',
                 'SOLARAD_MAX_FLAG',
                 'SOLARAD_MIN_FLAG',
                 'SUR_TEMP_FLAG',
                 'SUR_TEMP_MAX_FLAG',
                 'SUR_TEMP_MIN_FLAG']

    flagged_indicies = None
    for flag in flag_list:
        flagged = np.where(hourly_data[flag] > 0)[0]
        flagged_indicies = flagged if flagged_indicies is None else np.concatenate([flagged_indicies, flagged])

    flagged_indicies = np.concatenate([np.where(hourly_data['P_CALC'] < 0)[0], flagged_indicies])
    flagged_indicies = np.unique(flagged_indicies)

    # Date Extraction
    LCD_data["DATE"] = pd.to_datetime(LCD_data["DATE"])
    LCD_data["YEAR"] = LCD_data["DATE"].dt.year
    LCD_data["MONTH"] = LCD_data["DATE"].dt.month
    LCD_data["DAY"] = LCD_data["DATE"].dt.day
    LCD_data["HOUR"] = LCD_data["DATE"].dt.hour
    LCD_data["MINUTE"] = LCD_data["DATE"].dt.minute
    LCD_data = LCD_data[LCD_data["MINUTE"] == 15]
    LCD_data = LCD_data.drop(columns=["MINUTE"])

    """
    # Reduce data size to make training more managable
    data = data.iloc[70000:]
    """
    # data.to_csv("./Dataset_Dekalb_Processed.csv", index=False)

if __name__ == "__main__":
    main()
