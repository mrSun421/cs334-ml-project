from pathlib import Path
import numpy as np
import pandas as pd

def main():
    dataset_dir = Path("./dataset/Brunswick")

    # Collect LCD Brunswick Data into Single CSV
    filepaths = dataset_dir.glob("**/*.csv")
    LCD_data = pd.concat(map(lambda filepath: pd.read_csv(filepath_or_buffer=filepath), filepaths))
    # Drop Station Identifiers and Dummy columns
    LCD_data = LCD_data.drop(LCD_data.iloc[:, np.arange(24, len(LCD_data.columns))], axis=1) 
    LCD_data = LCD_data.drop(columns=['STATION',
                                      'LATITUDE',
                                      'LONGITUDE',
                                      'ELEVATION',
                                      'NAME',
                                      'SOURCE',
                                      'HourlyPrecipitation',
                                      'HourlyPresentWeatherType',
                                      'HourlyPressureChange',
                                      'HourlyPressureTendency',
                                      'HourlySeaLevelPressure',
                                      'HourlyWindGustSpeed'])


    # LCD_data.to_csv("./Dataset_Brunswick_LCD.csv", index=False)


    dataset_dir = Path("./dataset/")
    filepaths = dataset_dir.glob("**/*Brunswick*.csv")
    hourly_data = pd.concat(map(lambda filepath: pd.read_csv(filepath_or_buffer=filepath), filepaths))

    # Drop Flags, Dates, Station Identifiers, and Dummy columns
    hourly_data = hourly_data.drop(columns=['WBANNO',
                                            'LST_DATE',
                                            'LST_TIME',
                                            'CRX_VN',
                                            'SUR_TEMP_TYPE'])

    # hourly_data.to_csv("./Dataset_Brunswick_Hourly.csv", index=False)

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


    # Dropping with Seasonal Regularity
    for index in flagged_indices:
        hour = 


if __name__ == "__main__":
    main()
