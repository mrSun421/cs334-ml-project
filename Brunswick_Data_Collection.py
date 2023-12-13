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


    LCD_data.to_csv("./Dataset_Brunswick_LCD.csv", index=False)


    dataset_dir = Path("./dataset/")
    filepaths = dataset_dir.glob("**/*Brunswick*.csv")
    hourly_data = pd.concat(map(lambda filepath: pd.read_csv(filepath_or_buffer=filepath), filepaths))

    # Drop Flags, Dates, Station Identifiers, and Dummy columns
    hourly_data = hourly_data.drop(columns=['WBANNO',
                                            'UTC_DATE',
                                            'UTC_TIME',
                                            'CRX_VN',
                                            'SUR_TEMP_TYPE'])

    hourly_data.to_csv("./Dataset_Brunswick_Hourly.csv", index=False)

if __name__ == "__main__":
    main()
