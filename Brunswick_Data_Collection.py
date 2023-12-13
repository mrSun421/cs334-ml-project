from pathlib import Path
import pandas as pd

def main():
    dataset_dir = Path("./dataset/Brunswick")

    """
    # Collect LCD Brunswick Data into Single CSV
    filepaths = dataset_dir.glob("**/*.csv")
    LCD_data = pd.concat(map(lambda filepath: pd.read_csv(filepath_or_buffer=filepath), filepaths))
    """
    # Drop Station Identifiers and Dummy columns
    LCD_data = LCD_data.drop(LCD_data.columns[np.arange(24, len(LCD_data.columns))]) 
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

    # Use flags to elimiate bad data
    hourly_data = hourly_data.drop(hourly_data[hourly_data['SOLARAD_FLAG'] > 0].index)
    hourly_data = hourly_data.drop(hourly_data[hourly_data['SOLARAD_MAX_FLAG'] > 0].index)
    hourly_data = hourly_data.drop(hourly_data[hourly_data['SOLARAD_MIN_FLAG'] > 0].index)
    hourly_data = hourly_data.drop(hourly_data[hourly_data['SUR_TEMP_FLAG'] > 0].index)
    hourly_data = hourly_data.drop(hourly_data[hourly_data['SUR_TEMP_MAX_FLAG'] > 0].index)
    hourly_data = hourly_data.drop(hourly_data[hourly_data['SUR_TEMP_MIN_FLAG'] > 0].index)

    # Drop Flags, Dates, Station Identifiers, and Dummy columns
    hourly_data = data.drop(columns=['WBANNO',
                                     'UTC_DATE',
                                     'UTC_TIME',
                                     'CRX_VN',
                                     'SOLARAD_FLAG',
                                     'SOLARAD_MAX_FLAG',
                                     'SOLARAD_MIN_FLAG',
                                     'SUR_TEMP_TYPE',
                                     'SUR_TEMP_FLAG',
                                     'SUR_TEMP_MAX_FLAG',
                                     'SUR_TEMP_MIN_FLAG',
                                     'RH_HR_AVG_FLAG'])

    hourly_data.to_csv("./Dataset_Brunswick_Hourly.csv", index=False)

if __name__ == "__main__":
    main()
