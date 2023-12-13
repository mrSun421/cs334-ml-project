from pathlib import Path
import numpy as np
import pandas as pd


def main():
    dataset_dir = Path("./dataset/Brunswick")

    # Collect LCD Brunswick Data into Single CSV
    filepaths = dataset_dir.glob("**/*.csv")
    LCD_data = pd.concat(map(lambda filepath: pd.read_csv(
        filepath_or_buffer=filepath, index_col=None), filepaths))
    # Drop Station Identifiers and Dummy columns
    LCD_data = LCD_data.drop(
        LCD_data.iloc[:, np.arange(24, len(LCD_data.columns))], axis=1)
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
    hourly_data = pd.concat(map(lambda filepath: pd.read_csv(
        filepath_or_buffer=filepath, index_col=None), filepaths))

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
        flagged_indicies = flagged if flagged_indicies is None else np.concatenate(
            [flagged_indicies, flagged])

    flagged_indicies = np.concatenate(
        [np.where(hourly_data['P_CALC'] < 0)[0], flagged_indicies])
    flagged_indicies = np.unique(flagged_indicies)

    # Date Extraction
    LCD_data["DATE"] = pd.to_datetime(LCD_data["DATE"])

    LCD_data["MINUTE"] = LCD_data["DATE"].dt.minute
    LCD_data = LCD_data[LCD_data["MINUTE"] == 15]
    LCD_data["DATE"] = LCD_data["DATE"].apply(lambda dt: dt.replace(minute=0))
    LCD_data = LCD_data.drop(columns=["MINUTE"])

    hourly_data["UTC_TIME"] = hourly_data["UTC_TIME"].astype(str)
    hourly_data["UTC_TIME"] = hourly_data["UTC_TIME"].map(lambda x: x.zfill(4))
    hourly_data["DATE"] = hourly_data["UTC_DATE"].astype(
        str) + hourly_data["UTC_TIME"]
    hourly_data["DATE"] = pd.to_datetime(
        hourly_data["DATE"], format="%Y%m%d%H%M")
    hourly_data = hourly_data.drop(columns=["UTC_DATE", "UTC_TIME"])

    full_data = hourly_data.merge(LCD_data, on=["DATE"])

    full_data["YEAR"] = full_data["DATE"].dt.year
    full_data["MONTH"] = full_data["DATE"].dt.month
    full_data["DAY"] = full_data["DATE"].dt.day
    full_data["HOUR"] = full_data["DATE"].dt.hour

    print(full_data)
    # Dropping with Seasonal Regularity
    years = []
    for index in flagged_indicies:
        hour = full_data.iloc[index]['HOUR']
        date = full_data.iloc[index]['DATE']
        month = full_data.iloc[index]['MONTH']
        year = full_data.iloc[index]['YEAR']

        temp_years = years.remove(year)
        year = np.random.choice(temp_years, 1)

        full_data.iloc[index] = full_data[(full_data['YEAR'] == year) & (full_data['MONTH'] == month) & (
            # What if EVERY year bad data on given MDH?
            full_data['DATE'] == date) & (full_data['HOUR'] == hour)]


if __name__ == "__main__":
    main()
