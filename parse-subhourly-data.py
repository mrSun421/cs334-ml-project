import pandas as pd
from pathlib import Path


def main():
    dataset_dir = Path("./dataset/subhourly/2023")
    headers = "WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE AIR_TEMPERATURE PRECIPITATION SOLAR_RADIATION SR_FLAG SURFACE_TEMPERATURE ST_TYPE ST_FLAG RELATIVE_HUMIDITY RH_FLAG SOIL_MOISTURE_5 SOIL_TEMPERATURE_5 WETNESS WET_FLAG WIND_1_5 WIND_FLAG".split()

    print("parsing files:")
    for filepath in dataset_dir.glob("**/*.txt"):
        print(filepath)
        with open(filepath) as datafile:
            lines = [line.rstrip() for line in datafile]
            split_lines = list(map(lambda line: line.split(), lines))
            df = pd.DataFrame(split_lines, columns=headers)
            csv_path = filepath.with_suffix(".csv")
            df.to_csv(csv_path)

    print("Making Mega CSV")
    filepaths = dataset_dir.glob("**/*.csv")
    data = pd.concat(map(lambda filepath: pd.read_csv(
        filepath_or_buffer=filepath, index_col=0), filepaths))

    data.to_csv("./Subhourly_Dataset_2023.csv", index=False)


if __name__ == "__main__":
    main()
