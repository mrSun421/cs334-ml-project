import pandas as pd
from pathlib import Path

def main():
    dataset_dir = Path("./dataset/2023")
    headers = "WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE T_CALC T_HR_AVG T_MAX T_MIN P_CALC SOLARAD SOLARAD_FLAG SOLARAD_MAX SOLARAD_MAX_FLAG SOLARAD_MIN SOLARAD_MIN_FLAG SUR_TEMP_TYPE SUR_TEMP SUR_TEMP_FLAG SUR_TEMP_MAX SUR_TEMP_MAX_FLAG SUR_TEMP_MIN SUR_TEMP_MIN_FLAG RH_HR_AVG RH_HR_AVG_FLAG SOIL_MOISTURE_5 SOIL_MOISTURE_10 SOIL_MOISTURE_20 SOIL_MOISTURE_50 SOIL_MOISTURE_100 SOIL_TEMP_5 SOIL_TEMP_10 SOIL_TEMP_20 SOIL_TEMP_50 SOIL_TEMP_100 ".split()

    """
    for filepath in dataset_dir.glob("**/*.txt"):
        print(filepath)
        with open(filepath) as datafile:
            lines = [line.rstrip() for line in datafile]
            split_lines = list(map(lambda line: line.split(), lines)) 
            df=pd.DataFrame(split_lines, columns=headers)
            csv_path = filepath.with_suffix(".csv")
            df.to_csv(csv_path)
    """
            

    filepaths = dataset_dir.glob("**/*.csv")
    data = pd.concat(map(lambda filepath: pd.read_csv(filepath_or_buffer=filepath, index_col=0), filepaths))

    data.to_csv("./Dataset_2023.csv", index=False)

if __name__ == "__main__":
    main()
