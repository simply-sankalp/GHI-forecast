import os
import pandas as pd

DATA_DIR = os.path.join("dataset", "rajasthan1")

REQUIRED_COLS = [
    "Year", "Month", "Day", "Hour", "Minute",
    "DHI", "DNI", "GHI",
    "Clearsky DHI", "Clearsky DNI", "Clearsky GHI",
    "Dew Point", "Temperature", "Pressure",
    "Relative Humidity", "Solar Zenith Angle",
    "Snow Depth", "Wind Speed"
]

def load_all_files(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith(".xlsx")]
    all_df = []

    print("Loading Excel files:")
    for file in files:
        path = os.path.join(data_dir, file)
        print("  ->", file)

        # skip first 2 metadata rows
        df = pd.read_excel(path, skiprows=2)

        # Keep only required columns
        df = df[[c for c in REQUIRED_COLS if c in df.columns]]

        # Create datetime column
        df["Datetime"] = pd.to_datetime(
            df[["Year", "Month", "Day", "Hour", "Minute"]]
        )

        all_df.append(df)

    merged_df = pd.concat(all_df, ignore_index=True)
    merged_df = merged_df.set_index("Datetime")

    return merged_df


def main():
    print("Starting Rajasthan dataset merge...\n")
    df = load_all_files(DATA_DIR)

    print("\nFinal dataframe shape:", df.shape)
    print("\nColumns included:", df.columns.tolist())

    output_path = "merged_rajasthan_filtered.xlsx"
    df.to_excel(output_path)

    print(f"\nFiltered merged dataset saved as: {output_path}")


if __name__ == "__main__":
    main()
