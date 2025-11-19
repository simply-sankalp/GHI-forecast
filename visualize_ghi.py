import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "merged_rajasthan_filtered.xlsx"

def main():
    print("Loading merged dataset…")
    df = pd.read_excel(INPUT_FILE, index_col="Datetime", parse_dates=True)

    ghi = df["GHI"]

    print("Displaying GHI vs Time")

    plt.figure(figsize=(18, 6))
    plt.plot(ghi.index, ghi.values, linewidth=0.4)

    plt.title("GHI vs Time (2001–2014)")
    plt.xlabel("Time")
    plt.ylabel("GHI (W/m²)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
