# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import gaussian_kde

# INPUT_FILE = "merged_rajasthan_filtered.xlsx"
# COLUMN = "GHI"   # change to any other column if needed

# def main():
#     # Load merged dataset
#     df = pd.read_excel(INPUT_FILE, index_col="Datetime", parse_dates=True)
#     data = df[COLUMN].dropna()

#     # ---------------------------------------------------
#     # 1. FREQUENCY HISTOGRAM
#     # ---------------------------------------------------
#     plt.figure(figsize=(12, 5))
#     plt.hist(data, bins=60, color="gray", edgecolor="black")
#     plt.title(f"Frequency Histogram of {COLUMN}")
#     plt.xlabel(COLUMN)
#     plt.ylabel("Frequency (Counts)")
#     plt.tight_layout()
#     plt.show()

#     # ---------------------------------------------------
#     # 2. PROBABILITY DISTRIBUTION (Histogram + KDE Curve)
#     # ---------------------------------------------------
#     plt.figure(figsize=(12, 5))

#     # Probability density histogram
#     plt.hist(data, bins=60, density=True, alpha=0.4, color="gray", edgecolor="black")

#     # KDE curve
#     kde = gaussian_kde(data)
#     x = np.linspace(data.min(), data.max(), 500)
#     plt.plot(x, kde(x), color="blue", linewidth=2)

#     plt.title(f"Probability Distribution of {COLUMN}")
#     plt.xlabel(COLUMN)
#     plt.ylabel("Probability Density")
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

INPUT_FILE = "merged_rajasthan_filtered.xlsx"
COLUMN = "GHI"   # change if needed

def main():
    # Load merged dataset
    df = pd.read_excel(INPUT_FILE, index_col="Datetime", parse_dates=True)

    # Remove zero (night/morning) values
    data = df[COLUMN]
    data = data[data > 0].dropna()

    print(f"Using {len(data)} non-zero values of {COLUMN}")

    # ---------------------------------------------------
    # 1. FREQUENCY HISTOGRAM (NON-ZERO VALUES)
    # ---------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.hist(data, bins=60, color="gray", edgecolor="black")
    plt.title(f"Frequency Histogram of {COLUMN} (Zero Removed)")
    plt.xlabel(COLUMN)
    plt.ylabel("Frequency (Counts)")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------
    # 2. PROBABILITY DISTRIBUTION (Histogram + KDE Curve)
    # ---------------------------------------------------
    plt.figure(figsize=(12, 5))

    # Probability density histogram
    plt.hist(data, bins=60, density=True, alpha=0.4, color="gray", edgecolor="black")

    # KDE curve
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 500)
    plt.plot(x, kde(x), color="blue", linewidth=2)

    plt.title(f"Probability Distribution of {COLUMN} (Zero Removed)")
    plt.xlabel(COLUMN)
    plt.ylabel("Probability Density")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
