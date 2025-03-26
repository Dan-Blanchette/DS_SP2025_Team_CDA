import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_csv(file_path):
    """Preprocess a CSV file into a Pandas DataFrame."""
    df = pd.read_csv(file_path)

    # Fill missing values
    for column in df.columns:
        if df[column].dtype == 'object':  # Categorical columns
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:  # Numerical columns
            df[column].fillna(df[column].median(), inplace=True)

    return df


def visualize_data(df):
    """Visualize the dataset with histograms."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Plot numerical features
    df[numerical_cols].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Histograms of Numerical Features")
    plt.show()

    # Plot categorical features
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=df[col])
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()


asthma_df = preprocess_csv("asthma_hospitalizations.csv")
aqi_df = preprocess_csv("annual_aqi_by_county_2022.csv")

# Replace "Suppressed" with NaN in asthma data
asthma_df["Value"] = asthma_df["Value"].replace("Suppressed", np.nan)

# Merge data
merged_df = pd.merge(asthma_df, aqi_df, on=["County", "State", "Year"],
                     how="inner")

# Save and check results
merged_df.to_csv("merged_data.csv", index=False)
print(merged_df.head())
# visualize_data(merged_df)
