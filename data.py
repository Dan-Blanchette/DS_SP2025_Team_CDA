import pandas as pd
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


def visualize_correlation(df, target_column="Hospitalizations"):
    """Visualize correlation.

    Parameters:
        df (DataFrame): The merged dataset containing AQI and asthma
            hospitalization data.
        target_column (str): The column representing asthma hospitalizations.

    Returns:
        None (Displays plots)
    """
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    df = df.dropna(subset=[target_column])

    # List of AQI-related columns
    aqi_columns = [
        "Days with AQI", "Good Days", "Moderate Days", "Unhealthy for Sensitive Groups Days",
        "Unhealthy Days", "Very Unhealthy Days", "Hazardous Days", "Max AQI",
        "90th Percentile AQI", "Median AQI", "Days CO", "Days NO2", "Days Ozone",
        "Days PM2.5", "Days PM10"
    ]

    # Ensure AQI columns exist in the dataset
    relevant_columns = [col for col in aqi_columns if col in df.columns]

    # Correlation Heatmap
    plt.figure(figsize=(12, 6))
    correlation_matrix = df[[target_column] + relevant_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5)
    plt.title("Correlation Heatmap: AQI Factors vs Asthma Hospitalizations")
    plt.show()

    # Scatter Plot for Key AQI Indicators
    key_aqi_columns = ["Max AQI", "Median AQI", "Days PM2.5", "Days Ozone"]

    plt.figure(figsize=(12, 6))
    for i, col in enumerate(key_aqi_columns, 1):
        if col in df.columns:
            plt.subplot(2, 2, i)
            sns.scatterplot(x=df[col], y=df[target_column], alpha=0.6)
            plt.xlabel(col)
            plt.ylabel("Asthma Hospitalizations")
            plt.title(f"{col} vs. Asthma Hospitalizations")

    plt.tight_layout()
    plt.show()


merged_df = pd.read_csv("merged_data.csv")
visualize_correlation(merged_df)
