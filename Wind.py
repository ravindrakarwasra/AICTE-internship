import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Advanced visualization
from sklearn.model_selection import train_test_split  # Splitting data for ML
from sklearn.preprocessing import StandardScaler  # Data scaling

# Define dataset files
file_names = ['Location1.csv', 'Location2.csv', 'Location3.csv', 'Location4.csv']

# Load and merge datasets
dataframes = []
for file in file_names:
    try:
        df = pd.read_csv(file, low_memory=False)  # Handle large datasets efficiently
        dataframes.append(df)
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")
    except pd.errors.EmptyDataError:
        print(f"Warning: {file} is empty. Skipping.")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Ensure there is data to merge
if not dataframes:
    raise ValueError("No valid datasets found. Exiting.")

# Concatenating all locations into a single DataFrame
merged_data = pd.concat(dataframes, ignore_index=True)

# Save merged data
merged_data.to_csv('merged_locations.csv', index=False)

# Display basic information
print("Dataset Info:")
merged_data.info()

# Summary Statistics
print("\nSummary Statistics:")
print(merged_data.describe(include='all').T)  # Include categorical data

# Check for missing values and duplicates
missing_values = merged_data.isnull().sum()
missing_percentage = (missing_values / len(merged_data)) * 100

print("\nMissing Values (Count & %):\n", pd.DataFrame({'Count': missing_values, 'Percentage': missing_percentage}))
print("\nNumber of Duplicates:", merged_data.duplicated().sum())

# Drop duplicates if any
if merged_data.duplicated().sum() > 0:
    merged_data.drop_duplicates(inplace=True)
    print("Duplicates removed.")

# One-hot encode the 'Location' column while avoiding multicollinearity
if 'Location' in merged_data.columns:
    merged_data = pd.get_dummies(merged_data, columns=['Location'], drop_first=True)

# Drop the 'Time' column if it exists
if 'Time' in merged_data.columns:
    merged_data.drop(columns=['Time'], inplace=True)

# Display updated DataFrame structure
print("\nUpdated DataFrame Columns:\n", merged_data.columns)

# Display first few rows of cleaned dataset
print("\nCleaned Dataset Preview:\n", merged_data.head())
