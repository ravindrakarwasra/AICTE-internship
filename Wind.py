# Import necessary libraries
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Advanced visualization
from sklearn.model_selection import train_test_split  # Splitting data for ML
from sklearn.preprocessing import StandardScaler  # Data scaling

# Load and merge datasets
file_names = ['Location1.csv', 'Location2.csv', 'Location3.csv', 'Location4.csv']
dataframes = [pd.read_csv(file) for file in file_names]

# Concatenating all locations into a single DataFrame
merged_data = pd.concat(dataframes, ignore_index=True)

# Save merged data
merged_data.to_csv('merged_locations.csv', index=False)

# Display basic information
print("Dataset Info:")
merged_data.info()

print("\nSummary Statistics:")
print(merged_data.describe().T)

# Check for missing values and duplicates
print("\nMissing Values:\n", merged_data.isnull().sum())
print("\nNumber of Duplicates:", merged_data.duplicated().sum())

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
