# Let’s learn to deal with class-imbalance this time! We will consider the IDA2016 Challenge dataset for
# our experimentation. The dataset is a binary classification y = {‘pos’, ‘neg’} problem with 170
# features and 60,000 data points. The craziness here is that the class ratio is 1:59, that is, for every
# positive data point, there are 59 negative data points in the training data. The challenge dataset has
# a training file (aps_failure_training_set.csv) and a testing file (aps_failure_test_set.csv). We will
# consider only the training file for our experimentation.

# Task : Preprocess the dataset to make in amenable for building classifiers.

import pandas as pd
from sklearn.model_selection import train_test_split
import os

import seaborn as sns
import matplotlib.pyplot as plt

import common

# read from csv and convert to dataframe
_dir = os.path.dirname(__file__)

# Load the dataset
file_path = os.path.join(_dir, "aps_failure_training_set.csv")
data = pd.read_csv(file_path)

# print(data.head())
print(data.shape)

na_summary = common.count_value(data, 'na')
print(na_summary)



# Replace "na" with NaN to handle them uniformly (optional but recommended)
data.replace('na', pd.NA, inplace=True)

# Filling missing values with mode
for column in data.columns[1:]:  # Skip the 'class' column
    mode_value = data[column].mode()[0]  # Calculate mode
    data.fillna({column: mode_value}, inplace=True)
print("DataFrame after filling with mode:")
na_summary = common.count_value(data, 'na')
print(na_summary)

data.to_csv(os.path.join(_dir, 'data_processed.csv'), index=False)

# Import necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Separate features and target variables
# Assuming the dataset has features and a target column named 'target'
X = data.drop('class', axis=1)  # features
y = data['class']  # class variable

# Step 4: Standardize the features (PCA requires standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Step 5: Apply PCA
# # Specify the number of principal components to keep
# n_components = 80  # You can adjust this
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X_scaled)
# print(X_pca.shape, X_scaled.shape, X.shape)
# # Step 6: Check explained variance ratio to understand how much variance is captured
# explained_variance = pca.explained_variance_ratio_
# cumulative_explained_variance = explained_variance.cumsum()

# print(f'cumulative variance : {cumulative_explained_variance[-1]}')

# # Step 1: Convert X_pca to a DataFrame
# # Add column names for the PCA components (if desired)
# X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# # Step 2: Reset index of y to match X_pca if needed (in case indices don't align)
# y_reset = y.reset_index(drop=True)

# # Step 3: Concatenate X_pca and y into a single DataFrame
# df_pca = pd.concat([X_pca_df, y_reset], axis=1)

# # Step 4: Rename the target column (if needed)
# df_pca.rename(columns={'target': 'Target'}, inplace=True)

# # Concatenate X_pca and y
# df_pca = pd.concat([X_pca_df, y], axis=1)
# df_pca.to_csv(os.path.join(_dir, 'data_processed_pca.csv'), index=False)



# # Plot the explained variance ratio
# plt.figure(figsize=(8,6))
# plt.plot(range(1, n_components+1), cumulative_explained_variance, marker='o', linestyle='--')
# plt.title('Explained Variance by Components')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.grid()
# plt.show()

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


# Step 3: Apply Variance Threshold
# VarianceThreshold removes all features whose variance doesn't meet the threshold
# Set threshold as per your needs (e.g., 0.01 or higher)
var_thresh = VarianceThreshold(threshold=0.01)
X_var_thresh = var_thresh.fit_transform(X)

# Step 4: Convert to DataFrame for easier handling
X_var_df = pd.DataFrame(X_var_thresh, columns=X.columns[var_thresh.get_support()])

print(f"Features remaining after Variance Threshold: {X_var_df.shape[1]}")

# Step 5: Correlation Matrix
# Compute the correlation matrix for the filtered features
corr_matrix = X_var_df.corr().abs()

# Step 6: Identify and remove highly correlated features
# Create an upper triangle matrix and filter for correlation greater than 0.8
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.8
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]

print(f"Highly correlated features to be dropped: {to_drop}")

# Drop these features from the DataFrame
X_filtered = X_var_df.drop(columns=to_drop)

# Step 7: Print final feature set and shape
print(f"Final shape after Variance and Correlation filtering: {X_filtered.shape}")

# Optional: Save the reduced feature set to a new file
X_filtered['class'] = y  # Add the target column back for analysis
X_filtered.to_csv(os.path.join(_dir,'filtered_features.csv'), index=False)


# # Step 4: Standardize the features (PCA requires standardization)
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X_filtered)
# X_scaled = X_filtered.drop('class', axis=1)  # features
# y = X_filtered['class']  # class variable

# # Step 5: Apply PCA
# # Specify the number of principal components to keep
# n_components = 10  # You can adjust this
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X_scaled)
# print(X_pca.shape, X_scaled.shape, X.shape)
# # Step 6: Check explained variance ratio to understand how much variance is captured
# explained_variance = pca.explained_variance_ratio_
# cumulative_explained_variance = explained_variance.cumsum()

# print(f'cumulative variance : {cumulative_explained_variance[-1]}')

# # Step 1: Convert X_pca to a DataFrame
# # Add column names for the PCA components (if desired)
# X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# # Step 2: Reset index of y to match X_pca if needed (in case indices don't align)
# y_reset = y.reset_index(drop=True)

# # Step 3: Concatenate X_pca and y into a single DataFrame
# df_pca = pd.concat([X_pca_df, y_reset], axis=1)

# # Step 4: Rename the target column (if needed)
# df_pca.rename(columns={'class': 'class'}, inplace=True)

# # Concatenate X_pca and y
# df_pca = pd.concat([X_pca_df, y], axis=1)
# df_pca.to_csv(os.path.join(_dir, 'data_processed_pca2.csv'), index=False)
