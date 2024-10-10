import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, 
                             roc_curve, roc_auc_score, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

_dir = os.path.dirname(__file__)


# Load the datasets from CSV files
# data = pd.read_csv(os.path.join(_dir, 'data_processed_pca2.csv'))
data = pd.read_csv(os.path.join(_dir, 'filtered_features.csv'))



# Split the dataset into features and target variable
X = data.drop('class', axis=1)
y = data['class']

# print(y)

# Split the data into train and test partitions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = MinMaxScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# # Label encode the target variable if it's categorical
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)  # Convert labels to 0 and 1
# y_test = le.transform(y_test)


# Display the shape of the train and test sets
print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")


# # Define the parameter grids for each classifier
# param_grid_svc = {
#     'kernel': ['linear', 'rbf'],
#     'C': [0.1, 1, 10],
#     'gamma': ['scale', 'auto']
# }

# # Initialize the classifiers
# model = SVC()

# # Initialize tqdm progress bar
# best_score = -1
# best_params = None


# # Get total number of combinations
# grid = list(ParameterGrid(param_grid_svc))
# total_combinations = len(grid)

# with tqdm(total=total_combinations) as pbar:
#     # Iterate over all parameter combinations
#     for params in grid:
#         # Set the model parameters
#         model.set_params(**params)
        
#         # Fit the model
#         model.fit(X_train, y_train)
        
#         # Predict and evaluate
#         predict_test = model.predict(X_test)
#         score = accuracy_score(y_test, predict_test)
        
#         # Track the best score and parameters
#         if score > best_score:
#             best_score = score
#             best_params = params
        
#         # Update the progress bar
#         pbar.update(1)

# # Print the best results
# print(f"Best Score: {best_score}")
# print(f"Best Parameters: {best_params}")

# output
# Best Score: 0.9914166666666666
# Best Parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

best_params = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
best_model = SVC(probability=True, random_state=42)
best_model.set_params(**best_params)
# Fit the model
best_model.fit(X_train, y_train)

predict_train = best_model.predict(X_train)
predict_test = best_model.predict(X_test)
predict_test_prob = best_model.predict_proba(X_test)[:, 1]
print("Test Accuracy score of the model :",accuracy_score(y_test, predict_test))
print("Train Accuracy score of the model :",accuracy_score(y_train, predict_train))

#confusion matrix for test data
conf_matrix = confusion_matrix(y_test, predict_test)

# Step 7: Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
# plt.show()

print(y_test.shape, predict_test_prob.shape)
# Calculate ROC curve and AUC, specifying the positive label ('pos')
fpr, tpr, thresholds = roc_curve(y_test, predict_test_prob, pos_label='pos')
roc_auc = roc_auc_score(y_test, predict_test_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
# plt.show()

# print(y_test, predict_test)
# accuracy = accuracy_score(y_test, predict_test)
# precision = precision_score(y_test, predict_test, pos_label='pos')
# recall = recall_score(y_test, predict_test, pos_label='pos')
# f1 = f1_score(y_test, predict_test, pos_label='pos')
# roc_auc = roc_auc_score(y_test, predict_test_prob)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, predict_test))

# # Print individual metric scores
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print(f"AUC: {roc_auc}")
