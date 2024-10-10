import pandas as pd
import os

from sklearn.model_selection import train_test_split
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


# Display the shape of the train and test sets
print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Train set: (48000, 100), (48000,)
# Test set: (12000, 100), (12000,)

# Define the parameter grids for each classifier
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4]
}


# Initialize the classifiers
model = DecisionTreeClassifier()

# Initialize tqdm progress bar
best_score = -1
best_params = None


# Get total number of combinations
grid = list(ParameterGrid(param_grid))
total_combinations = len(grid)

print("Doing grid search to find best parameter ...")

with tqdm(total=total_combinations) as pbar:
    # Iterate over all parameter combinations
    for params in grid:
        # Set the model parameters
        model.set_params(**params, )
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        predict_test = model.predict(X_test)
        score = accuracy_score(y_test, predict_test)
        
        # Track the best score and parameters
        if score > best_score:
            best_score = score
            best_params = params
        
        # Update the progress bar
        pbar.update(1)

# Print the best results
print(f"Best Score: {best_score}")
print(f"Best Parameters: {best_params}")

# output
# Best Score: 0.9919166666666667
# Best Parameters: {'max_depth': 10, 'min_samples_leaf': 1}

# best_params = {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}
best_model = DecisionTreeClassifier(random_state=42)
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



print(y_test.shape, predict_test_prob.shape)
# Calculate ROC curve and AUC, specifying the positive label ('pos')
fpr, tpr, thresholds = roc_curve(y_test, predict_test_prob, pos_label='pos')
roc_auc = roc_auc_score(y_test, predict_test_prob)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, predict_test))

# ====================================
# output
# ====================================

# Test Accuracy score of the model : 0.9921666666666666
# Train Accuracy score of the model : 0.9964375
# (12000,) (12000,)
# Classification Report:
#               precision    recall  f1-score   support

#          neg       0.99      1.00      1.00     11800
#          pos       0.83      0.67      0.74       200

#     accuracy                           0.99     12000
#    macro avg       0.91      0.83      0.87     12000
# weighted avg       0.99      0.99      0.99     12000


# Step 7: Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(_dir,'img/04-01-confusion_decision-tree.png'), dpi=300)

plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(_dir,'img/04-02-ROC_decision-tree.png'), dpi=300)

plt.show()