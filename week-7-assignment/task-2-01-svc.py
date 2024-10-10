import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, 
                             roc_curve, roc_auc_score, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight


_dir = os.path.dirname(__file__)


# Load the datasets from CSV files
# data = pd.read_csv(os.path.join(_dir, 'data_processed.csv'))
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


# Best Parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
best_params = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

# Approach a: Undersampling the majority class and/or oversampling the minority class
rus = RandomUnderSampler(random_state=42, sampling_strategy=0.09)
ros = RandomOverSampler(random_state=42, sampling_strategy=0.33)
smote = SMOTE(sampling_strategy=0.33, k_neighbors=1, random_state=43)

X_resampled_rus, y_resampled_rus = rus.fit_resample(X_train, y_train)
X_resampled_ros, y_resampled_ros = ros.fit_resample(X_train, y_train)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)

# Approach b: Using class_weight
weights = {'pos': 10, 'neg': 15}
# logreg_weighted = SVC(class_weight='balanced', **best_params)
logreg_weighted = SVC(class_weight=weights, **best_params)

# Approach c: Using sample_weights
sample_weights = compute_sample_weight(class_weight=weights, y=y_train)



logreg_rus = SVC(**best_params)
logreg_ros = SVC(**best_params)
logreg_smote = SVC(**best_params)
logreg_sample_weighted = SVC(**best_params)



logreg_rus.fit(X_resampled_rus, y_resampled_rus)
logreg_ros.fit(X_resampled_ros, y_resampled_ros)
logreg_smote.fit(X_resampled_smote, y_resampled_smote)
logreg_weighted.fit(X_train, y_train)
logreg_sample_weighted.fit(X_train, y_train, sample_weight=sample_weights)

# Predict on the test data
y_pred_logreg_rus = logreg_rus.predict(X_test)
y_pred_logreg_ros = logreg_ros.predict(X_test)
y_pred_logreg_smote = logreg_smote.predict(X_test)
y_pred_logreg_weighted = logreg_weighted.predict(X_test)
y_pred_logreg_sample_weighted = logreg_sample_weighted.predict(X_test)

# Calculate the macro average F1 scores
f1_logreg_rus = f1_score(y_test, y_pred_logreg_rus, average='macro')
f1_logreg_ros = f1_score(y_test, y_pred_logreg_ros, average='macro')
f1_logreg_smote = f1_score(y_test, y_pred_logreg_smote, average='macro')
f1_logreg_weighted = f1_score(y_test, y_pred_logreg_weighted, average='macro')
f1_logreg_sample_weighted = f1_score(y_test, y_pred_logreg_sample_weighted, average='macro')

# Print the macro average F1 scores
print("Macro average F1 scores:")
print("SVC (baseline performance): 0.85",)
print("SVC with RandomUnderSampler:", round(f1_logreg_rus,2))
print("SVC with RandomOverSampler:", round(f1_logreg_ros,2))
print("SVC with SMOTE:", round(f1_logreg_smote,2))
print("SVC with class_weight='balanced':", round(f1_logreg_weighted,2))
print("SVC with sample weights:", round(f1_logreg_sample_weighted,2))



# Results
# Macro average F1 scores:
# SVC (baseline performance): 0.85
# SVC with RandomUnderSampler: 0.84
# SVC with RandomOverSampler: 0.84
# SVC with SMOTE: 0.84
# SVC with class_weight='balanced': 0.86
# SVC with sample weights: 0.86