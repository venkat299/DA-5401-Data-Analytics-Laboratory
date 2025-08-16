# Data Analytics Laboratory Coursework

This repository contains weekly assignments completed for the DA-5401 Data Analytics Laboratory. Each folder (`week-<n>-assignment`) holds the material for that week, including code, notebooks, reports, and associated resources.

## Week 1 – Data Acquisition & Image Transformation
- Converted the **Psyduck** image into CSV format and loaded it as a DataFrame.
- Performed cleansing to offset negative coordinates and discretised the points for rasterisation.
- Generated rotated and flipped versions of the image using permutation matrices and visualised the results.

## Week 2 – Linear Regression & Harmonic Oscillator Modelling
- Built multiple linear models to predict stock prices and harmonic oscillator amplitudes.
- Engineered features such as polynomial terms and trigonometric components to reduce error (SSE/MSE).
- Compared interpolation and extrapolation performance across training, evaluation, and test splits.

## Week 3 – Exploratory Data Analysis & OLS Refinement
- Established a baseline Ordinary Least Squares model and evaluated residuals.
- Conducted EDA: distribution plots, violin plots, pair plots, and correlation heatmaps.
- Reduced multicollinearity by removing redundant features and adding interaction terms.
- Benchmarked refined models against auto-generated regressors using `lazypredict`.

## Week 4 – Binary Classification on IRIS
- Binarised the IRIS dataset (Setosa vs. Others) and trained a `DummyBinaryClassifier`.
- Evaluated precision, recall, F1, PRC, and ROC metrics across thresholds.
- Visualised decision boundaries for different probability distributions and `p` values.

## Week 5 – Nursery Dataset & Bipolar Sigmoid Analysis
- Applied Decision Tree, Logistic Regression, and k-NN classifiers with ordinal and one-hot encoding.
- Tuned hyperparameters via cross‑validation and compared accuracy/precision with UCI baselines.
- Analysed the bipolar sigmoid function against `tanh`, exploring linearity ranges for varying `a` values.

## Week 6 – Multilingual Text Classification
- Assembled a 27‑locale corpus from the MASSIVE dataset and stored locale-specific text files.
- Normalised text by removing punctuation and stop words; grouped samples by continent.
- Reduced dimensionality with PCA and trained a Regularized Discriminant Analysis classifier.

## Week 7 – Imbalanced Dataset Modelling (IDA2016)
- Preprocessed the 60k‑sample APS failure dataset: imputation, scaling, variance and correlation filtering.
- Evaluated baseline SVC, Logistic Regression, and Decision Tree models on imbalanced data.
- Improved macro F1 scores using class weights and sample weighting; compared with resampling methods.

## Week 8 – AdaBoost from Scratch
- Implemented AdaBoost using decision stumps as weak learners on a synthetic circle dataset.
- Tracked accuracy across boosting rounds and visualised per‑iteration decision boundaries.
- Produced the final classifier boundary illustrating the combined ensemble.

## Week 9 – Visual Taxonomy Exploration
- Downloaded the Kaggle Visual Taxonomy dataset and inspected category/attribute metadata.
- Generated balanced samples for selected category–attribute pairs.
- Created visualisations mirroring course examples for exploratory analysis.

---
Each assignment folder contains the source notebooks, scripts, and reports detailing the work for that week.
