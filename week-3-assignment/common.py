import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def load_assignment_data()-> pd.DataFrame:
    _dir = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(_dir, './Assignment3.csv'), sep=',')   

def path_for_data():
    _dir = os.path.dirname(__file__)
    return os.path.join(_dir, './Assignment3.csv')        

# OLS : estimate the value of the beta vector assuming that X is made of independent features.
def estimateBeta(X, y):
    numerator = np.matmul(np.transpose(X), y)
    denom = np.matmul(np.transpose(X), X)
    denom_inv = np.linalg.inv(denom)
    beta = np.matmul(denom_inv, numerator)
    return beta

# create a helper that would estimate yhat from X and beta.
def predict(beta, X):
    # reshape the input to a matrix, if it is appearing like an 1d array.
    if len(X.shape) != 2:
        X = np.expand_dims(X,1)
    # convert the beta list in to an array.
    beta = np.array(beta)
    # perform estimation of yhat.
    return np.matmul(X, beta)

# compute the sum of squared error between y and yhat.
def SSE(y, yhat):
    return np.sum((y-yhat)**2)

def MSE(y, yhat):
    return np.sum((y-yhat)**2)/y.size

def split_data_for_interpolation(dataframe:pd.DataFrame, train_ratio, test_ratio):
    
    # shuffle the data set
    data_shuffled = dataframe.sample(frac=1, random_state=1)
    
    total_rows = dataframe.shape[0]
    train_size = int(total_rows*train_ratio)
    test_size = int(total_rows*test_ratio)
    
    # Split data into train, eval, test
    train = data_shuffled[0:train_size]
    test = data_shuffled[train_size:train_size+test_size]
    eval_dt = data_shuffled[train_size+test_size:]

    return train, test, eval_dt

def split_data_for_extrapolation(dataframe, train_ratio, test_ratio):
    
    total_rows = dataframe.shape[0]
    train_size = int(total_rows*train_ratio)
    test_size = int(total_rows*test_ratio)
    
    # Split data into train, eval, test
    train = dataframe[0:train_size]
    test = dataframe[train_size:train_size+test_size]
    eval_dt = dataframe[train_size+test_size:]

    return train, test, eval_dt





