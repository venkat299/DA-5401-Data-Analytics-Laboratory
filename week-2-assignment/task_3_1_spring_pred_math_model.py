# Task 3 [15 points] - spring[harmonic oscillator] dataset
# ------------------
# 1. Split your data into Train, Eval & Test.
# ------------------
# Interpolation: When you randomly split the data into train, eval and test; 
# your test and evaluation data points may be inside the data range (time range). 
# When you can predict those points correctly, you are essentially recovering 
# missing data in the regression line. 
# This is also called the interpolation problem.

# Extrapolation: In this scenario, the test and eval points should be outside 
# the time range of the training data. If your model is a good fit, and when 
# you predict the data point outside the range, you are essentially 
# extrapolating the regression line. This is also called the “Forecasting” task.
import pandas as pd

def split_data_for_interpolation(dataframe:pd.DataFrame, train_ratio, test_ratio):
    
    # shuffle the data set
    data_shuffled = dataframe.sample(frac=1, random_state=1)
    
    total_rows = dataframe.shape[0]
    train_size = int(total_rows*train_ratio)
    test_size = int(total_rows*test_ratio)
    
    # Split data into train, eval, test
    train = data_shuffled[0:train_size]
    test = data_shuffled[train_size:train_size+test_size]
    eval = data_shuffled[train_size+test_size:]

    return train, test, eval

def split_data_for_extrapolation(dataframe, train_ratio, test_ratio):
    
    total_rows = dataframe.shape[0]
    train_size = int(total_rows*train_ratio)
    test_size = int(total_rows*test_ratio)
    
    # Split data into train, eval, test
    train = dataframe[0:train_size]
    test = dataframe[train_size:train_size+test_size]
    eval = dataframe[train_size+test_size:]

    return train, test, eval





