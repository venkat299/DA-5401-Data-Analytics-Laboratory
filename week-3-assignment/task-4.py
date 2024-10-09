
from lazypredict.Supervised import LazyRegressor
# oh god why this library :(, it is asking me downgrade my python version 
# i had already wasted an hour trying to upgrade to 3.12 and now i have to downgrade to 3.8. why?
# i am going to colab 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

import common 

# load data
data = common.load_assignment_data()
data = data.astype('float')
x= data[['x1', 'x2', 'x3', 'x4', 'x5']]
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, data.y, test_size=0.2, random_state=42)

# Initialize LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit and predict using LazyRegressor
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Display the results
print(models)

# Results
#                                Adjusted R-Squared  R-Squared     RMSE  \
# Model                                                                   
# Lasso                                        1.00       1.00    43.02   
# LassoCV                                      1.00       1.00    43.03   
# RidgeCV                                      1.00       1.00    43.26   
# SGDRegressor                                 1.00       1.00    43.29   
# LassoLars                                    1.00       1.00    43.48   
# BayesianRidge                                1.00       1.00    43.72   
# Ridge                                        1.00       1.00    43.73   
# OrthogonalMatchingPursuitCV                  1.00       1.00    44.73   
# LassoLarsIC                                  1.00       1.00    44.73   
# LassoLarsCV                                  1.00       1.00    44.73   
# Lars                                         1.00       1.00    44.73   
# LinearRegression                             1.00       1.00    44.73   
# RANSACRegressor                              1.00       1.00    44.73   
# TransformedTargetRegressor                   1.00       1.00    44.73   
# HuberRegressor                               1.00       1.00    46.02   
# PoissonRegressor                             1.00       1.00    46.24   
# PassiveAggressiveRegressor                   1.00       1.00    46.99   
# OrthogonalMatchingPursuit                    1.00       1.00    70.15   
# LarsCV                                       0.99       0.99    94.60   
# ExtraTreesRegressor                          0.99       0.99   138.95   
# GradientBoostingRegressor                    0.98       0.99   148.53   
# BaggingRegressor                             0.98       0.99   157.87   
# XGBRegressor                                 0.98       0.99   158.89   
# RandomForestRegressor                        0.98       0.98   162.51   
# AdaBoostRegressor                            0.97       0.98   185.88   
# DecisionTreeRegressor                        0.97       0.98   194.79   
# ExtraTreeRegressor                           0.97       0.98   198.87   
# ElasticNet                                   0.95       0.96   250.34   
# ElasticNetCV                                 0.89       0.91   384.45   
# TweedieRegressor                             0.87       0.90   404.65   
# KNeighborsRegressor                          0.87       0.90   408.06   
# GammaRegressor                               0.87       0.90   411.87   
# HistGradientBoostingRegressor                0.83       0.87   468.41   
# LGBMRegressor                                0.82       0.87   474.94   
# SVR                                         -0.35      -0.01  1318.20   
# NuSVR                                       -0.35      -0.01  1319.64   
# DummyRegressor                              -0.35      -0.02  1320.43   
# GaussianProcessRegressor                    -3.61      -2.46  2436.72   
# KernelRidge                                -80.32     -59.99 10234.38   
# LinearSVR                                  -82.58     -61.69 10375.77   
# MLPRegressor                               -83.65     -62.49 10441.92   

# Ahhh! I really dont know what these models are. 

