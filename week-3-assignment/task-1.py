# Task 1 [5 points]
# -----------------
# Fit OLS on the data directly and evaluate the baseline SSE loss. 
# You will observe that the loss is very high, but that’s ok. 
# You will strive hard to apply creative ways to reduce the loss.

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import scale
import statsmodels
import statsmodels.formula.api as smf

import common

# load data
data = common.load_assignment_data()
data = data.astype('float')
print(data.head())

formula = "y ~ x1 + x2 + x3 + x4 + x5"
# Fitting linear model
model = smf.ols(formula= formula, data=data).fit()
summary = model.summary()

print(summary)

# Calculate RSS
print(f'Residual Sum of Squares (RSS): {model.ssr}')

# # OLS result
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.999
# Model:                            OLS   Adj. R-squared:                  0.999
# Method:                 Least Squares   F-statistic:                 2.763e+04
# Date:                Sat, 24 Aug 2024   Prob (F-statistic):          1.47e-148
# Time:                        18:49:43   Log-Likelihood:                -474.98
# No. Observations:                 101   AIC:                             962.0
# Df Residuals:                      95   BIC:                             977.6
# Df Model:                           5                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept  -9655.3103     83.303   -115.906      0.000   -9820.688   -9489.933
# x1         -1067.3690   1147.895     -0.930      0.355   -3346.229    1211.491
# x2             0.1007      0.014      7.289      0.000       0.073       0.128
# x3            -0.0572      0.053     -1.083      0.282      -0.162       0.048
# x4           284.3633     88.453      3.215      0.002     108.763     459.964
# x5             1.6285      0.090     18.017      0.000       1.449       1.808
# ==============================================================================
# Omnibus:                       24.930   Durbin-Watson:                   1.535
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               36.395
# Skew:                           1.149   Prob(JB):                     1.25e-08
# Kurtosis:                       4.836   Cond. No.                     1.23e+05
# ==============================================================================

# Inference 
# Loss for linear regression model is pretty high (71877)
# The condition number is large, 1.23e+05. This indicates
# that there are strong multicollinearity
