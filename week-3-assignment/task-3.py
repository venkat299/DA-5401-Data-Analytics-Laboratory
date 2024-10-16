# Task 3 [5 points]
# -----------------
# Fit OLS on the selected and transformed features and check if the loss has reduced from the baseline
# estimation.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error

import common

# load data
data = common.load_assignment_data()
data = data.astype('float')

# standardize the data
scaler = StandardScaler()
X = data[['x1', 'x2', 'x3', 'x4', 'x5']]
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
X_scaled["y"]= data["y"]
data = X_scaled
print(data.head())

# create numpy objects for X,y
# x = np.array(data[['x1','x2','x3','x4','x5']])
# y = np.expand_dims(data['y'], 1)
formula = "y ~ x1 + x2 + x3 + x4 + x5"

# Fitting linear model
base_model = smf.ols(formula= formula, data=data, ).fit()

# Calculate RSS
print(f'RSS({formula}): {base_model.ssr}')
print(f'{base_model.params}')
# results
# RSS(y ~ x1 + x2 + x3 + x4 + x5): 71877.84134016877
# this is our baseline estimation

# attempt 1:
# since x2 and x5 have quadratic relation so let us remove 
# one of the variable and check the model performance

# keep x5  and remove x2
formula = "y ~  x1+ x3 + x4 + x5 "
model1 = smf.ols(formula= formula, data=data).fit()
print(f'RSS({formula}): {model1.ssr}')
print(f'{model1.params}')

# results
# RSS(y ~  x1+ x3 + x4 + x5 ): 112078.8259338177

# attempt 2:
# keep x2 and remove x5
formula = "y ~  x1+ x3 + x4 + x5 "
model1 = smf.ols(formula= formula, data=data).fit()
print(f'RSS({formula}): {model1.ssr}')
print(f'{model1.params}')
# results
# RSS(y ~  x1+ x3 + x4 + x5 ): 112078.8259338177

# as seen above in both cases model performed poorly when either of the variable x5 and x2 removed
# so although correlated they have interaction effect on y and cant be dropped

# attempt 3:
# since x1 and x4 have linear relation so let us remove 
# one of the variable and check the model performance
# keep x1  and remove x4
formula = "y ~  x1+ x3 + x2 + x5 "
model1 = smf.ols(formula= formula, data=data).fit()
print(f'RSS({formula}): {model1.ssr}')
print(f'{model1.params}')
# RSS(y ~  x1+ x3 + x2 + x5 ): 79697.67292631933

# attempt 4:
# keep x4  and remove x1
formula = "y ~  x4+ x3 + x2 + x5 "
model1 = smf.ols(formula= formula, data=data).fit()
print(f'RSS({formula}): {model1.ssr}')
print(f'{model1.params}')
# results
# RSS(y ~  x4+ x3 + x2 + x5 ): 72532.01982531647

# attempt 5:
# since both x2,x5 are jointly affecting y let us
# add interaction variable for x2,x5
formula = "y ~  x1+ x2 +x3 + x4 + x5 +I(x2*x5)"
model1 = smf.ols(formula= formula, data=data).fit()
print(f'RSS({formula}): {model1.ssr}')
print(f'{model1.params}')
# RSS(y ~  x1+ x2 +x3 + x4 + x5 +I(x2*x5)): 71875.45909455666

# attempt 6:
# similarly
# add interaction variable for x1,x4
formula = "y ~  x1+ x2 +x3 + x4 + x5 +I(x1*x4)"
model1 = smf.ols(formula= formula, data=data).fit()
print(f'RSS({formula}): {model1.ssr}')
print(f'{model1.params}')
# RSS(y ~  x1+ x2 +x3 + x4 + x5 +I(x1*x4)): 43.30481729708009
# a huge improvement in performance let us investigate this further


# attempt 7:
# now add both the interaction variables
formula = "y ~  x1+ x2 +x3 + x4 + x5 +I(x1*x4)+I(x2*x5)"
model1 = smf.ols(formula= formula, data=data).fit()
print(f'RSS({formula}): {model1.ssr}')
print(f'{model1.params}')
# RSS(y ~  x1+ x2 +x3 + x4 + x5 +I(x1*x4)+I(x2*x5)): 42.36645052820778
# Intercept     10220.280003
# x1              -19.103697
# x2               29.120515
# x3                0.057477
# x4             1011.558463
# x5               39.503012
# I(x1 * x4)       24.181106
# I(x2 * x5)       -0.110380

# attempt 8:
# now  we will try to remove features with less weight 
# ie x1, x3 , x2x5 has relatively less weight than other variables let us remove it

formula = "y ~  x2 + x4 + x5 + I(x1*x4)"
model1 = smf.ols(formula= formula, data=data).fit()
print(f'RSS({formula}): {model1.ssr}')
print(f'{model1.params}')

# RSS(y ~  x2 + x4 + x5 + I(x1*x4)): 44.630075733517955
# MSE(y ~  x2 + x4 + x5 + I(x1*x4)): 0.06614430204204995
# Intercept     10220.275392
# x2               29.063253
# x4              992.418029
# x5               39.499114
# I(x1 * x4)       24.185429
print(model1.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       1.000
# Model:                            OLS   Adj. R-squared:                  1.000
# Method:                 Least Squares   F-statistic:                 5.625e+07
# Date:                Sun, 25 Aug 2024   Prob (F-statistic):          8.59e-305
# Time:                        10:34:21   Log-Likelihood:                -102.07
# No. Observations:                 101   AIC:                             214.1
# Df Residuals:                      96   BIC:                             227.2
# Df Model:                           4                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept   1.022e+04      0.091   1.12e+05      0.000    1.02e+04    1.02e+04
# x2            29.0633      0.095    304.566      0.000      28.874      29.253
# x4           992.4180      0.096   1.04e+04      0.000     992.228     992.608
# x5            39.4991      0.072    548.433      0.000      39.356      39.642
# I(x1 * x4)    24.1854      0.061    398.386      0.000      24.065      24.306
# ==============================================================================
# Omnibus:                       37.178   Durbin-Watson:                   1.638
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):                6.248
# Skew:                          -0.005   Prob(JB):                       0.0440
# Kurtosis:                       1.782   Cond. No.                         3.20
# ==============================================================================

# so there is no significant performance increase let us fix this model and evaluate on test data

(train, test, e)=  common.split_data_for_interpolation(data, 0.75, 0.25)



formula = "y ~  x2 + x4 + x5 + I(x1*x4)"
# train
train_model = smf.ols(formula= formula, data=train).fit()
mse = mean_squared_error(train.y, train_model.predict())
print(f'Mean Squared Error for train (MSE): {mse:.4f}')

# test
test_x = pd.DataFrame(test, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
y_pred = train_model.predict(test_x)
mse = mean_squared_error(test.y, y_pred)
print(f'Mean Squared Error for test (MSE): {mse:.4f}')

# Mean Squared Error for train (MSE): 0.4774
# Mean Squared Error for test (MSE): 0.3573