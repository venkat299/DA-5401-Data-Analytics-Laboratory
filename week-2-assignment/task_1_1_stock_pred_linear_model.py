# Task 1-1
# --------
# 1. Implement the OLS closed form solution using numpy’s matrix 
# operators to find the value of ‘m’ that minimizes SSE.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as lib

# load the assignment data
# --------------------
data_temp  = lib.load_assignment_data()
data = pd.DataFrame({"x":range(226), "y":data_temp.StockPrice})

# let's add an intercept (bias) to the data
# --------------------
data.insert(0, "bias", 1)
print(data.head())

# transform the data into matrices
# --------------------
y = np.array(data.y)
X = np.expand_dims(data.x, 1)
X_with_bias = np.array(data[["bias","x"]])

# run the closed form solution to estimate the beta parameter.
# --------------------
beta = lib.estimateBeta(X,y)
beta_with_bias = lib.estimateBeta(X_with_bias,y)


# prediction 
# --------------------
yhat_no_bias = lib.predict(beta, X) # model: Y = mX
yhat_with_bias = lib.predict(beta_with_bias, X_with_bias) # model: Y = mX+c

# SSE calculation
# --------------------
loss_no_bias = lib.SSE(y, yhat_no_bias)
loss_with_bias = lib.SSE(y, yhat_with_bias)


data["yhat_no_bias"] = yhat_no_bias
data["yhat_with_bias"] = yhat_with_bias
print(data.head())
print("beta (no intercept/bias) :", beta)
print("SSE(no bias) =", loss_no_bias)
print("beta (with intercept/bias) :", beta_with_bias)
print("SSE(bias) =", loss_with_bias)

# Let's plot the raw data and the regression line on the same plot
# --------------------
plt.plot(data.x, data.y, 'r+')
plt.plot(data.x, yhat_no_bias, 'b-')  # yhat = y2.x*beta[0]
plt.plot(data.x, yhat_with_bias, 'g-')  # yhat = y2.x*beta[0]
plt.ylabel('Stock Price')
plt.xlabel('Time')
# plt.show()

# conclusion :  
# m value without intercept which gives minimum sse is  [0.11899413]  and sse = 3850.33
# m value with intercept ie(c,m) which gives minimum sse is [3.18244786 0.09782485]  and sse = 3274.29


