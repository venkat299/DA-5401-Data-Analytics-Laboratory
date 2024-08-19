# Task 2 [15 points]
# ------------------
# 2. Implement the regression model (OLS or LinearRegression or equivalent) 
# using appropriate feature transformation so that the SSE is lower 
# than that of Task 1.
# ------------------

# You will notice that the linear model is an ok fit for the y2. 
# What should be the mathematical model of stock price dataset? 
# If you notice the periodicity in the data, you should factor that in your
# mathematical model using an appropriate function that’s periodic. 
# The challenge here is; the trend of the magnitude is also increasing, 
# which you confirmed in your previous task. 
# So, the math model should consider both properties.

# Let's model the periodicity

# Let's borrow the slope information from the fit to set up the scale of the time axis.
# Instead of using $x$ in integer scale, we shall use the floating point scale as $x_1 \leftarrow \beta_0 * x$
# Likewise, let's create a new data dimension to capture the periodicity as $x_2 \leftarrow sin(x_1)$
# Based on the expanded feature space, now let's try to model $\hat{y} = m_1 x_1 + m_2 x_2$, note that we don't have to use the intercept $c$ as our previous linear model passed through the origin.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import common as lib

# load the assignment data
# --------------------
data_temp  = lib.load_assignment_data()
data = pd.DataFrame({"x":range(226), "y":data_temp.StockPrice})


# scale the input as per the slope from task 1.1,  m=0.11899413
# --------------------
m = 0.09782485
data["x1_scaled"] = data.x*m

# add bias
# --------------------
data.insert(0, "bias", 1)

# transform x to sin(x)
# ---------------------
data["x2_sin_x"] = np.sin(data.x1_scaled)

# transform the data into matrices
# --------------------
X = np.array(data[['bias', 'x1_scaled', 'x2_sin_x']])
y = np.array(data.y)
# print(data.head())
# print(X)
# print(y)

# train the model
# --------------------
model = LinearRegression().fit(X, y)

# prediction 
# --------------------
yhat = model.predict(X)

# SSE calculation
# --------------------
loss = lib.SSE(y, yhat)

data["yhat"] = yhat
print(data.head())
print("Intercept=", model.intercept_, "Beta = ", model.coef_)
print("SSE =", loss)

# Let's plot the raw data and the regression line on the same plot
# --------------------
plt.plot(data.x, data.y, 'r+')
plt.plot(data.x, yhat, 'b-')  # yhat = y2.x*beta[0]
plt.ylabel('Stock Price')
plt.xlabel('Time')
# plt.show()

# conclusion :  
# m value with intercept ie(c,m) which gives minimum sse is [2.74, 1.00 4.82]  and sse = 701.20

# 1. Split your data into Train, Eval & Test.
# ------------------
# ◦ Interpolation: When you randomly split the data into train, eval and test; 
# your test and evaluation data points may be inside the data range (time range). 
# When you can predict those points correctly, you are essentially recovering 
# missing data in the regression line. 
# This is also called the interpolation problem.