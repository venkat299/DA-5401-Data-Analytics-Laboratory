# Task 2 [15 points]
# ------------------
# 4. Train the regression model for extrapolation and evaluate the SSE.
# ------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import common as lib
import task_2_1_stock_pred_math_model as selection

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

# get the training data
# ---------------------
(train, test, eval) = selection.split_data_for_extrapolation(data, 0.8, 0.2)

# transform the data into matrices
# --------------------
X = np.array(train[['bias', 'x1_scaled', 'x2_sin_x']])
y = np.array(train.y)
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
train_sse = round(lib.SSE(y, yhat),2)

train["yhat"] = yhat
print(train.head())
print("Intercept=", model.intercept_, "Beta = ", model.coef_)
print("train data SSE =", train_sse)


# now to evaluate model on test data
# ----------------------------------
X_test = np.array(test[['bias', 'x1_scaled', 'x2_sin_x']])
y_test = np.array(test.y)
yhat_test = model.predict(X_test)
test_sse = round(lib.SSE(y_test, yhat_test),2)
test["yhat"] = yhat_test
print("test data SSE =", test_sse)


# Let's plot the train data and the predicted values on the same plot
# --------------------
fig, axs = plt.subplots(2,1)
ax_1 = axs.flat[0]
ax_1.plot(train.x, train.y, 'r+')
ax_1.plot(train.x, yhat, 'b.') 
ax_1.set_ylabel('Stock Price')
ax_1.set_xlabel('Time')
ax_1.text(.05, 0.9, "train sse = "+str(train_sse), transform=ax_1.transAxes, ha="left", va="top")

# Let's plot the train data and the regression line on the same plot
# --------------------
ax_2 = axs.flat[1]
ax_2.plot(test.x, test.y, 'r+')
ax_2.plot(test.x, yhat_test, 'b.') 
ax_2.set_ylabel('Stock Price')
ax_2.set_xlabel('Time')
ax_2.text(.05, 0.9, "test sse = "+str(test_sse), transform=ax_2.transAxes, ha="left", va="top")

plt.show()




# conclusion :  
# m value with intercept ie(c,m) which gives minimum sse is [2.74, 1.00 4.82]  and sse = 701.20

# 1. Split your data into Train, Eval & Test.
# ------------------
# ◦ Interpolation: When you randomly split the data into train, eval and test; 
# your test and evaluation data points may be inside the data range (time range). 
# When you can predict those points correctly, you are essentially recovering 
# missing data in the regression line. 
# This is also called the interpolation problem.