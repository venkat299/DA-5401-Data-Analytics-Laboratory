# Task 1-1
# --------
# 1. Implement the OLS closed form solution using numpy’s matrix 
# operators to find the value of ‘m’ that minimizes SSE.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

import common as lib

# load the assignment data
# --------------------
data_temp  = lib.load_assignment_data()
data = pd.DataFrame({"x":range(226), "y":data_temp.StockPrice})

# scale the input as per the slope from task 1.1 m=0.11899413
# --------------------
m = 0.09782485
data.insert(1, "x_scaled", data.x)#*m)
# print(data.head())

# transform the data into matrices
# --------------------
y = np.array(data.y)
X = np.expand_dims(data.x_scaled, 1)
# X_with_bias = np.array(data[["bias","x"]])

# model : Y = mX where m = tan(theta)
# --------------------
beta_angle = list(range(0,65,5))
beta = list([math.tan(math.radians(i)) for i in range(0,65,5)])
sse_ls = []
for tan_theta in beta:
    # predict
    y_hat = lib.predict(np.array([tan_theta]), X)
    # compute SSE
    sse = lib.SSE(y,y_hat)
    # store SSE
    sse_ls.append(sse)
    # print(tan_theta, sse)

min_sse = min(sse_ls)
min_sse_id = sse_ls.index(min(sse_ls)) 
min_tan_theta = beta[min_sse_id]
yhat = lib.predict(np.array([min_tan_theta]), X)
data["yhat"] = yhat


angle = round(math.degrees(math.atan(min_tan_theta)))


# visulaize using subplots
# --------------------
fig, axs = plt.subplots(2,1)
ax_1 = axs.flat[0]
# ax_1.scatter(beta, sse_ls)
ax_1.plot(beta_angle, sse_ls, 'r-')
ax_1.set_ylabel('SSE')
ax_1.set_xlabel('angle of slope')
ax_1.text(.05, 0.9, "angle with min sse = "+str(angle)+"degree", transform=ax_1.transAxes, ha="left", va="top")
ymin, ymax = ax_1.get_ylim()
ax_1.vlines(angle, ymin, ymax)


# Let's plot the raw data and the regression line on the same plot
# --------------------
ax_2 = axs.flat[1]
ax_2.plot(data.x, data.y, 'r+')
ax_2.plot(data.x, yhat, 'b-')
ax_2.set_ylabel('Stock Price')
ax_2.set_xlabel('Time')

plt.show()


print(data.head())
print("min_sse = ", min_sse)
print("min beta = ", min_tan_theta)
print("min angle = ", angle)

# conclusion : m value ie theta value which gives minimum sse is 5 degree and sse = 7644.25
    
