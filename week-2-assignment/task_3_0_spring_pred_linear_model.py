# Task 3
# --------
# 0. Model the data using simple linear regression.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import common as lib

# load the assignment data
# --------------------
data_temp  = lib.load_assignment_data()
data = pd.DataFrame({"x":range(226), "y":data_temp.SpringPos})

# scale the input as per the slope from task 1.1, m=0.11899413
# --------------------

# transform the data into matrices
# --------------------
y = np.array(data.y)
X = np.expand_dims(data.x, 1)
# X = np.array(data[['x_scaled']])

poly_transformer = PolynomialFeatures(degree = 1) 
X_poly = poly_transformer.fit_transform(X)

# train the model
# --------------------
model = LinearRegression().fit(X_poly, y)

# prediction 
# --------------------
yhat = model.predict(X_poly)


# SSE calculation
# --------------------
loss = lib.SSE(y, yhat)


data["yhat"] = yhat


# Let's plot the raw data and the regression line on the same plot
# --------------------
plt.plot(data.x, data.y, 'r+')
plt.plot(data.x, yhat, 'b-', label='y fit; sse='+str(int(loss)))  # yhat = y2.x*beta[0]
plt.ylabel('Spring Position')
plt.xlabel('Time')
plt.title("simple linear regression model")
plt.legend()
plt.show()


print(data.head())

print("Intercept=", model.intercept_, "Beta = ", model.coef_)

print("SSE =", loss)

# conclusion :  
# m value with intercept ie(c,m) which gives minimum sse is [3.18244786 0.09782485]  and sse = 3274.29


