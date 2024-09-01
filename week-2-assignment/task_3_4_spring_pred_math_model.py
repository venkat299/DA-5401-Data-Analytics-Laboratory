# Task 4 [15 points] - spring[harmonic oscillator] dataset
# ------------------
# 4. Train the regression model for extrapolation and evaluate the SSE.
# ------------------
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import common as my_lib
import task_2_1_stock_pred_math_model as selection

# load the assignment data
# --------------------
data_temp  = my_lib.load_assignment_data()
data = pd.DataFrame({"x":range(1,226+1,1), "y":data_temp.SpringPos})

# get the transformed data from the previous task
# --------------------
valid_crossings = 7
cycles = valid_crossings/2
angular_freq = cycles*2*math.pi/226
data["x_as_theta"] = angular_freq*np.array(data.x) 
data["x2"] = np.sin(data.x_as_theta)
data["x3"] = np.array(data.x_as_theta)*(np.sin(data.x_as_theta))
data.insert(0, "bias", 1)

# get the training data
# ---------------------
(train, evaldt, test) = selection.split_data_for_extrapolation(data, 0.7, 0.15)


# transform the data into matrices
# --------------------
X = np.array(train[['bias', 'x_as_theta', 'x2', 'x3',]])
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
train_mse = round(my_lib.MSE(y, yhat),2)

train["yhat"] = yhat
train = train.sort_values(by="x", ascending=True)
print(train.head())
print("Intercept=", model.intercept_, "Beta = ", model.coef_)
print("train data MSE =", train_mse)



# now to evaluate model on eval data
# ----------------------------------
X_evaldt = np.array(evaldt[['bias', 'x_as_theta', 'x2', 'x3',]])
y_evaldt = np.array(evaldt.y)
yhat_evaldt = model.predict(X_evaldt)
evaldt_mse = round(my_lib.MSE(y_evaldt, yhat_evaldt),2)
evaldt["yhat"] = yhat_evaldt
evaldt = evaldt.sort_values(by="x", ascending=True)
print("evaldt data MSE =", evaldt_mse)


# now to evaluate model on test data
# ----------------------------------
X_test = np.array(test[['bias', 'x_as_theta', 'x2', 'x3',]])
y_test = np.array(test.y)
yhat_test = model.predict(X_test)
test_mse = round(my_lib.MSE(y_test, yhat_test),2)
test["yhat"] = yhat_test
test = test.sort_values(by="x", ascending=True)
print("test data MSE =", test_mse)


# Let's plot the train data and the predicted values on the same plot
# --------------------
fig, axs = plt.subplots(3,1)
fig.set_size_inches(12,15)
ax_1 = axs.flat[0]
ax_1.plot(train.x, train.y, 'r+',label='y actual')
ax_1.plot(train.x, train.yhat, 'b.', label='y predicted')
ax_1.plot(train.x, train.yhat, 'y-', label='y fit')
ax_1.set_ylabel('Spring Position')
# ax_1.set_xlabel('Time')
ax_1.text(.65, 0.9, "train MSE = "+str(train_mse), transform=ax_1.transAxes, ha="left", va="top")
ax_1.set_title(f'training results for 70% data')
ax_1.legend()

# Let's plot the test data and the predicted values on the same plot
# --------------------
ax_2 = axs.flat[1]
ax_2.plot(evaldt.x, evaldt.y, 'r+', label='y actual')
ax_2.plot(evaldt.x, evaldt.yhat, 'b.', label='y predicted') 
ax_2.set_ylabel('Spring Position')
# ax_2.set_xlabel('Time')
ax_2.text(.65, 0.9, "evaldt MSE = "+str(evaldt_mse), transform=ax_2.transAxes, ha="left", va="top")
ax_2.set_title(f'Extrapolation of eval data 15 %')
ax_2.legend()

# Let's plot the test data and the predicted values on the same plot
# --------------------
ax_3 = axs.flat[2]
ax_3.plot(test.x, test.y, 'r+', label='y actual')
ax_3.plot(test.x, test.yhat, 'b.', label='y predicted') 
ax_3.set_ylabel('Spring Position')
ax_3.set_xlabel('Time')
ax_3.text(.65, 0.9, "test MSE = "+str(test_mse), transform=ax_3.transAxes, ha="left", va="top")
ax_3.set_title(f'Extrapolation of test data 15 %')
ax_3.legend()

plt.show()


# conclusion
# train data MSE = 4.32
# test data MSE = 7.49
# test data MSE = 3.63