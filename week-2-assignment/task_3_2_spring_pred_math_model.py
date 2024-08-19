# Task 3 [15 points] - spring[harmonic oscillator] dataset
# ------------------
# 2. Implement the regression model (OLS or LinearRegression or equivalent) 
# using appropriate feature transformation so that the SSE is lower 
# than that of Task 1.
# ------------------

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import common as my_lib

# load the assignment data
# --------------------
data_temp  = my_lib.load_assignment_data()
data = pd.DataFrame({"x":range(1,226+1,1), "y":data_temp.SpringPos})

# sub task : to find the appropriate math model for data, 
# ------------------
# since the given data is from a damped harmonic oscillator
# below equation is difficult to linearize
# y= A exp(-dx)sin(sqrt(1-d^2)wx+p)
# alternatively a close approximate equation which fits the given data
# is selected for modelling 
# y = c + m1*x1 + m2*sin(x1)+ m3*x1*sin(x1)


# sub task : feature transformation
# ------------------
# the time data (x axis) is not given and only amplitude (y axis) is available
# let us scale the input data (x axis)
# let us find the number of the zero crossings, angular frequency from the amplitude data
# -------------------
amplitudes = data.y
crossings = my_lib.find_zero_crossings(amplitudes)


# After visual inspection of the calculated zero crossing points,
# due to noise in the data there are more zero crossings than the valid ones
# there is approximately 7 valid zero crossings.

valid_crossings = 7
cycles = valid_crossings/2 # 12 cycles
print(f"Approximate number of cycles: {cycles}")
omega = 2 * np.pi * 1 

# phase = my_lib.find_phase_from_zero_crossings(amplitudes, omega, zero_crossings)
# print(f"Estimated Phase: {phase} radians")

# convert x from int to radians based on number of cycles
# --------------------
angular_freq = cycles*2*math.pi/226
print(f"x scale_factor : {angular_freq}")
data["x_as_theta"] = angular_freq*np.array(data.x) #3.56 = 470

# add other feature components of model
data["x2"] = np.sin(data.x_as_theta)
data["x3"] = np.array(data.x_as_theta)*(np.sin(data.x_as_theta))

# add bias
# --------------------
data.insert(0, "bias", 1)

# print(data.head())

# transform the data into matrices
# --------------------
X = np.array(data[['bias', 'x_as_theta', 'x2', 'x3',]])
y = np.array(data.y)

# train the model
# --------------------
model = LinearRegression().fit(X, y)

# prediction 
# --------------------
yhat = model.predict(X)

data["yhat"] = yhat 

# SSE calculation
# --------------------
sse = my_lib.SSE(data.y, data.yhat)

print(data.head())

print("Intercept=", model.intercept_, "Beta = ", model.coef_)
print("SSE =", sse)

# Let's plot the raw data and the regression line on the same plot
# --------------------
# Plot the data and the identified peaks
fig, axs = plt.subplots(2,1)
fig.set_size_inches(12,12)

ax_1 = axs.flat[0]
ax_1.plot(data.x, data.y, label='y actual(Harmonic Oscillator Data)')
ax_1.plot(data.x[crossings], data.y[crossings], "x", label='zero crossing')
ax_1.set_xlabel('Time')
ax_1.set_ylabel('Amplitude')
ax_1.set_title(f'Harmonic Oscillator with {cycles} Cycles')
ax_1.legend()

ax_2 = axs.flat[1]
ax_2.plot(data.x, data.y, 'r+', label="y actual")
ax_2.plot(data.x, data.yhat, 'b-', label="Linear regression fit; sse="+str(round(sse))) 
ax_2.set_ylabel('Spring Position')
ax_2.set_xlabel('Time')
ax_2.legend()

plt.show()

# conclusion :  
# parameters which gives minimum sse of 1038.86 for the model
# y = c + m1*x1 + m2*sin(x1)+ m3*x1*sin(x1) is
# c= 2.5291836999098107 
# m1, m2, m3 =  [ 3.98147151e-03  2.49248222e+01 -1.05378241e+00]


# def get_transformed_data():
#     df_dropped = data.drop(columns=["yhat"])
#     return df_dropped
     