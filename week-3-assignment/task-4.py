from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

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


from sklearn.linear_model import LinearRegression

# Train an OLS regression model
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

# Predict and calculate RMSE
y_pred_ols = ols_model.predict(X_test)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))

print(f"OLS RMSE: {rmse_ols:.2f}")