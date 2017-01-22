# This application is designed to predict an animal's body weight by its brain weight.

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read Data
dataFrame = pd.read_fwf('brain_body.txt')
x_values = dataFrame[['Brain']]
y_values = dataFrame[['Body']]

# train model on data
# You use linear regression to find if their is a relation between two data sets
# in this case: the body and the brain weight

# So basically what linear regression does is finding a linear-function (f(x) = mx + b)
# which fits best to the data set, which the lowest possible error rate.
bodyReg = linear_model.LinearRegression()
bodyReg.fit(x_values, y_values)

# visualize results
# We scatter out the real values that we know and we plot what the linear-regression
# modul would have predicted for the given x_values aka brain weights.
plt.scatter(x_values, y_values)
plt.plot(x_values, bodyReg.predict(x_values))

plt.show()
