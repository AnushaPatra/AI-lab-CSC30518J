#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 5, 4])

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Print the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict new values
new_X = np.array([[5], [6]])
predictions = model.predict(new_X)
print("Predictions:", predictions)


# In[ ]:





# In[ ]:




