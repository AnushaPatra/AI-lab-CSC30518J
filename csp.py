#!/usr/bin/env python
# coding: utf-8

# In[48]:


from constraint import Problem

# Define the problem
problem = Problem()

# Add variables
problem.addVariable('A', range(1, 10))
problem.addVariable('B', range(1, 10))
problem.addVariable('C', range(1, 10))

# Add constraints
problem.addConstraint(lambda a, b, c: a + b == c, ('A', 'B', 'C'))

# Solve the problem
solutions = problem.getSolutions()

# Print solutions
print(solutions)


# In[ ]:





# In[ ]:




