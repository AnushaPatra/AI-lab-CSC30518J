#!/usr/bin/env python
# coding: utf-8

# In[60]:


# Define linguistic variables
service = {'poor': (0, 5, 10), 'average': (5, 10, 15), 'excellent': (10, 15, 20)}
food = {'poor': (0, 5, 10), 'average': (5, 10, 15), 'excellent': (10, 15, 20)}
tip = {'low': (0, 5, 10), 'medium': (10, 15, 20), 'high': (15, 20, 25)}

# Define fuzzy rules
rules = {
    ('poor', 'poor'): 'low',
    ('poor', 'average'): 'low',
    ('poor', 'excellent'): 'medium',
    ('average', 'poor'): 'low',
    ('average', 'average'): 'medium',
    ('average', 'excellent'): 'high',
    ('excellent', 'poor'): 'medium',
    ('excellent', 'average'): 'high',
    ('excellent', 'excellent'): 'high'
}

# Fuzzy inference
def fuzzy_inference(service_level, food_quality):
    return rules[(service_level, food_quality)]

# Example usage
service_level = 'excellent'
food_quality = 'average'
print("Service level:", service_level)
print("Food quality:", food_quality)
print("Recommended tip:", fuzzy_inference(service_level, food_quality), "USD")


# In[ ]:





# In[ ]:




