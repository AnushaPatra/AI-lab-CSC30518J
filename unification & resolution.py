#!/usr/bin/env python
# coding: utf-8

# In[42]:


def unify(var, x, theta):
    if theta is None:
        return None
    elif var == x:
        return theta
    elif isinstance(var, str) and var[0].islower():
        return unify_var(var, x, theta)
    elif isinstance(x, str) and x[0].islower():
        return unify_var(x, var, theta)
    elif isinstance(var, list) and isinstance(x, list) and len(var) == len(x):
        return unify(var[1:], x[1:], unify(var[0], x[0], theta))
    else:
        return None

def unify_var(var, x, theta):
    if var in theta:
        return unify(theta[var], x, theta)
    elif x in theta:
        return unify(var, theta[x], theta)
    else:
        theta[var] = x
        return theta


theta = unify('?x', 'apple', {})
print("Unification result:", theta)


# In[41]:


def resolve(clauses):
    resolvents = []
    for i, clause1 in enumerate(clauses):
        for clause2 in clauses[i+1:]:
            for literal1 in clause1:
                for literal2 in clause2:
                    if literal1 == -literal2:
                        resolvent = set(clause1).union(set(clause2))
                        resolvent.remove(literal1)
                        resolvent.remove(literal2)
                        resolvents.append(list(resolvent))
    return resolvents

# Example usage
clauses = [[1, 2], [-2, 3], [-1, -3]]
result = resolve(clauses)
print("Resolvents:", result)


# In[ ]:




