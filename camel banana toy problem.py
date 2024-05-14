#!/usr/bin/env python
# coding: utf-8

# In[11]:


dp = [[-1 for i in range (3002)] for j in range (1001)]
def recbananacnt(A ,B ,C):
    if (B <= A):
        return 0
    if (B <= C):
        return B-A
    if (A == 0):
        return B
    if (dp[A][B] != -1):
        return dp[A][B]
    maxCnt = -2**32
    tripCnt=((2 * B) // C) - 1 if (B % C == 0) else ((2 * B) // C) + 1
    for i in range (1, A+1):
        currCnt = recBananaCnt(A - i, B - tripCnt * i, C)
        if (currCnt > maxCnt):
            maxCnt = currCnt
            dp[A][B] = maxCnt
    return maxCnt
def maxbananacnt(A, B, C):
    return recbananacnt(A,B,C)

A=1000
B=3000
C=1000
print(maxbananacnt(A,B,C))


# In[ ]:





# In[ ]:




