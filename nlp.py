#!/usr/bin/env python
# coding: utf-8

# In[31]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score
text = 'I love this book'
sentiment_score = analyze_sentiment(text)
print("sentiment score:", sentiment_score)


# In[ ]:





# In[ ]:




