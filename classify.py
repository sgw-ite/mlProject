
# coding: utf-8

# In[1]:


import numpy as np
from math import sqrt
from collections import Counter as counter
def kNN_classify(k,X_train,Y_traiin,x):
    assert X_train.shape[0] >= k >= 1 , "k must be valid"
    assert Y_train.shape[0] == X_train.shape[0] , "the size of Y_train must be equal to X_train"
    assert x.shape[0] == X_train.shape[1] , "the feature number of x must be equal to X_train"
    distances = [sqrt(np.sum((x - x_train)**2)) for x_train in X_train]
    topk_y = [Y_train[i] for i in np.argsort(distances)[:k]]
    return counter(topk_y).most_common[1][0][0]

