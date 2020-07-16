#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np 


def holdout(X_train, y_train, rate = 0.7, random_state=777):
    np.random.seed(random_state)
    np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    _X_train = X_train[:(int(X_train.shape[0] * rate)), :]
    _X_test = X_train[(int(X_train.shape[0] * rate)):, :]
    _y_train = y_train[:(int(y_train.shape[0] * rate))]
    _y_test = y_train[(int(y_train.shape[0] * rate)):]
    return _X_train, _y_train, _X_test,  _y_test


# In[ ]:


def bootstrapping(X_train, y_train):
    Datas = np.concatenate([X_train,y_train.reshape(-1,1)],axis=1)
    train_Datas = np.ones((1,Datas.shape[1]))
    for i in range(len(Datas)):
        train_Datas = np.vstack([train_Datas,Datas[np.random.randint(0,len(X_train))]])
    train_Datas = train_Datas[1:,:]
    test_Datas = Datas.copy()
    ind = []
    for i in range(len(train_Datas)):
        for j in range(len(Datas)):
            if (train_Datas[i]==Datas[j]).all():
                ind.append(j)
    test_Datas = np.delete(test_Datas,ind,0)
    _X_train, _y_train = train_Datas[:,:(Datas.shape[1]-1)], train_Datas[:,-1].reshape(train_Datas.shape[0],)
    _X_test, _y_test = test_Datas[:,:(Datas.shape[1]-1)], test_Datas[:,-1].reshape(test_Datas.shape[0],)
    return _X_train, _y_train, _X_test, _y_test


