
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('iris.data.txt', header=None)


# In[3]:


data.head()


# In[11]:


plt.figure(figsize=(14,6))
sns.countplot(data=data,x=0,hue=4)


# In[13]:


plt.figure(figsize=(14,6))
sns.countplot(data=data,x=1,hue=4)


# In[14]:


plt.figure(figsize=(14,6))
sns.countplot(data=data,x=2,hue=4)


# In[15]:


plt.figure(figsize=(14,6))
sns.countplot(data=data,x=3,hue=4)


# In[143]:


sns.pairplot(data, hue=4)


# In[145]:


sns.heatmap(data.corr())


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X = data.drop(4,axis=1)


# In[28]:


y = data[4]


# In[33]:


y = y.map({'Iris-setosa':1,'Iris-versicolor':0,'Iris-virginica':2})


# In[38]:


y = y.astype(int)


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[49]:


from sklearn.linear_model import LogisticRegression


# In[50]:


lr_model = LogisticRegression()


# In[51]:


lr_model.fit(X_train, y_train)


# In[52]:


pred = lr_model.predict(X_test)


# In[53]:


from sklearn.metrics import confusion_matrix, classification_report


# In[54]:


pred


# In[55]:


print(classification_report(y_test, pred))


# In[56]:


print(confusion_matrix(y_test, pred))


# In[57]:


from sklearn.ensemble import RandomForestClassifier


# In[125]:


rfc = RandomForestClassifier(n_estimators=30, max_features=3)


# In[126]:


rfc.fit(X_train, y_train)


# In[127]:


pred1 = rfc.predict(X_test)


# In[128]:


print(classification_report(y_test, pred1))


# In[129]:


from sklearn.neighbors import KNeighborsClassifier


# In[130]:


knn = KNeighborsClassifier()


# In[131]:


knn.fit(X_train, y_train)


# In[132]:


pred2 = knn.predict(X_test)


# In[133]:


print(classification_report(y_test,pred2))


# In[134]:


print(confusion_matrix(y_test,pred2))

