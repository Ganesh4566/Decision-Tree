#!/usr/bin/env python
# coding: utf-8

# # Decision Tree  on Iris Dataset

# In[1]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


# In[2]:


from sklearn.datasets import load_iris


# In[4]:


from sklearn.tree import DecisionTreeClassifier


# In[5]:


iris_data=load_iris()


# In[7]:


iris=pd.DataFrame(iris_data.data)


# In[8]:


iris


# In[9]:


print('features',iris_data.feature_names)


# In[11]:


iris.shape


# In[15]:


X=iris.values[:,0:4]
Y=iris_data.target
print(X)
print(Y)


# In[16]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=3,random_state=3)


# In[18]:


clf=DecisionTreeClassifier(random_state=5)


# In[19]:


clf.fit(X_train,Y_train)


# In[25]:


X=[[6.5,1.6,6.6,2.1]]
Y_pred=clf.predict(X)
print(Y_pred)

Y_pred=clf.predict(X_test)


# In[26]:


from sklearn.metrics import accuracy_score
print('accuracy:',accuracy_score(Y_test,Y_pred))


# In[31]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
cm


# # Plot a Decision Tree

# In[33]:


from sklearn import tree
tree.plot_tree(clf)


# In[34]:


text_representation=tree.export_text(clf)
print(text_representation)


# In[36]:


import matplotlib.pyplot as plt
plt.savefig('tree.png',format='png',bbox_inches='tight')


# In[ ]:




