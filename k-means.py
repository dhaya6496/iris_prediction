#!/usr/bin/env python
# coding: utf-8

# # problem statement

# ## using iris data to predict the type of flowers

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


#read_data
data = pd.read_csv('Iris.csv')
data.head()
data.drop('Unnamed: 0',axis=1,inplace=True)
data.head()

data.info()

data.shape


# In[8]:


data.describe()


# In[9]:


data.dtypes
data['target']= data['target'].map({'0':'Iris','1':'Versicolor','2':'Setosa'})


# In[10]:


data.duplicated().sum()


# In[11]:


sns.boxplot(x=data['sepal_length'])


# In[12]:


sns.boxplot(x=data['sepal_width'])


# In[13]:


sns.distplot(x=data['sepal_width'],kde=True)


# In[14]:


low_limit = data['sepal_width'].mean()-3*data['sepal_width'].std()
low_limit


# In[15]:


up_limit = data['sepal_width'].mean()+3*data['sepal_width'].std()
up_limit


# In[16]:


data.loc[data['sepal_width']<low_limit]


# In[17]:


len(data.loc[data['sepal_width']>up_limit].shape)


# In[18]:


data.shape


# In[19]:


2/150*100


# In[20]:


data.loc[data['sepal_width']>up_limit,'sepal_width']= data['sepal_width'].mean()


# In[21]:


from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
data[data.columns]= scaling.fit_transform(data[data.columns])


# In[22]:


data.head()


# In[23]:


data.isnull().sum()


# In[25]:


data.duplicated().sum()


# In[26]:


x= data.drop('target',axis=1)


# In[27]:


y= data.target


# In[29]:


from sklearn.cluster import KMeans


# In[43]:


model = KMeans(n_clusters=3,random_state=10)


# In[44]:


model.fit(x)


# In[45]:


y_pred = model.predict(x)


# In[46]:


y_pred


# In[47]:


model.cluster_centers_


# In[48]:


model.labels_


# In[39]:


data['labels']= pd.DataFrame(model.labels_)


# In[49]:


data


# In[50]:


from sklearn.metrics import silhouette_score


# In[51]:


silhouette_score(x,model.labels_)


# In[52]:


wcss=[]
for i in range(1,10):
    model= KMeans(n_clusters=i,random_state=20)
    model.fit(x)
    wcss.append(model.inertia_)


# In[54]:


plt.plot(range(2,11),wcss)
plt.title('elbow method')
plt.xlabel('k')
plt.ylabel('wcss')


# In[69]:


model = KMeans(n_clusters=3,random_state=10)


# In[70]:


model.fit(x)


# In[71]:


model.predict(x)

filename = 'iris_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# In[72]:


silhouette_score(x,model.labels_)


# In[ ]:




