#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[9]:


df_train = pd.read_csv('../Titanic/titanic_datasets/train.csv')
df_test = pd.read_csv('../Titanic/titanic_datasets/test.csv')


# In[10]:


df_train.info()


# In[11]:


titles = df_train['Name'].apply(lambda X:X.split(' ')[1]).value_counts()[:4].index


# In[12]:


df_train['Title'] = df_train['Name'].apply(lambda X:X.split(' ')[1])
df_test['Title'] = df_test['Name'].apply(lambda X:X.split(' ')[1])

df_train['Title'] = df_train['Title'].apply(lambda X: X if X in titles else 'other')
df_test['Title'] = df_test['Title'].apply(lambda X: X if X in titles else 'other')


# In[13]:


# Handling Missing Data
columns_object = list(df_train.select_dtypes(include='object').columns)
columns_numeric = list(df_train.select_dtypes(exclude='object').columns)
columns_numeric, columns_object


# In[14]:


columns_numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
columns_object = ['Sex', 'Embarked', 'Title']


# In[15]:


from sklearn.impute import SimpleImputer


# In[16]:


imputer_numeric = SimpleImputer(strategy='median')
imputer_object = SimpleImputer(strategy='most_frequent')
imputer_numeric.fit(df_train[columns_numeric])
imputer_object.fit(df_train[columns_object])

df_train[columns_numeric] = imputer_numeric.transform(df_train[columns_numeric])
df_train[columns_object] = imputer_object.transform(df_train[columns_object])


# In[17]:


df_test[columns_numeric] = imputer_numeric.transform(df_test[columns_numeric])
df_test[columns_object] = imputer_object.transform(df_test[columns_object])


# In[18]:


df_train.isna().sum()


# In[19]:


df_test.isna().sum()


# In[20]:


def get_age_group(X):
    if X>50:
        return 'older'
    if X>25:
        return 'young'
    return 'kid'


# In[21]:


# BINNING => new => categorical
df_train['Age_group'] = df_train['Age'].apply(get_age_group)
df_test['Age_group'] = df_test['Age'].apply(get_age_group)


# In[22]:


columns_object.append('Age_group')


# In[23]:


columns_object


# In[ ]:





# In[24]:


# Feature Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(df_train[columns_object])


# In[25]:


temp = encoder.transform(df_train[columns_object]).toarray()
df_temp = pd.DataFrame(temp)
df_train = pd.concat([df_train,df_temp], axis=1)
columns_numeric.extend(list(df_temp.columns))


# In[26]:


temp = encoder.transform(df_test[columns_object]).toarray()
df_temp = pd.DataFrame(temp)
df_test = pd.concat([df_test,df_temp], axis=1)


# In[27]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[28]:


p = {'n_estimators':[25,50,100,150,200], 'max_depth':[2,3,4,5,6], 'min_samples_leaf':[3,4,5,6,7]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=1),p,n_jobs=-1,cv=5, verbose=3)
grid_cv.fit(df_train[columns_numeric], df_train['Survived'])


# In[29]:


grid_cv.best_estimator_


# In[30]:


model_final = RandomForestClassifier(random_state=1, max_depth=6, min_samples_leaf=3, n_estimators=150)
model_final.fit(df_train[columns_numeric], df_train['Survived'])


# In[31]:


y = model_final.predict(df_test[columns_numeric])


# In[32]:


df_test['Survived'] = y


# In[33]:


df_test[['PassengerId','Survived']].to_csv('sub2_dsml14.csv',index=False)


# In[ ]:




