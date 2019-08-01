#Practising my ML skills on the Titanic Survival data set.
#I am using multiple classifiers and then picking the best performing classifier for the final prediction

#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.getcwd()


# In[3]:


os.chdir('./Desktop/DS/Kaggle Titanic Data')


# In[120]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[129]:


submission=pd.DataFrame(test['PassengerId'])


# In[121]:


train.head()


# In[122]:


test.head()


# In[7]:


train.info()


# In[8]:


train.describe()


# In[9]:


test.info()


# In[10]:


test.describe()


# In[11]:


train.Survived.value_counts(normalize=True).plot.bar()


# In[12]:


train.Pclass.value_counts(normalize=True).plot.bar()


# In[13]:


sns.distplot(train['Age'][train['Age'].isna()==False])


# In[14]:


train.SibSp.value_counts()


# In[15]:


train.Parch.value_counts()


# In[16]:


train.groupby('Pclass')['Survived'].mean().plot.bar()


# In[17]:


survived_class=pd.crosstab(train['Pclass'],train['Survived'])
survived_class.div(survived_class.sum(1),axis=0).plot(kind='bar',stacked=True)
plt.yticks(np.arange(0,1.1,0.1))
plt.xlabel('Passenger Class')
plt.title('Survival rate of Passenger Classes')
plt.show()


# In[18]:


train.groupby('Sex')['Survived'].value_counts()


# In[19]:


train.groupby('Sex')['Survived'].mean().plot.bar()


# In[20]:


survived_sex=pd.crosstab(train['Sex'],train['Survived'])
survived_sex.div(survived_sex.sum(1),axis=0).plot(kind='bar',stacked=True)
plt.yticks(np.arange(0,1.1,0.1))
plt.legend(loc='upper center')


# In[21]:


train.loc[:,'Age']=train.loc[:,'Age'].apply(lambda x: train['Age'].mean() if np.isnan(x)==True else x)
test.loc[:,'Age']=test.loc[:,'Age'].apply(lambda x: test['Age'].mean() if np.isnan(x)==True else x)


# In[33]:


train['Age']=np.ceil(train['Age'])
test['Age']=np.ceil(test['Age'])


# In[23]:


train.info()


# In[24]:


train.loc[:,'Embarked'][train.loc[:,'Embarked'].isna()==True]


# In[25]:


train.loc[:,'Embarked']=train.loc[:,'Embarked'].apply(lambda x: train['Embarked'].mode() if pd.isna(x)==True else x)
test.loc[:,'Embarked']=test.loc[:,'Embarked'].apply(lambda x: test['Embarked'].mode() if pd.isna(x)==True else x)


# In[26]:


train.info()


# In[27]:


train['Sex'].replace({'male':0,'female':1},inplace=True)
test['Sex'].replace({'male':0,'female':1},inplace=True)


# In[294]:


train['Embarked'][829]


# In[28]:


dummies=pd.get_dummies(train['Embarked'].str.get(0))
test_dummies=pd.get_dummies(test['Embarked'].str.get(0))


# In[29]:


dummies


# In[30]:


test_dummies


# In[31]:


X=train[['Pclass','Sex','Age','SibSp','Parch']]


# In[35]:


test=test[['Pclass','Sex','Age','SibSp','Parch']]


# In[37]:


X.head()


# In[38]:


test.head()


# In[39]:


dummies=dummies.iloc[:,[0,1]]
test_dummies=test_dummies.iloc[:,[0,1]]


# In[40]:


X=pd.concat([X,dummies],axis=1,)
test=pd.concat([test,test_dummies],axis=1)


# In[41]:


X.head()


# In[42]:


test.head()


# In[43]:


y=train['Survived']


# In[111]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(criterion="entropy"),
    RandomForestClassifier(n_estimators=1000,criterion="entropy"),
    AdaBoostClassifier(learning_rate=0.5),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]


# In[112]:


sss=StratifiedShuffleSplit()

scores={}


# In[113]:


for clf in classifiers:
    clf_name=clf.__class__.__name__
    
    for train_index,test_index in sss.split(X,y):
        X_train,X_test,y_train,y_test=X.iloc[train_index],X.iloc[test_index],y.iloc[train_index],y.iloc[test_index]
        clf.fit(X_train,y_train)
        prediction=clf.predict(X_test)
        acc=accuracy_score(y_test,prediction)
        
        if clf_name in scores:
            scores[clf_name]+=(acc)/10
        else:
            scores[clf_name]=(acc)/10
    


# In[114]:


scores


# In[105]:


log=pd.DataFrame(columns=['Algorithm','Scores'])


# In[106]:


log


# In[107]:


for i in scores:
    log=log.append(pd.DataFrame([[i,scores[i]]],columns=['Algorithm','Scores']))


# In[108]:


log


# In[109]:


sns.barplot('Scores','Algorithm',data=log)


# In[118]:


final_classifier=RandomForestClassifier(n_estimators=1000,criterion="entropy")
final_classifier.fit(X,y)
test_predicted=final_classifier.predict(test)


# In[119]:


test_predicted


# In[131]:


submission['Survival']=test_predicted


# In[150]:


submission.rename({'Survival':'Survived'},axis='columns',inplace=True)


# In[153]:


submission.to_csv('submission.csv',index=False)

