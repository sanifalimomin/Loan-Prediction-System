#!/usr/bin/env python
# coding: utf-8

# # DATA SCIENCE PROJECT - LOAN PREDICTION SYSTEM

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[2]:


train=pd.read_csv("train_dataset.csv")
train.head()


# In[3]:


train.info()


# # Exploratory Data Analysis

# In[4]:


train.describe()


# In[5]:


train['ApplicantIncome'].hist(bins=50)
plt.xlabel('ApplicantIncome')


# In[6]:


train['CoapplicantIncome'].hist(bins=50)
plt.xlabel('CoapplicantIncome')


# In[7]:


train['LoanAmount'].hist(bins=50)
plt.xlabel('Loanamount')


# In[8]:


train['Loan_Amount_Term'].hist(bins=50)
plt.xlabel('Looan _Amount_Term')


# In[9]:


train['Credit_History'].hist(bins=50)


# In[10]:


train.boxplot(column='ApplicantIncome')


# In[11]:


train.boxplot(column='LoanAmount')


# In[12]:


temp1 = train['Credit_History'].value_counts(ascending=True)
temp2 = train.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print ('Frequency Table for Credit History:') 
print (temp1)

print ('\nProbility of getting loan for each Credit History class:')
print (temp2)


# In[13]:


temp3=pd.crosstab(train['Credit_History'],train['Loan_Status'])
temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)


# In[14]:


temp3=pd.crosstab(train['Gender'],train['Loan_Status'])
temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)


# In[15]:


temp3=pd.crosstab(train['Dependents'],train['Loan_Status'])
temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)


# In[16]:


temp3=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)


# In[17]:


temp3=pd.crosstab(train['Married'],train['Loan_Status'])
temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)


# # Data Preparation

# In[18]:


#missing values
train.apply(lambda x:sum(x.isnull()),axis=0)


# In[19]:


train['Gender'].fillna('Male',inplace=True)
train['Married'].fillna('Yes',inplace=True)
train['Dependents'].fillna(0,inplace=True)
train['Self_Employed'].fillna('No',inplace=True)
train['Credit_History'].fillna(1,inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)
train['Loan_Amount_Term'].fillna(360,inplace=True)


# In[20]:


train.apply(lambda x:sum(x.isnull()),axis=0)


# In[21]:


train['LoanAmount'] = np.log(train['LoanAmount'])
train['LoanAmount'].hist(bins=20)


# In[22]:


train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
train['TotalIncome'] = np.log(train['TotalIncome'])
train['TotalIncome'].hist(bins=20) 


# In[23]:


train.dtypes


# In[24]:


train['Dependents']=train['Dependents'].astype('str')


# In[25]:


var_mod=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le=LabelEncoder()
for i in var_mod:
    train[i]=le.fit_transform(train[i])
print(train.dtypes)


# In[26]:


X=pd.concat([train.iloc[:,1:6],train.iloc[:,8:12],train.iloc[:,13:14]],axis=1)
X


# In[27]:


Y=train.iloc[:,12:13]
Y


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=21)


# # Applying Models

# In[29]:


#logistic Regression
modelLR=LogisticRegression(solver='liblinear')
modelLR.fit(X_train,y_train.values.ravel())
predictions=modelLR.predict(X_test)
accuracy=metrics.accuracy_score(predictions,y_test)
print("Logistic Regression: ")
print("Accuracy : %s" % "{0:.3%}\n".format(accuracy))


# In[30]:


modelKNN=KNeighborsClassifier(n_neighbors=6)
predictor=['Credit_History','Gender','Married','Education']
modelKNN.fit(X_train[predictor],y_train.values.ravel())
predictions=modelKNN.predict(X_test[predictor])
accuracy=metrics.accuracy_score(predictions,y_test)
print("KNN Model with 6 neighbours: ")
print("Accuracy : %s" % "{0:.3%}\n".format(accuracy))


# In[49]:


modelKNN=KNeighborsClassifier(n_neighbors=10)
predictor=['Credit_History','Gender','Married','Education']
modelKNN.fit(X_train[predictor],y_train.values.ravel())
predictions=modelKNN.predict(X_test[predictor])
accuracy=metrics.accuracy_score(predictions,y_test)
print("KNN Model with 10 neighbours: ")
print("Accuracy : %s" % "{0:.3%}\n".format(accuracy))


# In[32]:


modelDT=DecisionTreeClassifier()
predictor=['Credit_History','Gender','Married','Education']
modelDT.fit(X_train[predictor],y_train.values.ravel())
predictions=modelDT.predict(X_test[predictor])
accuracy=metrics.accuracy_score(predictions,y_test)
print("Decision Tree: ")
print("Accuracy : %s" % "{0:.3%}\n".format(accuracy))


# In[33]:


from sklearn.ensemble import BaggingClassifier
modelBKNN = BaggingClassifier(KNeighborsClassifier(),max_samples=0.50, max_features=0.5)
predictor=['Education','Loan_Amount_Term','Credit_History','Dependents','Property_Area']
modelBKNN.fit(X_train[predictor],y_train.values.ravel())
predictions=modelBKNN.predict(X_test[predictor])
accuracy=metrics.accuracy_score(predictions,y_test)
print("Bagging using KNN Model: ")
print("Accuracy : %s" % "{0:.3%}\n".format(accuracy))


# In[34]:


from sklearn.ensemble import BaggingClassifier
modelBDT = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.50, max_features=0.5)
predictor=['Education','Loan_Amount_Term','Credit_History','Dependents','Property_Area']
modelBDT.fit(X_train[predictor],y_train.values.ravel())
predictions=modelBDT.predict(X_test[predictor])
accuracy=metrics.accuracy_score(predictions,y_test)
print("Bagging using Decision Tree: ")
print("Accuracy : %s" % "{0:.3%}\n".format(accuracy))


# In[50]:


modelRF = RandomForestClassifier(n_estimators=50,min_samples_split=25,max_depth=7,max_features=1)
modelRF.fit(X_train,y_train.values.ravel())
predictions=modelRF.predict(X_test)
accuracy=metrics.accuracy_score(predictions,y_test)
print("Random Forest: ")
print("Accuracy : %s" % "{0:.3%}\n".format(accuracy))


# In[36]:


modelNB = GaussianNB()
predictor=['Credit_History','Gender','Married','Education']
modelNB.fit(X_train[predictor],y_train.values.ravel())
predictions=modelNB.predict(X_test[predictor])
accuracy=metrics.accuracy_score(predictions,y_test)
print("Naive Baise: ")
print("Accuracy : %s" % "{0:.3%}\n".format(accuracy))


# In[37]:


test=pd.read_csv('test_dataset.csv')
test.head()


# In[38]:


test.info()


# In[39]:


print(test.apply(lambda x:sum(x.isnull()),axis=0))
print(test['Gender'].value_counts(ascending=True))
print(test['Dependents'].value_counts(ascending=True))
print(test['Self_Employed'].value_counts(ascending=True))
print(test['Credit_History'].value_counts(ascending=True))


# In[40]:


test['Gender'].fillna('Male',inplace=True)
test['Dependents'].fillna(0,inplace=True)
test['Self_Employed'].fillna('No',inplace=True)
test['Credit_History'].fillna(1,inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].mean(),inplace=True)
test['Loan_Amount_Term'].fillna(360,inplace=True)


# In[41]:


print(test.apply(lambda x:sum(x.isnull()),axis=0))


# In[42]:


test['TotalIncome'] = test['ApplicantIncome'] + test['CoapplicantIncome']
X1_test=pd.concat([test.iloc[:,1:6],test.iloc[:,8:]],axis=1)
X1_test


# In[43]:


X1_test['Dependents']=X1_test['Dependents'].astype('str')
var_mod=['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le=LabelEncoder()
for i in var_mod:
    X1_test[i]=le.fit_transform(X1_test[i])
print(X1_test.dtypes)


# In[44]:


X1_test.describe()


# In[67]:


modelRF2 = RandomForestClassifier(n_estimators=50,min_samples_split=25,max_depth=7,max_features=1)
modelRF2.fit(X,Y.values.ravel())
testpredictions=modelRF.predict(X1_test)
print(testpredictions)


# In[70]:


testpredictions=testpredictions.astype('str')
for i in range(0,367):
    if( testpredictions[i]=='1') :
        testpredictions[i]='Y'
    elif(testpredictions[i]=='0'):
        testpredictions[i]='N'
print(testpredictions)


# In[71]:


df=pd.concat([test['Loan_ID'],pd.DataFrame({'Loan_Status':testpredictions})],axis=1)
df


# In[72]:


df.to_excel("Result_Submission.xlsx",engine='xlsxwriter')


# In[ ]:




