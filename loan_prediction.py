#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://s3.ap-southeast-1.amazonaws.com/images.asianage.com/images/aa-Cover-duc2lce1dduh1qm0bg2vlts690-20180226061846.Medi.jpeg")


# In[3]:


#loading libaries
import numpy as np
import pandas as pd
import seaborn as sns
import os
sns.set()
import matplotlib.pyplot as plt
from  matplotlib import pyplot 
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import chi2_contingency
from scipy.special import erfc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
# from sklearn.datasets import make_classification


# In[4]:


#load the data
loan = pd.read_csv('C:/edwisor/bank-loan.csv')


# In[5]:


os.chdir('C:/edwisor')


# In[6]:


os.getcwd()


# In[7]:


#number of rows and columns
loan.shape


# # Explatory data analysis

# In[8]:


loan.columns


# In[9]:


loan.head()


#  1.Age Age of each customer Numerical
#  2.Education Education categories Categorical
#  3.Employment Employment status - Numerical Corresponds to job status and being converted to numeric format
#  4.Address Geographic area - Numerical Converted to numeric values 5 Income Gross Income of each Numerical customer
#  5.debtinc Individual’s debt Numerical payment to his or her gross income
#  6.creddebt debt-to-credit ratio is a Numerical measurement of how much you owe your creditors as a percentage of your  available credit (credit limits)
#  7.othdebt Any other debts Numerical

# In[10]:


#datatypes of the data
loan.dtypes


# In[11]:


loan.info()


# In[12]:


#descriptive statistics
loan.describe()


# In[13]:


#scatterplot
sns.jointplot(loan['othdebt'] , loan['income'])


# We can see the positive relationship between variables

# In[14]:


#stripplot
sns.stripplot(loan['default'], loan['income'], jitter=True)


# # Data Pre-Processing
# Missing Value Analysis

# In[15]:


#checking missing-value
loan.isnull().sum()


# In[16]:


#remove missing value
loan.dropna(inplace = True)


# In[17]:


loan.info()


# # Outlier analysis

# In[18]:


# convert ed  and default variable into object datatype
loan['ed']=loan['ed'].astype(object)
loan['default']=loan['default'].astype(object)


# In[19]:


loan.shape


# In[20]:


#select only numeric
cnames = loan.select_dtypes(include=np.number)


# In[21]:


#plot boxplot
f , ax = plt.subplots(figsize =(20,20))
fig = sns.boxplot(data =cnames)


# In[22]:


# #Detect and delete outliers from data

for i in cnames:
    print(i)
    q75, q25 = np.percentile(loan.loc[:,i], [75,25])
    iqr = q75 - q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print(min)
    print(max)
 
    loan = loan.drop(loan[loan.loc[:,i] < min].index)
    loan = loan.drop(loan[loan.loc[:,i] > max].index)
   


# In[23]:



loan.shape


# # feature selection
# 
# 

# In[24]:


# Heat map analysis


# In[25]:


#select only numeric
cnames = loan.select_dtypes(include=np.number)


# In[26]:


#correlation analysis

corr = loan.corr()
f,ax = plt.subplots(figsize=(10, 5))
plt.title('Heatmap',y = 1 , size = 16)
sns.heatmap(cnames.corr(), annot=True, fmt= '.1f',ax=ax, cmap="BrBG")
sns.set(font_scale=1.25)
plt.show()


#  Here, no correlation between variables
# 
# 
#  # Chi-square test

# In[27]:


#select categorical data
c_names = loan[['ed']]
#Chisquare test of independence
    
    #loop for chi square values
for i in c_names:
        print(i)
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(loan['default'], loan[i]))
        print(p)


# Here, p-value is very low So, we can remove this feature

# In[28]:


# drop a variable
loan = loan.drop('ed',axis = 1)


# In[29]:


loan.info


# # Feature Scaling 

# In[30]:


#To check distribution-Skewness

sns.distplot(loan['income']);
#This shows age independent variable is right skewed/positively skewed.


# In[31]:


#Normality check
# %matplotlib inline  
#plot histogram to check normalisation
loan.hist(grid = False,figsize =(10,10),bins=20)


# In[32]:


#select only numeric
cnames = loan.select_dtypes(include=np.number)


# In[33]:


#normalisation
for i in cnames:
    print(i)
    loan[i] = (loan[i] - (loan[i].min()))/((loan[i].max()) - (loan[i].min()))


# In[34]:


loan.dtypes


# In[35]:


loan['default']=loan['default'].astype(int)


# # Model development

# In[36]:


import seaborn as sns


# In[37]:


sns.pairplot(loan , hue="default" , diag_kind='kde')


# In[38]:


loan.shape


# In[39]:


#loan['default'] = loan['default'].replace(1,'Defaulted')
#loan['default'] = loan['default'].replace(0,'Non-defaulted')


# # Separate X and Y

# In[40]:


X = loan.drop('default',axis = 1)
y = loan[['default']]


# # Train-Test Split

# In[41]:


#dividing data into train and test

from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train , X_test , y_train , y_test = train_test_split(X, y , test_size = 0.2,  random_state =21)


# In[42]:


# fit train data
lr = LogisticRegression().fit(X_train , y_train)


# In[43]:


print(lr.score(X_test,y_test))


# In[44]:


# predict new test cases
lr_pred = lr.predict(X_test)


# In[45]:


from sklearn.metrics import confusion_matrix



# In[47]:


# Build confusion matrix
CM = confusion_matrix(y_test, lr_pred)


# In[49]:


#let us save TP, TN, FP, FN
TN = CM[0,0]
FN = CM[1,0]
TP = CM[1,1]
FP = CM[0,1]


# In[50]:


#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)



# In[51]:


#False Negative rate 
(FN*100)/(FN+TP)


# In[52]:


#True positive rate(Recall)
TPR= TP/(TP+FN)*100
TPR


# In[53]:


FPR = FP/(TN+FP)*100
FPR


# In[54]:


# True negative rate(specifity)
TNR = TN/(TN+FP)*100
TNR


# In[55]:


print("Defaulted", sum(lr_pred!=0))
print("Non-defaulted ", sum(lr_pred==0))
#Results
CM


# In[57]:


fpr, tpr, thresh = metrics.roc_curve(y_test, lr_pred)


# In[58]:


auc = metrics.auc(fpr, tpr)
print("AUC:", auc)


# In[59]:


# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='default')
plt.title('ROC curve')
# axis label
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
# show the plot
plt.legend()
# show the plot
plt.show()


# In[ ]:


#Accuracy score=80.73
# TPR(Recall)=29.62
# FNR=70.37
# FPR=2.43
# TNR(Specifity)=97.56


# # Decision trees

# In[60]:


# fit train data
dc = DecisionTreeClassifier(max_depth=3).fit(X_train , y_train)


# In[61]:


# predict new test cases
dc_pred = dc.predict(X_test)


# In[62]:


from sklearn import tree


# In[63]:


print(dc.score(X_test,y_test))


# In[64]:


# Build confusion matrix
CM = confusion_matrix(y_test, dc_pred)
CM


# In[65]:


#let us save TP, TN, FP, FN
TN = CM[0,0]
FN = CM[1,0]
TP = CM[1,1]
FP = CM[0,1]


# In[66]:


#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))


# In[67]:


#False Negative rate 
print((FN*100)/(FN+TP))


# In[68]:


#True positive rate(Recall)
TPR= TP/(TP+FN)*100
TPR


# In[69]:


FPR = FP/(TN+FP)*100
FPR


# In[70]:


# True negative rate(specifity)
TNR = TN/(TN+FP)*100
TNR


# In[71]:


print("Defaulted", sum(dc_pred!=0))
print("Non-defaulted ", sum(dc_pred==0))
#Results
CM


# In[72]:


fpr, tpr, thresh = metrics.roc_curve(y_test, dc_pred)


# In[73]:


auc = metrics.auc(fpr, tpr)
print("AUC:", auc)


# In[74]:


# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='default')
plt.title('ROC curve')
# axis label
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
# show the legend
plt.legend()
# show the plot
plt.show()


# In[75]:


#Accuracy score=77.98
# TPR(Recall)=29.62
# FNR=70.37
# FPR=6.09
# TNR(Specifity)=93.90


# # Random Forest

# In[76]:


# fit train data
rf = RandomForestClassifier(n_estimators=500, bootstrap= True).fit(X_train , y_train)


# In[77]:


# predict new test cases
rf_pred = rf.predict(X_test)


# In[78]:


print(rf.score(X_test,y_test))


# In[79]:


# Build confusion matrix
CM = confusion_matrix(y_test, rf_pred)
CM


# In[80]:


#let us save TP, TN, FP, FN
TN = CM[0,0]
FN = CM[1,0]
TP = CM[1,1]
FP = CM[0,1]


# In[83]:


#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))



# In[84]:


#False Negative rate 
print((FN*100)/(FN+TP))


# In[85]:


#True positive rate(Recall)
TPR= TP/(TP+FN)*100
TPR


# In[86]:


FPR = FP/(TN+FP)*100
FPR


# In[87]:


# True negative rate(specifity)
TNR = TN/(TN+FP)*100
TNR


# In[88]:


print("Defaulted", sum(dc_pred!=0))
print("Non-defaulted ", sum(dc_pred==0))
#Results
CM


# In[89]:


fpr, tpr, thresh = metrics.roc_curve(y_test, rf_pred)


# In[90]:


auc = metrics.auc(fpr, tpr)
print("AUC:", auc)


# In[91]:


# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='default')
plt.title('ROC curve')
# axis label
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend()
plt.show()


# In[ ]:


#Accuracy score=81.65
# TPR(Recall)=37.03
# FNR=59.25
# FPR=4.87
# TNR(Specifity)=95.12


# # Naive Bayes

# In[92]:



from sklearn.naive_bayes import GaussianNB


# In[93]:


# fit train data
NB_model = GaussianNB().fit(X_train, y_train)


# In[95]:


#predict test cases
NB_Predictions = NB_model.predict(X_test)


# In[96]:


print(NB_model.score(X_test,y_test))


# In[97]:


# Build confusion matrix
CM = confusion_matrix(y_test,NB_Predictions)
CM


# In[98]:


#let us save TP, TN, FP, FN
TN = CM[0,0]
FN = CM[1,0]
TP = CM[1,1]
FP = CM[0,1]


# In[99]:


#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))


# In[100]:


#False Negative rate 
print((FN*100)/(FN+TP))


# In[101]:


#True positive rate(Recall)
TPR= TP/(TP+FN)*100
TPR


# In[102]:


FPR = FP/(TN+FP)*100
FPR


# In[103]:


# True negative rate(specifity)
TNR = TN/(TN+FP)*100
TNR


# In[104]:


print("Defaulted", sum(NB_Predictions!=0))
print("Non-defaulted ", sum(NB_Predictions==0))
#Results
CM


# In[105]:


fpr, tpr, thresh = metrics.roc_curve(y_test, NB_Predictions)


# In[106]:


auc = metrics.auc(fpr, tpr)
print("AUC:", auc)


# In[ ]:


# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='default')
plt.title('ROC curve')
# axis label
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
# show the legend
plt.legend()
# show the plot
plt.show()


# In[ ]:


#Accuracy score=86.23853211009174
# TPR(Recall)=66.66
# FNR=33.33
# FPR=7.31
# TNR(Specifity)=92.68


# # Here, we can see above model shows higher accuracy among all the model so thats why we will freeze Naive bayes

# # Prediction on test .csv

# In[107]:


# predicted values into dataframe
NB_Predictions = pd.DataFrame(NB_Predictions) 


# In[108]:


NB_Predictions.to_csv('bank -loan python.csv',header= True , index= False)


# In[114]:


#saving X_test into directory
X_test.to_csv("xtest_final.csv", index = False , header= True)


# In[110]:


df = pd.read_csv('xtest_final.csv')
df


# In[111]:


#joining two dataframes
final_result = pd.concat([df, NB_Predictions], axis=1)


# In[112]:


#renaming column name
final_result.rename(columns={0: 'default'},inplace = True)


# In[113]:


#saving result in csv format
final_result.to_csv('loan_final_result_.csv',header= True , index= False)


# In[ ]:




