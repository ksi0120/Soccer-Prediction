#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier
from xgboost import plot_importance


# In[32]:


def eigen_decomposition(C_matrix, r):
    e_val, e_vec = np.linalg.eigh(C_matrix)
    e_vec = e_vec[:, np.argsort(-e_val)]
    e_val = e_val[np.argsort(-e_val)]
    eigenvector_matrix = e_vec[:,0:r]
    large_eigenvalue = e_val[:r]
    
    return eigenvector_matrix, large_eigenvalue

def PCA(data, SampleNum, r):
    m_X = np.sum(data, axis = 0)/SampleNum
    c_X = data - m_X
    C = (c_X.T@c_X)/SampleNum

    e_vec, e_val = eigen_decomposition(C, r)
    eigen_diag_reduce = np.diag(e_val**(-1/2))
    eigen_diag_reduce = np.diag(e_val**(-1/2))
    X_pca = (eigen_diag_reduce@e_vec.T@c_X.T).T
    
    return X_pca

def Plot_dataPoint(data, label, title):
    cmap_bold = ListedColormap(['coral', 'forestgreen'])

    x_min, x_max = data[0].min(), data[0].max() 
    y_min, y_max = data[1].min(), data[1].max()

    plt.figure()
    
    plt.title(title)
    plt.xlabel("Feature-1",fontsize=10)
    plt.ylabel("Feature-2",fontsize=10)

    plt.scatter(data[0], data[1], c = label, edgecolor='k', cmap = cmap_bold)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)


# In[24]:


training=pd.read_csv("training.csv")
testing=pd.read_csv('test_new.csv')


# In[25]:


training_re = training[training["FTR"] != 'D']
X_train_re=training_re.iloc[:,4:16]
y_train_re=training_re.iloc[:,18]

testing_re = testing[testing["FTR"] != 'D']
X_test_re=testing_re.iloc[:,4:16]
y_test_re=testing_re.iloc[:,18]

scaler=preprocessing.StandardScaler().fit(X_train_re)
X_train_re=scaler.transform(X_train_re)
X_test_re=scaler.transform(X_test_re)


# In[31]:


# pca for plot (Home win - 'coral', Draw - 'dodgerblue', Away win - 'forestgreen')
X_train_pca = pd.DataFrame(PCA(X_train_re, len(X_train_re), 2))
X_test_pca = pd.DataFrame(PCA(X_test_re, len(X_test_re), 2))
y_train_pca = pd.factorize(y_train_re)
Plot_dataPoint(X_train_pca, y_train_pca[0], 'Training Data Plot - 2D(PCA)')


# In[10]:


NB = GaussianNB().fit(X_train_re, y_train_re)
NB_pred=NB.predict(X_test_re)


# In[11]:


labels = np.unique(y_test_re)
pd.DataFrame(confusion_matrix(y_test_re, NB_pred, labels=labels), index=labels, columns=labels)


# In[12]:


print("Accuracy:",accuracy_score(y_test_re,NB_pred))


# In[13]:


LR = LogisticRegression(solver = 'newton-cg').fit(X_train_re, y_train_re)
LR_pred=LR.predict(X_test_re)
pd.DataFrame(confusion_matrix(y_test_re, LR_pred, labels=labels), index=labels, columns=labels)


# In[14]:


print("Accuracy:",accuracy_score(y_test_re,LR_pred))


# In[34]:


def RF_Parameter(data, label):
    parameters = {'n_estimators':range(800,1000,5)}
    Grid = GridSearchCV(RandomForestClassifier(random_state=1), parameters, cv=5).fit(data, label)    
    Best_parms = list(Grid.best_params_.values())
    return Best_parms

#Result: n_estimator='860'
#RF_parms = RF_Parameter(X_train_re, y_train_re)


# In[15]:


RF = RandomForestClassifier(n_estimators = 860, max_features = 'auto', random_state=1).fit(X_train_re, y_train_re)
RF_pred=RF.predict(X_test_re)
pd.DataFrame(confusion_matrix(y_test_re, RF_pred, labels=labels), index=labels, columns=labels)


# In[16]:


print("Accuracy:",accuracy_score(y_test_re,RF_pred))


# In[37]:


def XGB_Parameter(data, label):
    parameters = {'max_depth': range(5,10),'n_estimators': range(800,1000,5)}
    Grid = GridSearchCV(XGBClassifier(), parameters, cv=5).fit(data, label)    
    Best_parms = list(Grid.best_params_.values())
    return Best_parms

#Result: max_depth=5, n_estimators=960
#XGB_parms = XGB_Parameter(X_train_re, y_train_re)


# In[39]:


XGB = XGBClassifier(learning_rate=0.01, max_depth=5, n_estimators = 960).fit(X_train_re, y_train_re)
XGB_pred=XGB.predict(X_test_re)
pd.DataFrame(confusion_matrix(y_test_re, XGB_pred, labels=labels), index=labels, columns=labels)


# In[40]:


print("Accuracy:",accuracy_score(y_test_re,XGB_pred))


# In[19]:


var_index_1 = ['PV_H','PV_A','TV_H','TV_A','A_H','M_H','D_H','G_H','A_A','M_A','D_A','G_A']
XGB.get_booster().feature_names = var_index_1

plot_importance(XGB)
plt.show()


# In[20]:


# feature extraction
ETC = ExtraTreesClassifier().fit(X_train_re, y_train_re)
pd.DataFrame(np.round(ETC.feature_importances_*100,2),index = var_index_1).T


# In[21]:


# feature extraction
SKB = SelectKBest(score_func=f_classif, k=4).fit(X_train_re, y_train_re)
pd.DataFrame(np.round(SKB.scores_,0),index = var_index_1).T


# In[ ]:




