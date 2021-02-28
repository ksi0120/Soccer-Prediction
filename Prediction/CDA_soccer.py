#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[21]:


def Decision_Boundary(classifier, data, label):
    h = .01
    cmap_light = ListedColormap(['peachpuff',  'lightskyblue', 'mediumaquamarine'])
    cmap_bold = ListedColormap(['coral',  'dodgerblue', 'forestgreen'])

    x_min, x_max = data[0].min(), data[0].max() 
    y_min, y_max = data[1].min(), data[1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)

    plt.xlabel("Feature-1",fontsize=10)
    plt.ylabel("Feature-2",fontsize=10)

    plt.scatter(data[0], data[1], c = label, edgecolor='k', cmap = cmap_bold)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    
def Plot_dataPoint(data, label, title):
    cmap_bold = ListedColormap(['coral',  'dodgerblue', 'forestgreen'])

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
X_train=training.iloc[:,4:16]
y_train=training.iloc[:,18]

testing=pd.read_csv('test_new.csv')
X_test=testing.iloc[:,4:16]
y_test=testing.iloc[:,18]

labels = np.unique(y_test)


# In[25]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# In[48]:


# pca for plot (Home win - 'coral', Draw - 'dodgerblue', Away win - 'forestgreen')
X_train_pca = pd.DataFrame(PCA(X_train, len(X_train), 2))
X_test_pca = pd.DataFrame(PCA(X_test, len(X_test), 2))
y_train_pca = pd.factorize(y_train)
Plot_dataPoint(X_train_pca, y_train_pca[0], 'Training Data Plot - 2D(PCA)')


# ###### 1. Naive Bayes

# In[27]:


NB = GaussianNB().fit(X_train, y_train)
NB_y_pred=NB.predict(X_test)


# In[33]:


pd.DataFrame(confusion_matrix(y_test, NB_y_pred, labels=labels), index=labels, columns=labels)


# In[34]:


print("Accuracy:",accuracy_score(y_test,NB_y_pred))


# ###### 2. Logistic Regression

# In[35]:


def LR_Parameter(data, label):
    parameters = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']}
    Grid = GridSearchCV(LogisticRegression(), parameters, cv=10).fit(data, label)    
    Best_parms = list(Grid.best_params_.values())
    return Best_parms

#Result: solve='newton-cg'
#LR_parms = LR_Parameter(X_train, y_train)


# In[36]:


LR = LogisticRegression(solver = 'newton-cg').fit(X_train, y_train)
LR_y_pred=LR.predict(X_test)


# In[39]:


pd.DataFrame(confusion_matrix(y_test, LR_y_pred, labels=labels), index=labels, columns=labels)


# In[40]:


print("Accuracy:",accuracy_score(y_test,LR_y_pred))


# ###### 3. Random Forest

# In[220]:


def RF_Parameter(data, label):
    parameters = {'n_estimators':range(500,1000,20)}
    Grid = GridSearchCV(RandomForestClassifier(random_state=1), parameters, cv=5).fit(data, label)    
    Best_parms = list(Grid.best_params_.values())
    return Best_parms

#Result: n_estimator=''
#RF_parms = RF_Parameter(X_train, y_train)


# In[69]:


RF = RandomForestClassifier(n_estimators = 920, max_features = 'auto', random_state=1).fit(X_train, y_train)
RF_y_pred = RF.predict(X_test)


# In[70]:


pd.DataFrame(confusion_matrix(y_test, RF_y_pred, labels=labels), index=labels, columns=labels)


# In[71]:


print("Accuracy:",accuracy_score(y_test,RF_y_pred))


# In[45]:


# pca for plot (working..)
y_train_plot = pd.factorize(y_train)
y_test_plot = pd.factorize(y_test)

y_train_plot = pd.DataFrame(y_train_plot[0])
y_test_plot = pd.DataFrame(y_test_plot[0])


# In[49]:


RF_pca = RandomForestClassifier(n_estimators = 920, max_features = 'auto', random_state=1).fit(X_train_pca, y_train_pca[0])
RF_y_pred_pca = RF_pca.predict(X_test_pca)

Decision_Boundary(RF_pca, X_test_pca, y_test_plot[0])
pd.DataFrame(confusion_matrix(y_test_plot, RF_y_pred_pca), index=labels, columns=labels)


# ###### 4. XGBoost

# In[230]:


#for XGB tuning: https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e


# In[50]:


def XGB_Parameter(data, label):
    parameters = {'n_estimators': range(750,850,5)}
    Grid = GridSearchCV(XGBClassifier(), parameters, cv=5).fit(data, label)    
    Best_parms = list(Grid.best_params_.values())
    return Best_parms

#Result: max_depth=9, n_estimators=790
#XGB_parms = XGB_Parameter(X_train, y_train)


# In[66]:


XGB = XGBClassifier(learning_rate=0.01, max_depth=9, n_estimators = 800, gamma=5, colsample_bytree=0.5).fit(X_train, y_train)
y_pred = XGB.predict(X_test)


# In[67]:


pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels)


# In[68]:


result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[55]:


var_index_1 = ['PV_H','PV_A','TV_H','TV_A','A_H','M_H','D_H','G_H','A_A','M_A','D_A','G_A']
#var_index_2 = ['TV_H','TV_A','A_H','M_H','D_H','G_H','A_A','M_A','D_A','G_A']
XGB.get_booster().feature_names = var_index_1

plot_importance(XGB)
plt.show()


# In[84]:


# feature extraction
ETC = ExtraTreesClassifier(random_state=1).fit(X_train, y_train)
pd.DataFrame(np.round(ETC.feature_importances_*100,2),index = var_index_1).T


# In[85]:


# feature extraction
SKB = SelectKBest(score_func=f_classif, k=4).fit(X_train, y_train)
pd.DataFrame(np.round(SKB.scores_,0),index = var_index_1).T

