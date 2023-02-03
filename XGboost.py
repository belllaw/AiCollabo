#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from sklearn.metrics import classification_report


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

mpl.rc('font', family='NanumGothic') # 폰트 설정
mpl.rc('axes', unicode_minus=False) # 유니코드에서 음수 부호 설정

# 차트 스타일 설정
sns.set(font="NanumGothic", rc={"axes.unicode_minus":False}, style='darkgrid')
plt.rc("figure", figsize=(10,8))

warnings.filterwarnings("ignore")


# In[4]:


import xgboost as xgb
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('/Users/macbookair/Documents/Tuk_Folder/AIManu/project_02/dataset/train_normalization.csv')

X_features = dataset.iloc[:,0:501]
y_label = dataset.target


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size=0.25, random_state=156)

dtrain = xgb.DMatrix(data = X_train, label = y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)


# In[7]:


params = {
    "max_depth": 3, 
    "eta": 0.1, 
    "objective": "binary:logistic", 
    "eval_metric": "logloss",
    "early_stoppings": 100 
}

num_rounds = 5000 # 부스팅 반복 횟수


# In[8]:


# evals 파라미터에 train, test 셋을 명기하면 평가를 진행하면서 조기 중단을 적용 할 수 있다.
wlist = [(dtrain, "train"), (dtest, "eval")]

# 모델 학습: 사이킷런과 달리 train() 함수에 파라미터를 전달한다.
xgb_model = xgb.train(params = params, dtrain = dtrain, num_boost_round = num_rounds,evals = wlist)


# In[9]:


# 예측 확률
pred_probs = xgb_model.predict(dtest)
print("predict() 수행 결과값 10개만 표시")
print(np.round(pred_probs[:10], 3))

# 예측 분류
preds = [1 if x > 0.5 else 0 for x in pred_probs]
print("예측 분류 10개만 표시")
print(f"{preds[:10]}")


# In[10]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba_po=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba_po)
   
    print("오차 행렬")
    print(confusion)
    print(f"정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")


# In[12]:


get_clf_eval(y_test, pred = preds, pred_proba_po = pred_probs)


# In[13]:


#!pip3 install yellowbrick


# In[13]:


from xgboost import XGBClassifier
from yellowbrick.classifier import ROCAUC

xgb_basic = XGBClassifier()
xgb_basic.fit(X_train, y_train)

visualizer = ROCAUC(xgb_basic, classes=[0,1], micro=False, macro=False, per_class=False)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()


# In[14]:


print(classification_report(y_test, preds))


# In[15]:


# feature importance
from xgboost import plot_importance
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(20, 10))
plot_importance(xgb_model, ax=ax)


# In[2]:


#pip install graphviz


# In[17]:


# tree 시각환데 실행은 안됨
xgb.plot_tree(xgb_model,num_trees=0)
plt.rcParams['figure.figsize'] = [40, 10]
plt.show()

