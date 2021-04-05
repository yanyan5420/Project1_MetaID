#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from scipy.stats import uniform
from sklearn.base import clone
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from random import randint


# # Import Data and Prepare the splitted data index

train_data = pd.read_csv("combined_train_with_features.csv")
valid_data = pd.read_csv("combined_valid_with_features.csv")


def MAE(y, y_pred):
    # y, y_pred are numpy.ndarray
    mae = sum(abs(y-y_pred)) / len(y)
    return mae


def split_data_index(data, K):
    
    index_arr = data.index.values
    shuffle_index_arr = copy.copy(index_arr)
    
    random.seed(222)
    shuffle_index_arr = shuffle_index_arr.reshape(-1,K)
    for arr in shuffle_index_arr:
        random.shuffle(arr)
    
    cv_outer = []
    for i in range(K):

        test_index = shuffle_index_arr[:,i]
        train_index = np.delete(shuffle_index_arr, i, axis=1).reshape(1,-1)[0]

        cv_outer.append([train_index, test_index])
    
    return cv_outer

cv_outer = split_data_index(train_data, 10)


# # 1. Sigmoid

def sigmoid_cv(data, cv_outer):
    
    def sigmoid(x, L ,x0, k, a, b):
        y = L / (a + np.exp(-k*(x-x0)))+b
        return y
    
    MAE_results = []
    RMSE_results = []
    MedAE_results = []
    r2_results = []
    
    X = data['MolLogP'].values
    y = data['rtmed'].values
    for train_index, test_index in cv_outer:

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        
        p0 = [max(y), np.median(X), 1, 1, min(y)] 
        popt, pcov = curve_fit(sigmoid, X_train, y_train, p0, maxfev = 5000)
        
        y_pred = sigmoid(X_test, *popt)
        
        mae = MAE(y_test, y_pred)
        MAE_results.append(mae)
        
        rmse = mean_squared_error(y_test,y_pred,squared=False)
        RMSE_results.append(rmse)
        
        med = median_absolute_error(y_test, y_pred)
        MedAE_results.append(med)
        
        r2 = r2_score(y_test, y_pred)
        r2_results.append(r2)
        
    
    return MAE_results, RMSE_results, MedAE_results, r2_results


MAE_results, RMSE_results, MedAE_results, r2_results = sigmoid_cv(train_data, cv_outer)



def learning_curve_sigmoid(train_data, sample_seed):
    
    train_index_arr = train_data.index.values.reshape(10,-1)
    MAE_curve_list = []
    RMSE_curve_list = []
    MedAE_curve_list = []
    r_curve_list = []
    
    for i in range(20, 200, 20):

        num = int(i/10)
        np.random.seed(sample_seed)
        data_index = np.array(list(map(lambda x: list(np.random.choice(x, num, replace=False)), train_index_arr))).reshape(1,-1)[0]
        sample_data = train_data.iloc[list(data_index),:].reset_index(drop=True)
        cv_outer = split_data_index(sample_data, 10)
        MAE_results, RMSE_results, MedAE_results, r_results = sigmoid_cv(sample_data, cv_outer)

        MAE_curve_list.append(np.mean(MAE_results))
        RMSE_curve_list.append(np.mean(RMSE_results))
        MedAE_curve_list.append(np.mean(MedAE_results))
        r_curve_list.append(np.mean(r_results))
    
    return MAE_curve_list, RMSE_curve_list, MedAE_curve_list, r_curve_list


# # 2. SVR

# In[10]:


def scale_data(data):
    
    data_copy = data.copy()
    scaler = MinMaxScaler()
    data_copy.iloc[:,6:] = scaler.fit_transform(data_copy.iloc[:,6:])
    
    return data_copy


# In[11]:


scale_train_data = scale_data(train_data)
scale_valid_data = scale_data(valid_data)


# In[12]:


def svr_cv(cv_outer, data):
    
    MAE_results = []
    RMSE_results = []
    MedAE_results = []
    r2_results = []
    model_params = []
    
    for train_index, test_index in cv_outer:
        
        X_train, y_train = data.iloc[train_index,6:].values, data.iloc[train_index,3].values
        X_test, y_test = data.iloc[test_index,6:].values, data.iloc[test_index,3].values
        
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        
        model = SVR(kernel='rbf')
        params = {
                   'C': np.arange(100, 250, 10),
                    'epsilon': [0.0001, 0.001, 0.01],
                   'gamma' : [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
                    }

        search = RandomizedSearchCV(model,params,cv=cv_inner,scoring='neg_mean_absolute_error',verbose=0,n_jobs=-1,n_iter=100,refit=False,                                   random_state=0)
        
        search.fit(X_train, y_train)
        
        model_params.append(search.best_params_)
    
        model.set_params(**search.best_params_)
        model.fit(X_train, y_train)
#         print(search.best_params_)
        y_pred = model.predict(X_test)
        
        mae = MAE(y_test, y_pred)
        MAE_results.append(mae)
        
        rmse = mean_squared_error(y_test,y_pred,squared=False)
        RMSE_results.append(rmse)
        
        med = median_absolute_error(y_test, y_pred)
        MedAE_results.append(med)
        
        r2 = r2_score(y_test, y_pred)
        r2_results.append(r2)
        
    
    return MAE_results, RMSE_results, MedAE_results, r2_results, model_params


# In[13]:


cv_outer = split_data_index(scale_train_data, 10)
MAE_results, RMSE_results, MedAE_results, r2_results, model_params = svr_cv(cv_outer, scale_train_data)


# In[15]:


def learning_curve_svr(scale_train_data, sample_seed):
    
    train_index_arr = scale_train_data.index.values.reshape(10,-1)
    MAE_curve_list = []
    RMSE_curve_list = []
    MedAE_curve_list = []
    r_curve_list = []
    
    for i in range(20, 200, 20):
        
        num = int(i/10)
        np.random.seed(sample_seed)
        data_index = np.array(list(map(lambda x: list(np.random.choice(x, num, replace=False)), train_index_arr))).reshape(1,-1)[0]
        sample_data = scale_train_data.iloc[list(data_index),:].reset_index(drop=True)

        cv_outer = split_data_index(sample_data, 10)
        MAE_results, RMSE_results, MedAE_results, r2_results, model_params = svr_cv(cv_outer, sample_data)

        MAE_curve_list.append(np.mean(MAE_results))
        RMSE_curve_list.append(np.mean(RMSE_results))
        MedAE_curve_list.append(np.mean(MedAE_results))
        r_curve_list.append(np.mean(r2_results))
    
    return MAE_curve_list, RMSE_curve_list, MedAE_curve_list, r_curve_list


# # 3. AB

# In[16]:


def ab_cv(cv_outer, data):
    
    MAE_results = []
    RMSE_results = []
    MedAE_results = []
    r2_results = []
    model_params = []
    
    for train_index, test_index in cv_outer:
        
        X_train, y_train = data.iloc[train_index,6:].values, data.iloc[train_index,3].values
        X_test, y_test = data.iloc[test_index,6:].values, data.iloc[test_index,3].values
        
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

        model = AdaBoostRegressor(random_state=0)
        
        params = {'n_estimators': np.arange(10,500,10),
                  'learning_rate': np.arange(0.005, 0.5, 0.001),
                 }

        search = RandomizedSearchCV(model,params,cv=cv_inner,scoring='neg_mean_absolute_error',verbose=0,n_jobs=-1,n_iter=100,refit=False,                                   random_state=999)
        search.fit(X_train, y_train)
        
        model_params.append(search.best_params_)
    
        model.set_params(**search.best_params_)
        model.fit(X_train, y_train)
        print(search.best_params_)
        y_pred = model.predict(X_test)
        
        mae = MAE(y_test, y_pred)
        MAE_results.append(mae)
        
        rmse = mean_squared_error(y_test,y_pred,squared=False)
        RMSE_results.append(rmse)
        
        med = median_absolute_error(y_test, y_pred)
        MedAE_results.append(med)
        
        r2 = r2_score(y_test, y_pred)
        r2_results.append(r2)
        
    
    return MAE_results, RMSE_results, MedAE_results, r2_results, model_params
    


# In[17]:


cv_outer = split_data_index(train_data, 10)
MAE_results, RMSE_results, MedAE_results, r2_results, model_params = ab_cv(cv_outer, train_data)


# In[18]:


def learning_curve_ab(train_data, sample_seed):
    
    train_index_arr = train_data.index.values.reshape(10,-1)
    MAE_curve_list = []
    RMSE_curve_list = []
    MedAE_curve_list = []
    r_curve_list = []
    
    for i in range(20, 200, 20):
        
#         print(i)
        num = int(i/10)
        np.random.seed(sample_seed)
        data_index = np.array(list(map(lambda x: list(np.random.choice(x, num, replace=False)), train_index_arr))).reshape(1,-1)[0]
        sample_data = train_data.iloc[list(data_index),:].reset_index(drop=True)

        cv_outer = split_data_index(sample_data, 10)
        MAE_results, RMSE_results, MedAE_results, r2_results, model_params = ab_cv(cv_outer, sample_data)

        MAE_curve_list.append(np.mean(MAE_results))
        RMSE_curve_list.append(np.mean(RMSE_results))
        MedAE_curve_list.append(np.mean(MedAE_results))
        r_curve_list.append(np.mean(r2_results))
    
    return MAE_curve_list, RMSE_curve_list, MedAE_curve_list, r_curve_list


# # 4. GB

# In[19]:


def gb_cv(cv_outer, data):
    
    MAE_results = []
    RMSE_results = []
    MedAE_results = []
    r2_results = []
    model_params = []
    
    for train_index, test_index in cv_outer:
        
        X_train, y_train = data.iloc[train_index,6:].values, data.iloc[train_index,3].values
        X_test, y_test = data.iloc[test_index,6:].values, data.iloc[test_index,3].values
        
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

        model = GradientBoostingRegressor(random_state=222)
        
        params = {'n_estimators': np.arange(10,300,10),
                  'max_depth': np.arange(1, 5, 1),
                  'learning_rate': np.arange(0.005, 0.2, 0.001),   
                    }

        search = RandomizedSearchCV(model,params,cv=cv_inner,scoring='neg_mean_absolute_error',verbose=0,n_jobs=-1,n_iter=100,refit=False,                                   random_state=999)
        search.fit(X_train, y_train)
        
        model_params.append(search.best_params_)
    
        model.set_params(**search.best_params_)
        model.fit(X_train, y_train)
        print(search.best_params_)
        y_pred = model.predict(X_test)
        
        mae = MAE(y_test, y_pred)
        MAE_results.append(mae)
        
        rmse = mean_squared_error(y_test,y_pred,squared=False)
        RMSE_results.append(rmse)
        
        med = median_absolute_error(y_test, y_pred)
        MedAE_results.append(med)
        
        r2 = r2_score(y_test, y_pred)
        r2_results.append(r2)
        
    
    return MAE_results, RMSE_results, MedAE_results, r2_results, model_params


# In[20]:


cv_outer = split_data_index(train_data, 10)
MAE_results, RMSE_results, MedAE_results, r2_results, model_params = gb_cv(cv_outer, train_data)


# In[22]:


def learning_curve_gb(train_data, sample_seed):
    
    train_index_arr = train_data.index.values.reshape(10,-1)
    MAE_curve_list = []
    RMSE_curve_list = []
    MedAE_curve_list = []
    r_curve_list = []
    
    for i in range(20, 200, 20):
        
#         print(i)
        num = int(i/10)
        np.random.seed(sample_seed)
        data_index = np.array(list(map(lambda x: list(np.random.choice(x, num, replace=False)), train_index_arr))).reshape(1,-1)[0]
        sample_data = train_data.iloc[list(data_index),:].reset_index(drop=True)

        cv_outer = split_data_index(sample_data, 10)
        MAE_results, RMSE_results, MedAE_results, r2_results, model_params = gb_cv(cv_outer, sample_data)

        MAE_curve_list.append(np.mean(MAE_results))
        RMSE_curve_list.append(np.mean(RMSE_results))
        MedAE_curve_list.append(np.mean(MedAE_results))
        r_curve_list.append(np.mean(r2_results))
    
    return MAE_curve_list, RMSE_curve_list, MedAE_curve_list, r_curve_list


# # 5. Run 10 times

# In[25]:


random.seed(66)
sample_seed_list = random.sample(range(1000), k=10)

sigmoid_MAE = pd.DataFrame()
sigmoid_RMSE = pd.DataFrame()
sigmoid_MedAE = pd.DataFrame()
sigmoid_r = pd.DataFrame()

svr_MAE = pd.DataFrame()
svr_RMSE = pd.DataFrame()
svr_MedAE = pd.DataFrame()
svr_r = pd.DataFrame()

ab_MAE = pd.DataFrame()
ab_RMSE = pd.DataFrame()
ab_MedAE = pd.DataFrame()
ab_r = pd.DataFrame()

gb_MAE = pd.DataFrame()
gb_RMSE = pd.DataFrame()
gb_MedAE = pd.DataFrame()
gb_r = pd.DataFrame()

for seed in sample_seed_list:
    
    # sigmoid
    sigmoid_MAE_curve_list, sigmoid_RMSE_curve_list,sigmoid_MedAE_curve_list,    sigmoid_r_curve_list = learning_curve_sigmoid(train_data, seed)
    
    sigmoid_MAE[seed] = sigmoid_MAE_curve_list
    sigmoid_RMSE[seed] = sigmoid_RMSE_curve_list
    sigmoid_MedAE[seed] = sigmoid_MedAE_curve_list
    sigmoid_r[seed] = sigmoid_r_curve_list
    
    
    # svr
    svr_MAE_curve_list, svr_RMSE_curve_list, svr_MedAE_curve_list,    svr_r_curve_list = learning_curve_svr(scale_train_data, seed)
    
    svr_MAE[seed] = svr_MAE_curve_list
    svr_RMSE[seed] = svr_RMSE_curve_list
    svr_MedAE[seed] = svr_MedAE_curve_list
    svr_r[seed] = svr_r_curve_list
    
    # ab
    ab_MAE_curve_list, ab_RMSE_curve_list, ab_MedAE_curve_list,    ab_r_curve_list = learning_curve_ab(train_data, seed)
    
    ab_MAE[seed] = ab_MAE_curve_list
    ab_RMSE[seed] = ab_RMSE_curve_list
    ab_MedAE[seed] = ab_MedAE_curve_list
    ab_r[seed] = ab_r_curve_list
    
    
    # gb
    gb_MAE_curve_list, gb_RMSE_curve_list, gb_MedAE_curve_list,    gb_r_curve_list = learning_curve_gb(train_data, seed)
    
    gb_MAE[seed] = gb_MAE_curve_list
    gb_RMSE[seed] = gb_RMSE_curve_list
    gb_MedAE[seed] = gb_MedAE_curve_list
    gb_r[seed] = gb_r_curve_list
    
    print(seed)


# In[26]:


sigmoid_MAE['mean'] = sigmoid_MAE.iloc[:,:10].mean(axis=1)
sigmoid_MAE['std'] = sigmoid_MAE.iloc[:,:10].std(axis=1)

sigmoid_RMSE['mean'] = sigmoid_RMSE.iloc[:,:10].mean(axis=1)
sigmoid_RMSE['std'] = sigmoid_RMSE.iloc[:,:10].std(axis=1)

sigmoid_MedAE['mean'] = sigmoid_MedAE.iloc[:,:10].mean(axis=1)
sigmoid_MedAE['std'] = sigmoid_MedAE.iloc[:,:10].std(axis=1)

sigmoid_r['mean'] = sigmoid_r.iloc[:,:10].mean(axis=1)
sigmoid_r['std'] = sigmoid_r.iloc[:,:10].std(axis=1)


# In[27]:


sigmoid_MAE


# In[28]:


sigmoid = pd.DataFrame([sigmoid_MAE['mean'], sigmoid_RMSE['mean'], sigmoid_MedAE['mean'], sigmoid_r['mean']]).transpose()
sigmoid.columns = ['MAE','RMSE','MedAE','R2']


# In[29]:


sigmoid


# In[30]:


svr_MAE['mean'] = svr_MAE.iloc[:,:10].mean(axis=1)
svr_MAE['std'] = svr_MAE.iloc[:,:10].std(axis=1)

svr_RMSE['mean'] = svr_RMSE.iloc[:,:10].mean(axis=1)
svr_RMSE['std'] = svr_RMSE.iloc[:,:10].std(axis=1)

svr_MedAE['mean'] = svr_MedAE.iloc[:,:10].mean(axis=1)
svr_MedAE['std'] = svr_MedAE.iloc[:,:10].std(axis=1)

svr_r['mean'] = svr_r.iloc[:,:10].mean(axis=1)
svr_r['std'] = svr_r.iloc[:,:10].std(axis=1)


svr = pd.DataFrame([svr_MAE['mean'],svr_RMSE['mean'],svr_MedAE['mean'],svr_r['mean']]).transpose()
svr.columns = ['MAE','RMSE','MedAE','R2']


# In[31]:


svr_MAE


# In[32]:


svr


# In[33]:


ab_MAE['mean'] = ab_MAE.iloc[:,:10].mean(axis=1)
ab_MAE['std'] = ab_MAE.iloc[:,:10].std(axis=1)

ab_RMSE['mean'] = ab_RMSE.iloc[:,:10].mean(axis=1)
ab_RMSE['std'] = ab_RMSE.iloc[:,:10].std(axis=1)

ab_MedAE['mean'] = ab_MedAE.iloc[:,:10].mean(axis=1)
ab_MedAE['std'] = ab_MedAE.iloc[:,:10].std(axis=1)

ab_r['mean'] = ab_r.iloc[:,:10].mean(axis=1)
ab_r['std'] = ab_r.iloc[:,:10].std(axis=1)

ab = pd.DataFrame([ab_MAE['mean'],ab_RMSE['mean'],ab_MedAE['mean'],ab_r['mean']]).transpose()
ab.columns = ['MAE','RMSE','MedAE','R2']


# In[34]:


ab


# In[36]:


gb_MAE['mean'] = gb_MAE.iloc[:,:10].mean(axis=1)
gb_MAE['std'] = gb_MAE.iloc[:,:10].std(axis=1)

gb_RMSE['mean'] = gb_RMSE.iloc[:,:10].mean(axis=1)
gb_RMSE['std'] = gb_RMSE.iloc[:,:10].std(axis=1)

gb_MedAE['mean'] = gb_MedAE.iloc[:,:10].mean(axis=1)
gb_MedAE['std'] = gb_MedAE.iloc[:,:10].std(axis=1)

gb_r['mean'] = gb_r.iloc[:,:10].mean(axis=1)
gb_r['std'] = gb_r.iloc[:,:10].std(axis=1)

gb = pd.DataFrame([gb_MAE['mean'],gb_RMSE['mean'],gb_MedAE['mean'],gb_r['mean']]).transpose()
gb.columns = ['MAE','RMSE','MedAE','R2']


# In[37]:


gb_MAE


# In[38]:


gb


# In[39]:


sigmoid.to_csv("COM_LearningCurves_Sigmoid.csv", index=False)
svr.to_csv("COM_LearningCurves_SVR.csv", index=False)
ab.to_csv("COM_LearningCurves_AB.csv", index=False)
gb.to_csv("COM_LearningCurves_GB.csv", index=False)


# In[ ]:





# In[2]:


sigmoid = pd.read_csv("COM_LearningCurves_Sigmoid.csv")
svr = pd.read_csv("COM_LearningCurves_SVR.csv")
gb = pd.read_csv("COM_LearningCurves_GB.csv")
ab = pd.read_csv("COM_LearningCurves_AB.csv")


# In[3]:


from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family('serif')
font.set_name('Arial')
font.set_size('18')
font.set_weight('bold')

font1 = FontProperties()
font1.set_family('serif')
font1.set_name('Arial')
font1.set_size('14')


# In[6]:


fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(22,5))
fig.patch.set_facecolor('white')
ax1.grid(True, which='major', linestyle=':')
ax2.grid(True, which='major', linestyle=':')
ax3.grid(True, which='major', linestyle=':')
ax4.grid(True, which='major', linestyle=':')

X = range(20, 200, 20)
y1 = sigmoid.MAE
y2 = svr.MAE
y3 = gb.MAE
y4 = ab.MAE
ax1.plot(X, y1, 'd--', markersize=7.5, label='Sigmoid')
# ax.plot(X, y1, '--', color='darkgreen', label='SVR')
ax1.plot(X, y2, '^--', color='darkred', markersize=7.5, label='SVR')
ax1.plot(X, y3, 's--', color='darkgreen', markersize=7.5, label='GB')
ax1.plot(X, y4, 'o--', color='orange', markersize=7.5, label='AB')

# ax1.legend(bbox_to_anchor=(0.9, 0.95))

# fig.suptitle("Model Performance vs Training Size", y=0.96, fontsize=14, fontweight='bold')
ax1.set_xlabel("Number of training data", fontproperties=font1)
ax1.set_ylabel("MAE (min)", fontproperties=font1)



X = range(20, 200, 20)
y1 = sigmoid.RMSE
y2 = svr.RMSE
y3 = gb.RMSE
y4 = ab.RMSE
ax2.plot(X, y1, 'd--', markersize=7.5, label='Sigmoid')
# ax.plot(X, y1, '--', color='darkgreen', label='SVR')
ax2.plot(X, y2, '^--', color='darkred', markersize=7.5, label='SVR')
ax2.plot(X, y3, 's--', color='darkgreen', markersize=7.5, label='GB')
ax2.plot(X, y4, 'o--', color='orange', markersize=7.5, label='AB')

# ax2.legend(bbox_to_anchor=(0.9, 0.95))

# fig.suptitle("Model Performance vs Training Size", y=0.96, fontsize=14, fontweight='bold')
ax2.set_xlabel("Number of training data", fontproperties=font1)
ax2.set_ylabel("RMSE (min)", fontproperties=font1)



y1 = sigmoid.MedAE
y2 = svr.MedAE
y3 = gb.MedAE
y4 = ab.MedAE
ax3.plot(X, y1, 'd--', markersize=7.5, label='Sigmoid')
# ax.plot(X, y1, '--', color='darkgreen', label='SVR')
ax3.plot(X, y2, '^--', color='darkred', markersize=7.5, label='SVR')
ax3.plot(X, y3, 's--', color='darkgreen', markersize=7.5, label='GB')
ax3.plot(X, y4, 'o--', color='orange', markersize=7.5, label='AB')

# ax3.legend(bbox_to_anchor=(0.9, 0.95))

# fig.suptitle("Model Performance vs Training Size", y=0.96, fontsize=14, fontweight='bold')
ax3.set_xlabel("Number of training data", fontproperties=font1)
ax3.set_ylabel("MedAE (min)", fontproperties=font1)


y1 = sigmoid.R2
y2 = svr.R2
y3 = gb.R2
y4 = ab.R2
ax4.plot(X, y1, 'd--', markersize=7.5, label='Sigmoid')
# ax.plot(X, y1, '--', color='darkgreen', label='SVR')
ax4.plot(X, y2, '^--', color='darkred', markersize=7.5, label='SVR')
ax4.plot(X, y3, 's--', color='darkgreen', markersize=7.5, label='GB')
ax4.plot(X, y4, 'o--', color='orange', markersize=7.5, label='AB')

ax4.set_xlabel("Number of training data", fontproperties=font1)
ax4.set_ylabel("R-squared", fontproperties=font1)

ax1.legend(bbox_to_anchor=(3, 1.25),fontsize=14, ncol=4)

# lines, labels = ax4.get_legend_handles_labels()
# fig.legend(lines, labels, loc='upper center', ncol=4, fontsize=14)

# fig.suptitle("", y=0.96, fontproperties=font)

ax1.set_title('a)', loc='left', fontsize=16)
ax2.set_title('b)', loc='left', fontsize=16)
ax3.set_title('c)', loc='left', fontsize=16)
ax4.set_title('d)', loc='left', fontsize=16)


# fig.savefig("/Users/yanyan/Desktop/RT_Model_fig/COM_Learning_curve.png", dpi=300, bbox_inches='tight')


# fig.savefig("COM_Learning_curve.png", dpi=300, bbox_inches='tight')


# In[ ]:





# In[ ]:




