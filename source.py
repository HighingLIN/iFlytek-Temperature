#%%
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
# from fancyimpute import KNN
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')



#%%
train_df = pd.read_csv('./train/train.csv')
test_df = pd.read_csv('./test/test.csv')
sub = pd.DataFrame(test_df['time'])


## %%
# 填充缺失值
train_df = train_df[train_df['temperature'].notnull()]
train_df=train_df.reset_index(drop=True)
train_df = train_df.fillna(method='bfill')
test_df = test_df.fillna(method='bfill')


##%%
# 缺失值分析
def missing_values(df):
    alldata_na = pd.DataFrame(df.isnull().sum(), columns={'missingNum'})
    alldata_na['existNum'] = len(df) - alldata_na['missingNum']
    alldata_na['sum'] = len(df)
    alldata_na['missingRatio'] = alldata_na['missingNum']/len(df)*100
    alldata_na['dtype'] = df.dtypes
    #ascending：默认True升序排列；False降序排列
    alldata_na = alldata_na[alldata_na['missingNum']>0].reset_index().sort_values(by=['missingNum','index'],ascending=[False,True])
    alldata_na.set_index('index',inplace=True)
    return alldata_na


##%%
# 修改列名
train_df.columns = ['time','year','month','day','hour','min','sec','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo', 'temperature']
test_df.columns = ['time','year','month','day','hour','min','sec','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']

data_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

#%%
    # data_df['subHum']=data_df['indoorHum']-data_df['outdoorHum']
    # data_df['subAtmo']=data_df['indoorAtmo']-data_df['outdoorAtmo']
    # data_df['outdoorHumAtmo']=data_df['outdoorHum']+data_df['outdoorAtmo']
    # data_df['indoorHumAtmo']=data_df['indoorHum']+data_df['indoorAtmo']
#%%
# 基本聚合特征
def mode(col):
    a=col.mode()
    return a[0]

def q75(x):
    return x.quantile(0.75)

def q25(x):
    return x.quantile(0.25)

group_feats = []
for f in tqdm(['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']+['outdoorHumAtmo','indoorHumAtmo','subHum','subAtmo']):
    data_df['MDH_{}_medi'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('median')
    data_df['MDH_{}_mean'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('mean')
    data_df['MDH_{}_max'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('max')
    data_df['MDH_{}_min'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('min')
    data_df['MDH_{}_std'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('std')
    data_df['MDH_{}_skew'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('skew')
    data_df['MDH_{}_mode'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform(mode)
    data_df['MDH_{}_q25'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform(q25)
    data_df['MDH_{}_q75'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform(q75)
    

    group_feats.append('MDH_{}_medi'.format(f))
    group_feats.append('MDH_{}_mean'.format(f))

#%%
# 基本交叉特征
for f1 in tqdm(['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']+group_feats+['outdoorHumAtmo','indoorHumAtmo','subHum','subAtmo']):
    
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']+group_feats+['outdoorHumAtmo','indoorHumAtmo','subHum','subAtmo']:
        if f1 != f2:
            colname = '{}_{}_ratio'.format(f1, f2)
            data_df[colname] = data_df[f1].values / data_df[f2].values

data_df = data_df.fillna(method='bfill')


#%%
# 历史信息提取
data_df['dt'] = data_df['day'].values + (data_df['month'].values - 3) * 31
for f in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo', 'temperature']+['outdoorHumAtmo','indoorHumAtmo','subHum','subAtmo']:
    tmp_df = pd.DataFrame()
    for t in tqdm(range(15, 45)):
        tmp = data_df[data_df['dt']<t].groupby(['hour'])[f].agg(['mean']).reset_index()
        tmp.columns = ['hour','hit_{}_mean'.format(f)]
        tmp['dt'] = t
        tmp_df = tmp_df.append(tmp)
    
    data_df = data_df.merge(tmp_df, on=['dt','hour'], how='left')

data_df = data_df.fillna(method='bfill')
# df_temp=data_df.copy()
## %%
# 重新赋值
# data_df=df_temp.copy()



#%%
# 离散化
for f in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
    data_df[f+'_20_bin'] = pd.cut(data_df[f], 20, duplicates='drop').apply(lambda x:x.left).astype(int)
#%%
for f1 in tqdm(['outdoorTemp_20_bin','outdoorHum_20_bin','outdoorAtmo_20_bin','indoorHum_20_bin','indoorAtmo_20_bin']):
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
        data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')
        # data_df['{}_{}_skew'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('skew')
        # data_df['{}_{}_std'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('std')
        # data_df['{}_{}_mode'.format(f1,f2)] = data_df.groupby([f1])[f2].transform(mode)
        # data_df['{}_{}_q25'.format(f1,f2)] = data_df.groupby([f1])[f2].transform(q25)
        # data_df['{}_{}_q75'.format(f1,f2)] = data_df.groupby([f1])[f2].transform(q75)
        



#%%
#构造训练数据集
drop_columns=["time","year","sec","temperature"]

train_count = train_df.shape[0]
train_df = data_df[:train_count].copy().reset_index(drop=True)
test_df = data_df[train_count:].copy().reset_index(drop=True)

features = train_df[:1].drop(drop_columns,axis=1).columns

#%%
x_train = train_df[features]
x_test = test_df[features]

y_train = train_df['temperature'].values - train_df['outdoorTemp'].values


#%%
#单模xgb训练
train_x=x_train
# train_x=x_train.loc[:,select_columns]
train_y=y_train
test_x=x_test
# test_x=x_te.loc[:,select_columns]
clf=xgb

print(train_x.shape)

# nums=int(train_x.shape[0]*0.8)
# trn_x, trn_y, val_x, val_y = train_x[:nums], train_y[:nums], train_x[nums:], train_y[nums:]

#数据切割
from sklearn.model_selection import train_test_split
trn_x, val_x, trn_y, val_y = train_test_split(train_x, train_y, random_state=1, test_size=0.2)

train_matrix = clf.DMatrix(trn_x , label=trn_y, missing=np.nan)
valid_matrix = clf.DMatrix(val_x , label=val_y, missing=np.nan)
test_matrix = clf.DMatrix(test_x , missing=np.nan)
params = {  'booster': 'gbtree',
            'eval_metric': 'mae',
            'min_child_weight': 5,
            'max_depth': 8,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'eta': 0.001,
            'seed': 2020,
            'nthread': 36,
            }

watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]

model = clf.train(params, train_matrix, num_boost_round=10000, evals=watchlist, verbose_eval=500, early_stopping_rounds=1000)
val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)
test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit).reshape(-1,1)
xgb_test=test_pred

print("mse_score:%f" %mean_squared_error(val_y, val_pred))

#%%
sub["temperature"] = xgb_test[:,0] + test_df['outdoorTemp'].values
#%%
sub.to_csv('pre24.csv', index=False)
#%%
test_pred = (lgb_test[:,0] + xgb_test[:,0]) /2
#%%
sub["temperature"] = test_pred + test_df['outdoorTemp'].values



#%%
# GMM聚类
from sklearn.mixture import GaussianMixture

for f1 in ['outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
    col=['outdoorTemp',f1]
    gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=1)
    data_df['{}_clu'.format(f1)]= pd.DataFrame(gmm.fit_predict(data_df[col]))

for f1 in tqdm(['outdoorHum_clu', 'outdoorAtmo_clu', 'indoorHum_clu','indoorAtmo_clu']):
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
        data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')



#%%
def single_model(clf, train_x, train_y, test_x, clf_name, numr=10000): 
    # nums = int(train_x.shape[0] * 0.80)
    
    if clf_name in ['sgd','ridge']:
        print('MinMaxScaler...')
        for col in features:
            ss = MinMaxScaler()
            ss.fit(np.vstack([train_x[[col]].values, test_x[[col]].values]))
            train_x[col] = ss.transform(train_x[[col]].values).flatten()
            test_x[col] = ss.transform(test_x[[col]].values).flatten()
    
    # trn_x, trn_y, val_x, val_y = train_x[:nums], train_y[:nums], train_x[nums:], train_y[nums:]
    from sklearn.model_selection import train_test_split
    trn_x, val_x, trn_y, val_y = train_test_split(train_x, train_y, random_state=1, test_size=0.2)
    
    if clf_name == "lgb":
        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)
        data_matrix  = clf.Dataset(train_x, label=train_y)
        
        params = {
            'boosting_type': 'gbdt',
            'objective': 'mse',
            'metric':'mae',
            'min_child_weight': 5,
            'num_leaves': 2**8,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 1,
            'learning_rate': 0.001,
            'min_split_gain':0.05,
            # 'lambda_l1':0.1,
            # 'lambda_l2':0.1,
            'min_data_in_leaf':60,
            'nthread': -1,
            'seed': 2020}

        model = clf.train(params, train_matrix, numr, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,early_stopping_rounds=1000)
        # model2 = clf.train(params, data_matrix, model.best_iteration)
        # print(model.best_iteration,model2.best_iteration)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration).reshape(-1,1)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration).reshape(-1,1)

    if clf_name == "xgb":
        train_matrix = clf.DMatrix(trn_x , label=trn_y, missing=np.nan)
        valid_matrix = clf.DMatrix(val_x , label=val_y, missing=np.nan)
        test_matrix = clf.DMatrix(test_x , missing=np.nan)
        params = {'booster': 'gbtree',
                  'eval_metric': 'mae',
                  'min_child_weight': 5,
                  'max_depth': 8,
                  'subsample': 0.5,
                  'colsample_bytree': 0.5,
                  'eta': 0.001,
                  'seed': 2020,
                  'nthread': 36,
                #   'silent': True,
                  }

        watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]

        model = clf.train(params, train_matrix, num_boost_round=10000, evals=watchlist, verbose_eval=500, early_stopping_rounds=1000)
        val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)
        test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit).reshape(-1,1)

    if clf_name == "cat":
        params = {'learning_rate': 0.001, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

        model = clf(iterations=15000, **params)
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=[], use_best_model=True, verbose=500)

        val_pred  = model.predict(val_x)
        test_pred = model.predict(test_x)
    
    if clf_name == "sgd":
        params = {
            'loss': 'squared_loss',
            'penalty': 'l2',
            'alpha': 0.00001,
            'random_state': 2020,
        }
        model = SGDRegressor(**params)
        model.fit(trn_x, trn_y)
        val_pred  = model.predict(val_x)
        test_pred = model.predict(test_x)
    
    if clf_name == "ridge":
        params = {
                'alpha': 1.0,
                'random_state': 2020,
            }
        model = Ridge(**params)
        model.fit(trn_x, trn_y)
        val_pred  = model.predict(val_x)
        test_pred = model.predict(test_x)

    
    print("%s_mse_score:%f" %(clf_name, mean_squared_error(val_y, val_pred)))
    
    return val_pred, test_pred

def lgb_model(x_train, y_train, x_valid, numr):
    lgb_train, lgb_test = single_model(lgb, x_train, y_train, x_valid, "lgb", numr)
    return lgb_train, lgb_test

def xgb_model(x_train, y_train, x_valid):
    xgb_train, xgb_test = single_model(xgb, x_train, y_train, x_valid, "xgb", 1)
    return xgb_train, xgb_test

def cat_model(x_train, y_train, x_valid):
    cat_train, cat_test = single_model(CatBoostRegressor, x_train, y_train, x_valid, "cat", 1)
    return cat_train, cat_test

def sgd_model(x_train, y_train, x_valid):
    sgd_train, sgd_test = single_model(SGDRegressor, x_train, y_train, x_valid, "sgd", 1)
    return sgd_train, sgd_test

def ridge_model(x_train, y_train, x_valid):
    ridge_train, ridge_test = single_model(Ridge, x_train, y_train, x_valid, "ridge", 1)
    return ridge_train, ridge_test


#%%
lr_train, lr_test = ridge_model(x_train, y_train, x_test)
#%%
sgd_train, sgd_test = sgd_model(x_train, y_train, x_test)
#%%
lgb_train, lgb_test = lgb_model(x_train, y_train, x_test, 10000)
#%%
xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)
#%%
cat_train, cat_test = cat_model(x_train, y_train, x_test)


#%%
train_pred = (lr_train + sgd_train + lgb_train[:,0] + xgb_train[:,0] + cat_train) / 5
#%%
test_pred = (lr_test + sgd_test + lgb_test[:,0] + xgb_test[:,0] + cat_test) /5
#%%
sub["temperature"] = test_pred + test_df['outdoorTemp'].values


#%%
sub["temperature"] = lr_test + test_df['outdoorTemp'].values
#%%
sub.to_csv('lr1.csv', index=False)
#%%
sub["temperature"] = sgd_test + test_df['outdoorTemp'].values
#%%
sub.to_csv('sgd1.csv', index=False)
#%%
sub["temperature"] = lgb_test[:,0] + test_df['outdoorTemp'].values
#%%
sub.to_csv('lgb4.csv', index=False)
#%%
sub["temperature"] = cat_test + test_df['outdoorTemp'].values
#%%
sub.to_csv('cat1.csv', index=False)
#%%
sub["temperature"] = xgb_test[:,0] + test_df['outdoorTemp'].values
#%%
sub.to_csv('xgblgb1.csv', index=False)



#%%
#GRU神经网络训练
import tensorflow as tf
from tensorflow import keras
import random as rn

tf.random.set_seed(1)
np.random.seed(1)
rn.seed(1)

class Printlogs(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            print(epoch + 1, "\t", logs)

batch_size = 64
epochs = 75
act = 'relu'
opt = keras.optimizers.Nadam()
units = 64
init_weights = 'he_normal'

train_x=x_train
train_y=y_train
test_x=x_test

for col in features:
    ss = MinMaxScaler()
    ss.fit(np.vstack([train_x[[col]].values, test_x[[col]].values]))
    train_x[col] = ss.transform(train_x[[col]].values).flatten()
    test_x[col] = ss.transform(test_x[[col]].values).flatten()

# nums = int(train_x.shape[0] * 0.80)
# trn_x, trn_y, val_x, val_y = train_x[:nums], train_y[:nums], train_x[nums:], train_y[nums:]
from sklearn.model_selection import train_test_split
trn_x, val_x, trn_y, val_y = train_test_split(train_x, train_y, random_state=1, test_size=0.2)

# trn_x=np.array(trn_x,dtype=np.float32)
# trn_x=trn_x.reshape(trn_x.shape[0],1,trn_x.shape[1])
# val_x=np.array(val_x,dtype=np.float32)
# val_x=val_x.reshape(val_x.shape[0],1,val_x.shape[1])
# test_x=np.array(test_x,dtype=np.float32)
# test_x=test_x.reshape(test_x.shape[0],1,test_x.shape[1])

model = keras.models.Sequential()
# model.add(keras.layers.GRU(128,input_shape=(1,trn_x.shape[2])),)
model.add(keras.layers.Dense(units, kernel_initializer=init_weights,input_shape=(None,trn_x.shape[1])))
model.add(keras.layers.Activation(act))
# model.add(keras.layers.GaussianNoise(1))
model.add(keras.layers.Dense(units, kernel_initializer=init_weights,))
model.add(keras.layers.Activation(act))
model.add(keras.layers.Dense(1,kernel_initializer=init_weights,))

model.compile(loss='mse',optimizer=opt,metrics=['mae'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001, mode='min')

model.fit(trn_x, trn_y, epochs=epochs, batch_size=batch_size, shuffle=False,
                validation_data=(val_x, val_y), verbose=0,
                callbacks=[Printlogs(), early_stop])

gru_test = model.predict(test_x).flatten()

#%%
sub["temperature"] = gru_test + test_df['outdoorTemp'].values
#%%
sub.to_csv('gru1.csv', index=False)


# %%
# lgb模型参数调优
from __future__ import print_function
import lightgbm as lgb
import xgboost as xgb
import sklearn
import numpy
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np
from sklearn.metrics import mean_squared_error
import colorama

N_HYPEROPT_PROBES = 50
HYPEROPT_ALGO = tpe.suggest                 #  tpe.suggest OR hyperopt.rand.suggest

def get_lgb_params(space):
    lgb_params = dict()
    lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'rmse'
    lgb_params['learning_rate'] = space['learning_rate']
    lgb_params['num_leaves'] = int(space['num_leaves'])
    lgb_params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    # lgb_params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    lgb_params['max_depth'] = -1
    lgb_params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
    lgb_params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
    lgb_params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 255
    lgb_params['feature_fraction'] = space['feature_fraction']
    lgb_params['bagging_fraction'] = space['bagging_fraction']
    lgb_params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1
    lgb_params['nthread'] = -1
    return lgb_params

# ---------------------------------------------------------------------

obj_call_count = 0
cur_best_score = 0 # 0 or np.inf
best_param=None
log_writer = open( './lgb-hyperopt-log.txt', 'a+' )

def objective(space):
    global obj_call_count, cur_best_score,best_param

    obj_call_count += 1
    score=0

    print('\nLightGBM objective call #{} cur_best_score={:7.5f} best_param={}'.format(obj_call_count,cur_best_score,best_param) )

    lgb_params = get_lgb_params(space)

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('\nParams: {}'.format(params_str))

    train2=x_train
    y=y_train
    kf = KFold(n_splits=5, shuffle=True,random_state=0)
    out_of_fold = np.zeros(len(train2))
    for fold, (train_idx, val_idx) in enumerate(kf.split(train2)):
        D_train = lgb.Dataset(train2.iloc[train_idx], label=y[train_idx])
        D_val = lgb.Dataset(train2.iloc[val_idx], label=y[val_idx])
        # Train
        num_round = 10000
        clf = lgb.train(lgb_params,
                           D_train,
                           num_boost_round=num_round,
                           valid_sets=D_val,
                           early_stopping_rounds=500,
                           verbose_eval=False,
                           )
        # predict
        nb_trees = clf.best_iteration
        out_of_fold[val_idx] = clf.predict(train2.iloc[val_idx], num_iteration=nb_trees)
        print('nb_trees={} score={}'.format(nb_trees, mean_squared_error(y[val_idx], out_of_fold[val_idx])))
        score += mean_squared_error(y[val_idx], out_of_fold[val_idx])
    
    score/=5
    
    print('val_r2_score={} {}'.format(score,mean_squared_error(y,out_of_fold)))

    log_writer.write('score={} Params:{} nb_trees={}\n'.format(score, params_str, nb_trees ))
    log_writer.flush()
    
    if obj_call_count==1:
        cur_best_score=score

    if score<=cur_best_score:
        best_param=lgb_params
        cur_best_score = score
        print(colorama.Fore.GREEN + 'NEW BEST SCORE={}'.format(cur_best_score) + colorama.Fore.RESET)
    return {'loss': score, 'status': STATUS_OK}

# --------------------------------------------------------------------------------

space ={
        'num_leaves': hp.quniform ('num_leaves', 10, 100, 5),
        'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 30, 2),
        'feature_fraction': hp.uniform('feature_fraction', 0.7, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.7, 1.0),
        'learning_rate': hp.uniform('learning_rate', 0.01, 1),
        # 'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'max_bin': hp.quniform ('max_bin', 100,280 , 10),
        'bagging_freq': hp.quniform ('bagging_freq', 1, 10, 2),
        # 'lambda_l1': hp.uniform('lambda_l1', 0, 10 ),
        # 'lambda_l2': hp.uniform('lambda_l2', 0, 10 ),
       }

trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

print('-'*50)
print('The best params:')
print( best )
print('\n\n')
log_writer.write('The best params:{}\n'.format(best))
log_writer.flush()


#%%
#单模lgb-k折训练，数据集打乱
lgb_params={'boosting_type': 'gbdt', 'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.47219279108693424, 'num_leaves': 80, 'min_data_in_leaf': 12, 'max_depth': -1, 'lambda_l1': 0.0, 'lambda_l2': 0.0, 'max_bin': 280, 'feature_fraction': 0.7735836890196606, 'bagging_fraction': 0.7415373779585942, 'bagging_freq': 8, 'nthread': -1}

train2=x_train
y=y_train
test2=x_test
kf = KFold(n_splits=5, shuffle=True,random_state=0)
out_of_fold = np.zeros(len(train2))
test_pred = np.zeros(len(test2))
score=0

for fold, (train_idx, val_idx) in enumerate(kf.split(train2)):
    D_train = lgb.Dataset(train2.iloc[train_idx], label=y[train_idx])
    D_val = lgb.Dataset(train2.iloc[val_idx], label=y[val_idx])
    # Train
    num_round = 10000
    clf = lgb.train(lgb_params,
                        D_train,
                        num_boost_round=num_round,
                        valid_sets=D_val,
                        early_stopping_rounds=200,
                        verbose_eval=False,
                        )
    # predict
    nb_trees = clf.best_iteration
    out_of_fold[val_idx] = clf.predict(train2.iloc[val_idx], num_iteration=nb_trees)
    print('nb_trees={} score={}'.format(nb_trees, mean_squared_error(y[val_idx], out_of_fold[val_idx])))
    score += mean_squared_error(y[val_idx], out_of_fold[val_idx])
    test_pred += clf.predict(test2, num_iteration=clf.best_iteration)/kf.n_splits

score/=5
print(score)



#%%
# 单模xgb-k折训练
params = {'booster': 'gbtree',
            'eval_metric': 'mae',
            'min_child_weight': 5,
            'max_depth': 8,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'eta': 0.001,
            'seed': 2020,
            'nthread': 36,
        #   'silent': True,
            }
shuffle=True
kf = KFold(n_splits=5, shuffle=shuffle, random_state=0)
train_x=x_train
train_y=y_train
test_x=x_test
class_num=1
clf=xgb
train = np.zeros((train_x.shape[0]))
test = np.zeros((test_x.shape[0]))
score=0

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x)):
    train_matrix = clf.DMatrix(train_x.iloc[train_idx] , label=train_y[train_idx], missing=np.nan)
    valid_matrix = clf.DMatrix(train_x.iloc[val_idx] , label=train_y[val_idx], missing=np.nan)
    test_matrix = clf.DMatrix(test_x , missing=np.nan)
    # Train

    # predict
    watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]

    model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=500, early_stopping_rounds=500)
    train[val_idx] = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
    print('nb_trees={} score={}'.format(model.best_ntree_limit, mean_squared_error(train_y[val_idx], train[val_idx])))
    score += mean_squared_error(train_y[val_idx], train[val_idx])
    test += model.predict(test_matrix , ntree_limit=model.best_ntree_limit)/kf.n_splits

score/=5
xgb_test=test_pred
print("mse_score:%f" %mean_squared_error(train_y, train))


#%%
#单模xgb训练
train_x=x_train
# train_x=x_train.iloc[:,select_columns]
# train_x=x_train.loc[:,select_columns]
train_y=y_train
test_x=x_test
# test_x=x_test.iloc[:,select_columns]
# test_x=x_test.loc[:,select_columns]
clf=xgb

print(train_x.shape)

# nums = int(train_x.shape[0] * 0.80)
# trn_x, trn_y, val_x, val_y = train_x[:nums], train_y[:nums], train_x[nums:], train_y[nums:]

#数据切割
from sklearn.model_selection import train_test_split
trn_x, val_x, trn_y, val_y = train_test_split(train_x, train_y, random_state=1, test_size=0.2)

train_matrix = clf.DMatrix(trn_x , label=trn_y, missing=np.nan)
valid_matrix = clf.DMatrix(val_x , label=val_y, missing=np.nan)
test_matrix = clf.DMatrix(test_x , missing=np.nan)
params = {'booster': 'gbtree',
            'eval_metric': 'mae',
            'min_child_weight': 5,
            'max_depth': 8,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'eta': 0.001,
            'seed': 2020,
            'nthread': 36,
        #   'silent': True,
            }

watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]

model = clf.train(params, train_matrix, num_boost_round=10000, evals=watchlist, verbose_eval=500, early_stopping_rounds=1000)
val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)
test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit).reshape(-1,1)
xgb_test=test_pred

print("mse_score:%f" %mean_squared_error(val_y, val_pred))



#%%
# KNN填充值
# col=['time', '年', '月', '日', '小时', '分钟', '秒', '温度(室外)', '湿度(室外)', '气压(室外)','湿度(室内)', '气压(室内)']
# test_df.columns=col
# train_df['label']=0
# test_df['label']=1
# imp_cols=['温度(室外)', '湿度(室外)', '气压(室外)','湿度(室内)', '气压(室内)','label']
# train_df_all=pd.concat([train_df.loc[:,imp_cols],test_df.loc[:,imp_cols]],axis=0).reset_index(drop=True)
# train_df1 = pd.DataFrame(KNN(k=6).fit_transform(train_df_all), columns=imp_cols)
# #%%
# imp_cols=['温度(室外)', '湿度(室外)', '气压(室外)','湿度(室内)', '气压(室内)']
# train_df.loc[:,imp_cols]=train_df1.loc[train_df1['label']==0.,imp_cols]
# test=train_df1.loc[train_df1['label']==1.,imp_cols].copy().reset_index(drop=True)
# test_df.loc[:,imp_cols]=test
# train_df.drop(['label'],axis=1,inplace=True)
# test_df.drop(['label'],axis=1,inplace=True)
# #%%
# test=train_df1.loc[train_df1['label']==1.,imp_cols].copy().reset_index(drop=True)
# test_df.loc[:,imp_cols]=test


#%%
# 特征筛选
# %%
from xgboost import XGBRegressor

model=XGBRegressor(max_depth=8 , learning_rate=0.001, n_estimators=10000,
                        subsample=0.5,booster='gbtree',
                        colsample_bytree=0.5, min_child_samples=5,eval_metric = 'mae')
model.fit(trn_x,trn_y,
            eval_set=[(trn_x, trn_y),(val_x, val_y)],
            early_stopping_rounds=1000, verbose=500)

col=list(x_train.columns)
fea_imp=abs(model.feature_importances_)
index=np.argsort(-fea_imp)
fea=dict()
for i in range(137):
    j=index[i]
    fea[col[j]]=fea_imp[j]
#%%
# from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
print(x_train.shape)

# sk=SelectKBest(f_regression,k=280)
# new_train=sk.fit_transform(x_train,y_train)
# 获取对应列索引
# select_columns=sk.get_support(indices = True)

ridge = Ridge()
rfe = RFE(ridge, n_features_to_select=150)
rfe.fit(x_train,y_train)

select_columns = [f for f, s in zip(x_train.columns, rfe.support_) if s]
# new_train = x_train[select_columns]

# print(new_train.shape)

print(len(select_columns))


#%%
cut=10
# 离散化
for f in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
    data_df[f+'_10_bin'] = pd.cut(data_df[f], cut, duplicates='drop').apply(lambda x:x.left).astype(int)
    # data_df[f+'_20_bin'] = pd.cut(data_df[f], 20, duplicates='drop').apply(lambda x:x.left).astype(int)
    # data_df[f+'_50_bin'] = pd.cut(data_df[f], 50, duplicates='drop').apply(lambda x:x.left).astype(int)
    # data_df[f+'_100_bin'] = pd.cut(data_df[f], 100, duplicates='drop').apply(lambda x:x.left).astype(int)
    # data_df[f+'_200_bin'] = pd.cut(data_df[f], 200, duplicates='drop').apply(lambda x:x.left).astype(int)
#%%
#序列化数据
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def tfidf(input_values, output_num, output_prefix, f1, seed=1024):
    tfidf_enc = TfidfVectorizer()
    tfidf_vec = tfidf_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(tfidf_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_{}_tfidf_{}'.format(f1,output_prefix, i) for i in range(output_num)]
    return svd_tmp

def w2v_feat(df, group_id, feat, length):
    print('start word2vec ...')
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    model = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1, hs=1, workers=1, iter=10, seed=1)
    data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([model[c] for c in x]))
    for m in range(length):
        data_frame['w2v_{}_{}_{}_mean'.format(group_id,feat,m)] = data_frame[feat].apply(lambda x: x[m].mean())
    del data_frame[feat]
    return data_frame
#%%
# num=20
# data_df2=data_df.copy()
# data_df2['outdoorTemp']=data_df2['outdoorTemp'].astype(str)
# data_df2['outdoorHum']=data_df2['outdoorHum'].astype(str)
# data_df2['outdoorAtmo']=data_df2['outdoorAtmo'].astype(str)
# data_df2['indoorHum']=data_df2['indoorHum'].astype(str) 
# data_df2['indoorAtmo']=data_df2['indoorAtmo'].astype(str)
for f1 in tqdm(['outdoorTemp_20_bin','outdoorHum_20_bin','outdoorAtmo_20_bin','indoorHum_20_bin','indoorAtmo_20_bin']):
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
        data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')
        # if f2 == 'outdoorTemp':
        # group_target='outdoorTemp'
        # group_target=f2
        # w2v_df = w2v_feat(data_df2, f1, group_target, num)
        # data_df = data_df.merge(w2v_df, on=f1, how='left')
            # tmp = data_df2.groupby(f1)[f2].agg(list).reset_index()
            # tmp[f2] = tmp[f2].apply(lambda x: ' '.join(x))
            # tfidf_tmp = tfidf(tmp[f2], num, f2, f1)
            # group_df=pd.concat([tmp[[f1]], tfidf_tmp], axis=1)
            # data_df = data_df.merge(group_df, on=f1, how='left')
#%%  
for f1 in tqdm(['outdoorTemp_50_bin','outdoorHum_50_bin','outdoorAtmo_50_bin','indoorHum_50_bin','indoorAtmo_50_bin']):
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
        data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')
#%%    
for f1 in tqdm(['outdoorTemp_100_bin','outdoorHum_100_bin','outdoorAtmo_100_bin','indoorHum_100_bin','indoorAtmo_100_bin']):
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
        data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')
#%%  
for f1 in tqdm(['outdoorTemp_200_bin','outdoorHum_200_bin','outdoorAtmo_200_bin','indoorHum_200_bin','indoorAtmo_200_bin']):
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
        data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')


#%%
# 众数
def mode(col):
    a=col.mode()
    return a[0]

def q75(x):
    return x.quantile(0.75)

def q25(x):
    return x.quantile(0.25)

for f1 in tqdm(['outdoorTemp_10_bin','outdoorHum_10_bin','outdoorAtmo_10_bin','indoorHum_10_bin','indoorAtmo_10_bin']):
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
        data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')