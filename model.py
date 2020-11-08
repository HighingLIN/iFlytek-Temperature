#%%
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from fancyimpute import KNN
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')


#%%
# **data import**
"""数据读入"""
train_df = pd.read_csv('./复赛数据_温室温度预测挑战赛_温室温度预测挑战赛复赛数据/训练集/train.csv')
test_df = pd.read_csv('./复赛数据_温室温度预测挑战赛_温室温度预测挑战赛复赛数据/测试集/test.csv')
sample_df = pd.read_csv('./复赛数据_温室温度预测挑战赛_温室温度预测挑战赛复赛数据/提交样例.csv')

sub = pd.DataFrame(test_df['时间戳'])

"""去掉无室内温度的值，再进行缺失值填充"""
train_df = train_df[train_df['temperature'].notnull()]
train_df=train_df.reset_index(drop=True)
train_df = train_df.fillna(method='bfill')
test_df = test_df.fillna(method='bfill')

"""修改列名"""
train_df.columns = ['time','year','month','day','hour','min','sec','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo', 'temperature']
test_df.columns = ['time','year','month','day','hour','min','sec','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']


#%%
"""清除Atmo异常值"""
train_df['atmodif']=abs(train_df['indoorAtmo']-train_df['outdoorAtmo'])

# 获取同一时刻室内室外atmo值相差大于等于5的索引
index=list(train_df[train_df['atmodif']>=5].index)
train_df.drop('atmodif',axis=1,inplace=True)

# 将下一时刻与当前时刻的atmo值大于等于5的置空
for i in index:
    a=i
    b=i+1
    if(abs(train_df.loc[a,'outdoorAtmo']-train_df.loc[b,'outdoorAtmo'])>=5):
        train_df.loc[a,'outdoorAtmo']=np.nan
    if(abs(train_df.loc[a,'indoorAtmo']-train_df.loc[b,'indoorAtmo'])>=5):
        train_df.loc[a,'indoorAtmo']=np.nan

# 后向填充
train_df = train_df.fillna(method='bfill')

# 减去百位数，标准化atmo
train_df['indoorAtmo']=np.round(train_df['indoorAtmo']-900)
train_df['outdoorAtmo']=np.round(train_df['outdoorAtmo']-900)

train_df.loc[train_df['indoorAtmo']<80,'indoorAtmo']=np.nan
train_df.loc[train_df['indoorAtmo']>100,'indoorAtmo']=np.nan
train_df.loc[train_df['outdoorAtmo']<80,'outdoorAtmo']=np.nan
train_df.loc[train_df['outdoorAtmo']>100,'outdoorAtmo']=np.nan
train_df = train_df.fillna(method='bfill')

# 获取前后时刻atmo值相差大于等于3的索引并置空
for i in range(len(train_df)-2):
    a=i+1
    b=a+1
    if abs(train_df.loc[a,'outdoorAtmo']-train_df.loc[i,'outdoorAtmo'])>=3:
        if abs(train_df.loc[b,'outdoorAtmo']-train_df.loc[a,'outdoorAtmo'])<3:
            train_df.loc[i,'outdoorAtmo']=np.nan
for i in range(len(train_df)-2):
    a=i+1
    b=a+1
    if abs(train_df.loc[a,'indoorAtmo']-train_df.loc[i,'indoorAtmo'])>=3:
        if abs(train_df.loc[b,'indoorAtmo']-train_df.loc[a,'indoorAtmo'])<3:
            train_df.loc[i,'indoorAtmo']=np.nan

# knn填充缺失值
imp_cols=['time','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']
imp_train=train_df.loc[:,imp_cols].copy()
imp_train_df = pd.DataFrame(KNN(k=12).fit_transform(imp_train), columns=imp_cols)

imp_cols=['outdoorAtmo','indoorAtmo']
train_df.loc[:,imp_cols]=imp_train_df.loc[:,imp_cols]


#%%
"""测试集与提交样例合并，填充日期"""
test_df = sample_df.merge(test_df,on=['time'],how='left')
test_df=test_df.drop(['temperature'],axis=1)

# 填充日期
test_df.loc[:,'year']=2020
test_df.loc[:,'month']=2
test_df.loc[:,'sec']=30
day=1
hour=0
minu=0
for i in range(test_df.shape[0]):
    if hour != 23 and minu == 59:
        test_df.loc[i,'day']=day
        test_df.loc[i,'hour']=hour
        test_df.loc[i,'min']=minu
        hour+=1
        minu=0
        continue
    if hour == 23 and minu == 59:
        test_df.loc[i,'day']=day
        test_df.loc[i,'hour']=hour
        test_df.loc[i,'min']=minu
        minu=0
        hour=0
        day+=1
        continue
    test_df.loc[i,'day']=day
    test_df.loc[i,'hour']=hour
    test_df.loc[i,'min']=minu
    minu+=1



#%%
"""knn法填充'outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo'缺失值"""
imp_cols=['time','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']
imp_test=test_df.loc[:,imp_cols].copy()
imp_test_df = pd.DataFrame(KNN(k=12).fit_transform(imp_test), columns=imp_cols)

imp_cols=['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']
test_df.loc[:,imp_cols]=imp_test_df.loc[:,imp_cols]


#%%
"""清除Atmo异常值"""
"""与train操作相同"""
test_df['atmodif']=abs(test_df['indoorAtmo']-test_df['outdoorAtmo'])

index=list(test_df[test_df['atmodif']>=5].index)
test_df.drop('atmodif',axis=1,inplace=True)

for i in index:
    a=i
    b=i+1
    if(abs(test_df.loc[a,'outdoorAtmo']-test_df.loc[b,'outdoorAtmo'])>=5):
        test_df.loc[a,'outdoorAtmo']=np.nan
    if(abs(test_df.loc[a,'indoorAtmo']-test_df.loc[b,'indoorAtmo'])>=5):
        test_df.loc[a,'indoorAtmo']=np.nan

test_df = test_df.fillna(method='bfill')

test_df['indoorAtmo']=np.round(test_df['indoorAtmo']-900)
test_df['outdoorAtmo']=np.round(test_df['outdoorAtmo']-900)

test_df.loc[test_df['indoorAtmo']<80,'indoorAtmo']=np.nan
test_df.loc[test_df['indoorAtmo']>100,'indoorAtmo']=np.nan
test_df.loc[test_df['outdoorAtmo']<80,'outdoorAtmo']=np.nan
test_df.loc[test_df['outdoorAtmo']>100,'outdoorAtmo']=np.nan
test_df = test_df.fillna(method='bfill')

for i in range(len(test_df)-2):
    a=i+1
    b=a+1
    if abs(test_df.loc[a,'outdoorAtmo']-test_df.loc[i,'outdoorAtmo'])>=3:
        if abs(test_df.loc[b,'outdoorAtmo']-test_df.loc[a,'outdoorAtmo'])<3:
            test_df.loc[i,'outdoorAtmo']=np.nan
for i in range(len(test_df)-2):
    a=i+1
    b=a+1
    if abs(test_df.loc[a,'indoorAtmo']-test_df.loc[i,'indoorAtmo'])>=3:
        if abs(test_df.loc[b,'indoorAtmo']-test_df.loc[a,'indoorAtmo'])<3:
            test_df.loc[i,'indoorAtmo']=np.nan

# 填充缺失值
imp_cols=['time','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']
imp_test=test_df.loc[:,imp_cols].copy()
imp_test_df = pd.DataFrame(KNN(k=12).fit_transform(imp_test), columns=imp_cols)

imp_cols=['outdoorAtmo','indoorAtmo']
test_df.loc[:,imp_cols]=imp_test_df.loc[:,imp_cols]


#%%
"""合并训练集与测试集"""
data_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

"对'month','day','hour'聚合求'outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo'的统计特征"
group_feats = []
for f in tqdm(['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']):
    data_df['MDH_{}_medi'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('median')
    data_df['MDH_{}_mean'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('mean')
    data_df['MDH_{}_max'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('max')
    data_df['MDH_{}_min'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('min')
    data_df['MDH_{}_std'.format(f)] = data_df.groupby(['month','day','hour'])[f].transform('std')
    
    group_feats.append('MDH_{}_medi'.format(f))
    group_feats.append('MDH_{}_mean'.format(f))



"""对'outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo''MDH_{}_medi', 'MDH_{}_mean'交叉相除获取交叉特征"""
for f1 in tqdm(['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']+group_feats):
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']+group_feats:
        if f1 != f2:
            colname = '{}_{}_ratio'.format(f1, f2)
            data_df[colname] = data_df[f1].values / data_df[f2].values

data_df = data_df.fillna(method='bfill')



"""分别对前几天提取统计特征"""
data_df['dt'] = data_df['day'].values + (data_df['month'].values - 1) * 31

for f in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo', 'temperature']:
    tmp_df = pd.DataFrame()

    for t in tqdm(range(14, 36)):
        tmp = data_df[data_df['dt']<t].groupby(['hour'])[f].agg(['mean']).reset_index()
        tmp.columns = ['hour','hit_{}_mean'.format(f)]
        tmp['dt'] = t
        tmp_df = tmp_df.append(tmp)
    
    data_df = data_df.merge(tmp_df, on=['dt','hour'], how='left')

data_df = data_df.fillna(method='bfill')



"""对'outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo'分20个桶，然后对分桶求统计特征"""
for f in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
    data_df[f+'_20_bin'] = pd.cut(data_df[f], 20, duplicates='drop').apply(lambda x:x.left).astype(int)

for f1 in tqdm(['outdoorTemp_20_bin','outdoorHum_20_bin','outdoorAtmo_20_bin','indoorHum_20_bin','indoorAtmo_20_bin']):
    for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
        data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
        data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
        data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
        data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')
    

"""构造数据集"""
drop_columns=["time","year","sec","temperature"]

train_count = train_df.shape[0]
train_df = data_df[:train_count].copy().reset_index(drop=True)
test_df = data_df[train_count:].copy().reset_index(drop=True)

features = train_df[:1].drop(drop_columns,axis=1).columns


# 筛选后的特征
select_columns=['outdoorTemp',
 'day',
 'hour',
 'outdoorAtmo',
 'indoorHum',
 'MDH_outdoorTemp_medi',
 'MDH_outdoorTemp_mean',
 'MDH_outdoorTemp_max',
 'MDH_outdoorTemp_min',
 'MDH_outdoorHum_medi',
 'MDH_outdoorHum_mean',
 'MDH_outdoorHum_max',
 'MDH_outdoorHum_min',
 'MDH_outdoorAtmo_medi',
 'MDH_outdoorAtmo_max',
 'MDH_outdoorAtmo_min',
 'MDH_indoorAtmo_max',
 'outdoorTemp_MDH_outdoorTemp_medi_ratio',
 'outdoorTemp_MDH_outdoorAtmo_mean_ratio',
 'outdoorTemp_MDH_indoorAtmo_medi_ratio',
 'outdoorTemp_MDH_indoorAtmo_mean_ratio',
 'outdoorHum_MDH_outdoorHum_medi_ratio',
 'outdoorHum_MDH_outdoorHum_mean_ratio',
 'outdoorHum_MDH_outdoorAtmo_medi_ratio',
 'outdoorHum_MDH_outdoorAtmo_mean_ratio',
 'outdoorHum_MDH_indoorHum_medi_ratio',
 'outdoorHum_MDH_indoorHum_mean_ratio',
 'outdoorHum_MDH_indoorAtmo_mean_ratio',
 'outdoorAtmo_outdoorHum_ratio',
 'outdoorAtmo_indoorAtmo_ratio',
 'outdoorAtmo_MDH_outdoorTemp_mean_ratio',
 'outdoorAtmo_MDH_outdoorHum_medi_ratio',
 'outdoorAtmo_MDH_outdoorAtmo_mean_ratio',
 'indoorHum_MDH_outdoorHum_mean_ratio',
 'indoorHum_MDH_outdoorAtmo_medi_ratio',
 'indoorHum_MDH_indoorHum_medi_ratio',
 'indoorAtmo_outdoorTemp_ratio',
 'indoorAtmo_indoorHum_ratio',
 'indoorAtmo_MDH_outdoorTemp_mean_ratio',
 'indoorAtmo_MDH_outdoorHum_mean_ratio',
 'indoorAtmo_MDH_outdoorAtmo_mean_ratio',
 'indoorAtmo_MDH_indoorAtmo_medi_ratio',
 'indoorAtmo_MDH_indoorAtmo_mean_ratio',
 'MDH_outdoorTemp_medi_MDH_outdoorTemp_mean_ratio',
 'MDH_outdoorTemp_medi_MDH_outdoorHum_medi_ratio',
 'MDH_outdoorTemp_medi_MDH_indoorHum_medi_ratio',
 'MDH_outdoorTemp_mean_outdoorTemp_ratio',
 'MDH_outdoorHum_medi_indoorAtmo_ratio',
 'MDH_outdoorHum_medi_MDH_outdoorHum_mean_ratio',
 'MDH_outdoorHum_medi_MDH_outdoorAtmo_medi_ratio',
 'MDH_outdoorHum_medi_MDH_indoorHum_medi_ratio',
 'MDH_outdoorHum_medi_MDH_indoorHum_mean_ratio',
 'MDH_outdoorHum_mean_MDH_outdoorTemp_medi_ratio',
 'MDH_outdoorHum_mean_MDH_indoorHum_medi_ratio',
 'MDH_outdoorAtmo_medi_outdoorTemp_ratio',
 'MDH_outdoorAtmo_medi_MDH_outdoorTemp_medi_ratio',
 'MDH_outdoorAtmo_medi_MDH_indoorAtmo_mean_ratio',
 'MDH_outdoorAtmo_mean_MDH_indoorHum_mean_ratio',
 'MDH_indoorHum_mean_indoorHum_ratio',
 'hit_outdoorTemp_mean',
 'hit_outdoorAtmo_mean',
 'hit_indoorAtmo_mean',
 'outdoorHum_20_bin_outdoorAtmo_medi',
 'outdoorAtmo_20_bin_outdoorHum_medi',
 'indoorHum_20_bin_outdoorAtmo_mean']
"""如果是非复赛数据测试则将下面两行取消注释，如是则忽视"""
# col_temp=pd.read_csv('select_columns.csv')
# select_columns=np.array(col_temp).reshape(-1,).tolist()

x_train = train_df[select_columns]
x_test = test_df[select_columns]

# 平滑y值
y_train = train_df['temperature'] - train_df['outdoorTemp']


#%%
"""lgb模型训练"""

#随机切割数据集
from sklearn.model_selection import train_test_split
trn_x, val_x, trn_y, val_y = train_test_split(x_train, y_train, random_state=1, test_size=0.2)

train_matrix = lgb.Dataset(trn_x, label=trn_y)
valid_matrix = lgb.Dataset(val_x, label=val_y)

params = {
    'boosting_type': 'gbdt',
    'objective': 'mse',
    'metric':'mae',
    'min_child_weight': 5,
    'num_leaves': 2**8,
    'feature_fraction': 1,
    'bagging_fraction': 1,
    'bagging_freq': 1,
    'learning_rate': 0.001,
    'nthread': -1,
    'seed': 2020}

model = lgb.train(params, train_matrix, 10000, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,early_stopping_rounds=1e3)
val_pred = model.predict(val_x, num_iteration=model.best_iteration).reshape(-1,)
lgb_test = model.predict(x_test, num_iteration=model.best_iteration).reshape(-1,)

print("%s_mse_score:%f" %('lgb', mean_squared_error(val_y, val_pred)))

def single_model(clf, train_x, train_y, test_x, clf_name, numr=10000): 

    from sklearn.model_selection import train_test_split
    trn_x, val_x, trn_y, val_y = train_test_split(train_x, train_y, random_state=1, test_size=0.2)
    
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
        val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1)
        test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit).reshape(-1)

    if clf_name == "cat":
        params = {'learning_rate': 0.001, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

        model = clf(iterations=10000, **params)
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=[], use_best_model=True, verbose=500)

        val_pred  = model.predict(val_x)
        test_pred = model.predict(test_x)
    
    print("%s_mse_score:%f" %(clf_name, mean_squared_error(val_y, val_pred)))
    
    return val_pred, test_pred

def xgb_model(x_train, y_train, x_valid):
    xgb_train, xgb_test = single_model(xgb, x_train, y_train, x_valid, "xgb", 1)
    return xgb_train, xgb_test

def cat_model(x_train, y_train, x_valid):
    cat_train, cat_test = single_model(CatBoostRegressor, x_train, y_train, x_valid, "cat", 1)
    return cat_train, cat_test

"""xgbm模型训练"""
xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)

"""cat模型训练"""
cat_train, cat_test = cat_model(x_train, y_train, x_test)

"""融合lgb、xgb、cat模型，加权平均"""
test_pred = (lgb_test  + xgb_test + cat_test) /3


#%%
"""构造提交样本"""
sample_df["temperature"] = test_pred + test_df['outdoorTemp'].values
sample_df.columns = ['time','temperature']
sub=sample_df.loc[:,['time','temperature']].copy()
sub.to_csv('sub.csv', index=False)