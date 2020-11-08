#%%
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fancyimpute import KNN
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')


#%%
"""数据读入"""
train_df = pd.read_csv('./复赛数据/train.csv')
test_df = pd.read_csv('./复赛数据/test.csv')
sample_df = pd.read_csv('./复赛数据/提交样例.csv')

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

index=list(train_df[train_df['atmodif']>=5].index)
train_df.drop('atmodif',axis=1,inplace=True)

for i in index:
    a=i
    b=i+1
    if(abs(train_df.loc[a,'outdoorAtmo']-train_df.loc[b,'outdoorAtmo'])>=5):
        train_df.loc[a,'outdoorAtmo']=np.nan
    if(abs(train_df.loc[a,'indoorAtmo']-train_df.loc[b,'indoorAtmo'])>=5):
        train_df.loc[a,'indoorAtmo']=np.nan

train_df = train_df.fillna(method='bfill')

train_df['indoorAtmo']=np.round(train_df['indoorAtmo']-900)
train_df['outdoorAtmo']=np.round(train_df['outdoorAtmo']-900)

train_df.loc[train_df['indoorAtmo']<80,'indoorAtmo']=np.nan
train_df.loc[train_df['indoorAtmo']>100,'indoorAtmo']=np.nan
train_df.loc[train_df['outdoorAtmo']<80,'outdoorAtmo']=np.nan
train_df.loc[train_df['outdoorAtmo']>100,'outdoorAtmo']=np.nan
train_df = train_df.fillna(method='bfill')


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

x_train = train_df[features]
y_train = train_df['temperature'] - train_df['outdoorTemp']


#%%
"""默认筛选特征"""
select_columns=['outdoorTemp']


#%%
"""获取初始分数"""
num=50

train_x=x_train.loc[:,select_columns]
train_y=y_train
clf=xgb

"""数据切割"""
from sklearn.model_selection import train_test_split
trn_x, val_x, trn_y, val_y = train_test_split(train_x, train_y, random_state=1, test_size=0.2)

train_matrix = clf.DMatrix(trn_x , label=trn_y, missing=np.nan)
valid_matrix = clf.DMatrix(val_x , label=val_y, missing=np.nan)

params = {'booster': 'gbtree',
            'eval_metric': 'mae',
            'min_child_weight': 5,
            'max_depth': 8,
            'subsample': 1,
            'colsample_bytree': 1,
            'eta': 0.001,
            'seed': 2020,
            'nthread': 36,
            # 'gpu_id':1,
            # 'tree_method':'gpu_hist',
            }

watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
model = clf.train(params, train_matrix, num_boost_round=num, evals=watchlist, verbose_eval=0, early_stopping_rounds=1000)
val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)
score=mean_squared_error(val_y, val_pred)
print("mse_score:%f" %(score))



#%%
"""xgb筛选"""

import colorama
fea={}
for i in features:
    if i in select_columns:
        print("已存在 下一个")
        continue
    # 添加新特征进select_columns进行筛选
    tem=select_columns+[i]
    train_x=x_train.loc[:,tem]
    train_y=y_train
    clf=xgb

    #数据切割
    from sklearn.model_selection import train_test_split
    trn_x, val_x, trn_y, val_y = train_test_split(train_x, train_y, random_state=1, test_size=0.2)

    train_matrix = clf.DMatrix(trn_x , label=trn_y, missing=np.nan)
    valid_matrix = clf.DMatrix(val_x , label=val_y, missing=np.nan)

    params = {'booster': 'gbtree',
                'eval_metric': 'mae',
                'min_child_weight': 5,
                'max_depth': 8,
                'subsample': 1,
                'colsample_bytree': 1,
                'eta': 0.001,
                'seed': 2020,
                'nthread': 36,
                }

    watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]

    model = clf.train(params, train_matrix, num_boost_round=num, evals=watchlist, verbose_eval=0, early_stopping_rounds=1000)
    val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)
    tem_score=mean_squared_error(val_y, val_pred)

    # 分数有降低则添加进select_columns，无则跳入下一个特征
    if(tem_score<score):
        select_columns.append(i)
        score=tem_score
        print(colorama.Fore.GREEN + "mse_score:%f     %s" %(score, i) + colorama.Fore.RESET)
    else:
        fea[i]=tem_score
        print("Fail！score:%f mse_score:%f      %s" %(tem_score,score,i))

print("Finish!")



#%%
"""结果输出"""
select_columns
#%%
a=pd.DataFrame(select_columns,columns=['select_columns'])
a.to_csv('select_columns.csv',index=False)





