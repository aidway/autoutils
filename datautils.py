import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn import preprocessing

def label_encode(data, cat_column):
    '''
    功能: 对类别型字段进行label encoder
    
    参数:
      - data:
      - cat_column:
      
    返回: 转换后的data
    '''
    lbl = preprocessing.LabelEncoder()
    lbl.fit(data[cat_column])
    data[cat_column] = lbl.transform(data[cat_column])
    return data
    
def down_sampling(data, samp_ratio=None, samp_cnt=None):
    '''
    功能: down sampling
    参数:
      - data: 输入数据
      - samp_ratio: 按比例下采样, [0, 1]
      - samp_cnt: 按数量下采样, [0, len(data)]
    注意: samp_ratio 优先被使用

    返回：采样后的数据
    '''
    data_len = len(data)
    data_idx = data.index
    random_cnt = 0
    
    if samp_ratio is not None:
        if samp_ratio < 0 or samp_ratio > 1:
            raise Exception('samp_ratio must in [0,1] .')
        random_cnt = int(data_len * samp_ratio)
    elif samp_cnt is not None:
        if samp_cnt < 0 or samp_cnt > data_len:
            raise Exception('samp_ratio must in [0,1] .')
        random_cnt = samp_cnt
    else:
        raise Exception('Must assign samp_ratio or samp_cnt.')
        
    random_data_index = np.random.choice(data_idx, size=random_cnt, replace=False)
    
    return data.loc[data.index.isin(random_data_index)]
    


# 提取身份证的性别
def f_sex(s):
    if len(s) < 18:
        # aae
        return 3
    else:
        if s[-1] == 'X' or s[-1] == 'x':
            return 0
        else:
            return int(s[-1]) % 2
        
def get_age(x):
    #获取年龄
    now = (datetime.datetime.now() + datetime.timedelta(days=1))
    year = now.year
    month = now.month
    day = now.day
    birth_year = int(x[6:10])
    birth_month = int(x[10:12])
    birth_day = int(x[12:14])
    if year == birth_year:
        return 0
    else:
        if birth_month > month or (birth_month == month and birth_day > day):
            return year - birth_year - 1
        else:
            return year - birth_year
        
        
# nan填充 用其他相同标签的非nan分布填充
def fill_nan(data, col):
    column = data[col]
    col_not_nan = column.dropna()
    col_nan = column[column.isna()]
    
    index = np.random.randint(0, col_not_nan.shape[0]+1, size=col_nan.shape[0])
    value = col_not_nan.iloc[index].values
    np.random.shuffle(value)
    nan_index = column[column.isna()].index
    
    for i, item in enumerate(value):
        data.loc[nan_index[i], [col]] = item
        
    return data

def print_metrics(y_true, y_pred, y_pred_score):
    print('roc_auc_score:%.4f' % roc_auc_score(y_true, y_pred_score[:,1]))
    print('f1_score:%.4f' %f1_score(y_true, y_pred))
    print('recall_score: %.4f' % recall_score(y_true, y_pred))
    print('precision_score:%.4f' % precision_score(y_true, y_pred))
    print('accuracy_score: %.4f' % accuracy_score(y_true, y_pred))
    print('confusion_matrix:\n' , confusion_matrix(y_true, y_pred))
    
    
def check_missing(data):
    '''
    check missing data
    '''
    missing_df = data.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['columns_name', 'missing_count']
    missing_df['missing_ratio'] = np.round(missing_df['missing_count']/data.shape[0] * 100, 2)
    miss_df1 = missing_df.sort_values('missing_ratio', ascending =False).reset_index(drop =True)
    return miss_df1
    