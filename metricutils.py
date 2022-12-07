import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import *


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
    