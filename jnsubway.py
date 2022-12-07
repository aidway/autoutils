import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates

from imp import reload
from datetime import datetime


import sys
import os
import warnings 
from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

import statsmodels.tsa.api as smt
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import scipy.stats as scs                             
from scipy.optimize import minimize              # for function minimization

from dateutil.relativedelta import relativedelta # working with dates with style

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

sys.path.append(".")
from autoutils import dbutils
reload(dbutils)



pred2db_mode = 'hdfs' # 预测结果入库模式：hdfs、jdbc。
debug_mode = True  # 是否debug模式
n_steps = 21   # 预测天数
adf_pvalue_threshold = 0.0000000001   # ADF平稳性阈值，小于则平稳
min_data_threshold = 90  # 训练数据最少数目，小于则不进行预测
multi_process_num = 4  #  多进程数目
sarima_21d_pred_hdfs_path='/tmp/jnsubway/ai_pred_travel_21d_sarima'
arimax_21d_pred_hdfs_path='/tmp/jnsubway/ai_pred_travel_21d_arimax'


FILE_BASE_PATH=__file__

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PlotConfig:
    font_path=os.path.join(BASE_DIR, 'fonts', 'simhei.ttf')  
    y1_c='#EE7E2D'
    y2_c='#4273C5'
    fig_x=25
    fig_y=8
    lw=2
    tick_len=30
    tick_size=25
    label_size=25
    legend_size=20
    title_size=25
    colors = ['#EE7E2D', '#4273C5', '#FFC100','#C10000', '#9F59A6', '#00FFFF', '#B0ADAD']
    date_type = "%Y-%m-%d"


def get_cal_dt(default_cal_dt=None):
    '''
    功能：返回数据日期。
    default_cal_dt：默认数据日期，如果未指定，则返回sysdate，否则返回指定的数据日期
    '''
    if default_cal_dt is not None:
        return default_cal_dt
    
    cal_dt = dbutils.execSqlPyhive('select tdh_todate(sysdate) cal_dt from system.dual')
    cal_dt = cal_dt.values[0][0]
    return cal_dt
    

def get_getin_data(cal_dt, condition=""):
    '''
    功能：获取进站量数据
    cal_dt：数据日期，[cal_dt-365, cal_dt]为训练数据
    '''    
    sql = "SELECT cal_dt, \
                  stn_cd, \
                  start_tm, \
                  end_tm, \
                  tm_seg, \
                  getin_qty  \
             from ads.ads_a01_bi_travel_tm_d   \
            where cal_dt between date_add('" + cal_dt + "', -365) and '" + cal_dt  + "' " + condition + " \
            order by stn_cd asc, cal_dt ASC, start_tm asc   \
           "
#     print(sql)
    data = dbutils.execSqlPyhive(sql)
    data['getin_qty'] = data['getin_qty'].astype(np.float)

    return data

def get_getin_data_arimax(cal_dt, condition=""):
    '''
    功能：获取进站量数据
    cal_dt：数据日期，[cal_dt-365, cal_dt]为训练数据
    '''    
    sql = "SELECT s.cal_dt, \
                  s.quater,\
                  (CASE WHEN s.is_work_day = 'Y' THEN 1 ELSE 0 end) is_work_day,\
                  p.stn_cd,\
                  p.start_tm,\
                  p.end_tm,\
                  p.getin_qty  \
             FROM dim.dim_a01_pub_date_parm s  \
             LEFT JOIN ads.ads_a01_bi_travel_tm_d p  \
               on s.cal_dt = p.cal_dt  \
            where s.cal_dt between date_add('" + cal_dt + "', -365) and '" + cal_dt  + "' " + condition + " \
            ORDER BY s.cal_dt ASC, p.stn_cd asc, p.start_tm asc  \
        "
#     print(sql)
    data = dbutils.execSqlPyhive(sql)
    data['getin_qty'] = data['getin_qty'].astype(np.float)

    return data


def adf_test(X):
    '''
    功能：判断数据X是否平稳。P-value是否非常接近0，接近0，则是平稳的，否则，不平稳
    X：待检测数据
    '''
    adf = adfuller(X, autolag='AIC')
    return adf[1]

def mean_absolute_percentage_error(y_true, y_pred): 
    '''
    功能：计算MAPE
    y_true： 真实值
    y_pred： 预测值
    '''
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        # P-value 是否非常接近0，接近0，则是平稳的，否则，不平稳。
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        # acf
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        # pacf
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()

def plotARIMAX(endog_train, exog_test, model, n_steps, offset):
    """
    功能：对预测值和真实值进行可视化展示
    endog_train - 训练时间序列
    exog_test   - 用于预测的exog
    model   - ARIMAX best model
    n_steps - 要预测的未来周期数    
    """

    # 将预测数据和真实数据进行整合
    data = endog_train.copy()
    data.columns = ['true']   # 更改列名
    data['arimax_model'] = model.fittedvalues  # 预测数据

    # 由于周期及差分原因，前offset(s+d)步无法预测
    data['arimax_model'][:offset] = np.NaN

    # 预测未来 n_steps
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps-1, exog=exog_test)
    forecast = np.append(data['arimax_model'], forecast)

    # 计算预测误差
    error = mean_absolute_percentage_error(data['true'][offset:], data['arimax_model'][offset:])
    
    # 构造x轴数据
    x_true = [datetime.strptime(d, '%Y-%m-%d').date() for d in data.index ] 
    x_pred = [datetime.strptime(d, '%Y-%m-%d').date() for d in data.index ] 
    x_add = pd.date_range(start=data.index[-1], periods=n_steps).date
    x_pred.extend(x_add)

    # 结果可视化
    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error), {'size':16})
    # 预测值
    plt.plot(x_pred, forecast, color='#EE7E2D', label="predict")
    plt.axvspan(datetime.strptime(data.index[-1], '%Y-%m-%d' ), x_pred[-1],  alpha=0.5, color='lightgrey')
    # 真实值
    plt.plot(x_true, data.true, color='#4273C5', label="true")
    
    plt.tick_params(labelsize=14)
    plt.xlabel('日期',  {'size':15})
    plt.ylabel('进站量(人数)', {'size':15})
    plt.legend()
    plt.grid(True)
    
def plotSARIMA(series, model, n_steps, offset):
    """
    功能：对预测值和真实值进行可视化展示
        
    series  - 原始时间序列
    model   - SARIMA best model
    n_steps - 要预测的未来周期数    
    """

    # 将预测数据和真实数据进行整合
    data = series.copy()
    data.columns = ['true']   # 更改列名
    data['sarima_model'] = model.fittedvalues  # 预测数据

    # 由于周期及差分原因，前offset(s+d)步无法预测
    data['sarima_model'][:offset] = np.NaN

    # 预测未来 n_steps
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps-1)
    forecast = np.append(data['sarima_model'], forecast)

    # 计算预测误差
    error = mean_absolute_percentage_error(data['true'][offset:], data['sarima_model'][offset:])
    
    # 构造x轴数据
    x_true = [datetime.strptime(d, '%Y-%m-%d').date() for d in data.index ] 
    x_pred = [datetime.strptime(d, '%Y-%m-%d').date() for d in data.index ] 
    x_add = pd.date_range(start=data.index[-1], periods=n_steps).date
    x_pred.extend(x_add)

    # 结果可视化
    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error), {'size':16})
    # 预测值
    plt.plot(x_pred, forecast, color='#EE7E2D', label="predict")
    plt.axvspan(datetime.strptime(data.index[-1], '%Y-%m-%d' ), x_pred[-1],  alpha=0.5, color='lightgrey')
    # 真实值
    plt.plot(x_true, data.true, color='#4273C5', label="true")
    
    plt.tick_params(labelsize=14)
    plt.xlabel('日期',  {'size':15})
    plt.ylabel('进站量(人数)', {'size':15})
    plt.legend()
    plt.grid(True)
    
def optimizeARIMAX(endog, exog, parameters_list, d, D, s):
    """ ARIMAX参数搜索，返回每组参数及其AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):        
        # 因某些参数无法使模型收敛，因此需要使用 try-except 跳过
        try:
            # model = SARIMAX(endog=y, exog=x, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))

            model=sm.tsa.statespace.SARIMAX(endog=endog,
                                            exog=exog,
                                            order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s),
                                            trend='n').fit(disp=-1)
        except:
            continue
        aic = model.aic

        # 保存best model，AIC和对应的参数
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
        print([param, model.aic])
        
    if len(results) == 0:
        print('错误：模型未收敛，请调整参数。')
        return None

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # 根据AIC升序排序，AIC越小，模型越好
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table
    
def optimizeSARIMA(data, parameters_list, d, D, s):
    """ SARIMA参数搜索，返回每组参数及其AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):        
        # 因某些参数无法使模型收敛，因此需要使用 try-except 跳过
        try:
            model=sm.tsa.statespace.SARIMAX(data, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s),
                                           trend='n').fit(disp=-1)
        except:
            continue
        aic = model.aic

        # 保存best model，AIC和对应的参数
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
        print([param, model.aic])
        
    if len(results) == 0:
        print('错误：模型未收敛，请调整参数。')
        return None

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # 根据AIC升序排序，AIC越小，模型越好
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table


def lbtest(ts, lags=30):
    '''
    作用：LB检验，检验数据ts是否为白噪声。如果 p > 0.05，是白噪声
    ts：残差序列
    '''
    lbvalue, p_value = acorr_ljungbox(ts, lags=[lags])
    return p_value[0]


def predict_sarima(begin, end, model):
    '''
    功能：使用SARIMA进行预测
    
    begin: 起点
    end:   终点
    model: 已训练的模型
    '''
    pred = model.predict(begin, end)
    return pred.astype(np.int)
    
def get_mae_score(pred_value, true_value):
    '''
    功能：返回mae score
    '''
    return np.round(mean_absolute_percentage_error(pred_value, true_value) / 100, 4)

def plot_true_vs_pred(x, true_value, pred_value):
    '''
    功能：对预测结果和真实结果做可视化
    
    x：坐标轴
    true_value：真实值
    pred_value：预测值
    '''
    #myfont = FontProperties(fname=PlotConfig.font_path, size=PlotConfig.legend_size)
    
    error = mean_absolute_percentage_error(pred_value, true_value)
    
    # 结果可视化
    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error), {'size':16})
    # 预测值
    plt.plot(x, pred_value, color='#EE7E2D', label="predict")
    # 真实值
    plt.plot(x, true_value, color='#4273C5', label="true")
    
    plt.tick_params(labelsize=14)
    plt.xlabel('日期',   {'size':15})
    plt.ylabel('进站量(人数)', {'size':15})
    plt.legend()
    plt.grid(True)
    
def predict_arimax(begin, end, exog, model):
    '''
    功能：使用ARIMAX进行预测
    
    begin: 起点
    end:   终点
    exog:  用于预测的exog
    model: 已训练的模型
    '''
    pred = model.predict(begin, end, exog=exog)
    return pred.astype(np.int)


def get_arimax_exog_for_pred(cal_dt, n_steps=21):
    '''
    功能：读入ARIMAX预测时使用的exog(x)数据。如果cal_dt是2022-03-01，则读入2022-03-02 ~ 2022-03-22的quater、is_work_day
    cal_dt: 数据日期
    '''
    sql = "SELECT s.cal_dt, \
                  s.quater,\
                  (CASE WHEN s.is_work_day = 'Y' THEN 1 ELSE 0 end) is_work_day \
             FROM dim.dim_a01_pub_date_parm s  \
            where s.cal_dt between  date_add('" + cal_dt  + "',1) and date_add('" + cal_dt + "', 21)   \
            ORDER BY s.cal_dt ASC  \
          "
    test_data = dbutils.execSqlPyhive(sql)
#     x_test = test_data[['quater', 'is_work_day']]
    return test_data