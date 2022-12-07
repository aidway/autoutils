import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker
import mpl_toolkits.axisartist as axisartist
import matplotlib.dates as mdate
from datetime import datetime
import os


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



def plot_one(x, y, x_label, y_label, title, ptConfig=PlotConfig):
    '''
    x：日期，date类型
    y：时序数据
    '''
    myfont = FontProperties(fname=ptConfig.font_path, size=ptConfig.legend_size)
    fig = plt.figure(figsize=(ptConfig.fig_x, ptConfig.fig_y))
    
    ax = fig.add_subplot(111)
    
    #plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_formatter(mdate.DateFormatter(ptConfig.date_type))
    plt.gca().xaxis.set_major_locator(mdate.MonthLocator())
    
    #### 不使用科学计数法
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    
    ax.set_xlabel(x_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
    ax.set_ylabel(y_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
    
    plt.tick_params(labelsize=ptConfig.tick_size)
    
    # 画图
    ax.plot(x, y, color=ptConfig.y1_c, label=y_label, lw=ptConfig.lw)
    
    ## 坐标轴
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    
    # plt.xticks(rotation=30)
    # plt.legend(prop=myfont)
    plt.title(title, {'fontproperties':myfont, 'size':ptConfig.title_size})
    
    
def plot_two(x, y1, y2, x_label, y_label,y1_legend, y2_legend,title, ptConfig=PlotConfig):
    '''
    作用：两列数据对比，单坐标轴
    x: x轴数据，如时间
    y1: 第一列，如交易量
    y2: 第二列，如持仓量
    x_label: x轴名称
    y_label: y轴名称
    y1_legend: 第一列的图例
    y2_legend: 第二列的图例
    title: 标题
    ptConfig: 参数，默认为PlotConfig
    '''
    myfont = FontProperties(fname=ptConfig.font_path, size=ptConfig.legend_size)
    fig = plt.figure(figsize=(ptConfig.fig_x, ptConfig.fig_y))
    
    
    
    ax = fig.add_subplot(111)
    
    plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdate.MonthLocator())
    
    
    ## ax
    ax.plot(x, y1, color=ptConfig.y1_c, label=y1_legend, lw=ptConfig.lw)
    ax.plot(x, y2, color=ptConfig.y2_c, label=y2_legend, lw=ptConfig.lw)
    
    
    #### 不使用科学计数法
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    
    ax.set_xlabel(x_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
    ax.set_ylabel(y_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
    
    ## 坐标轴
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    
    plt.tick_params(labelsize=ptConfig.tick_size)
#     plt.xticks(np.arange(0, len(x), ptConfig.tick_len), x[::ptConfig.tick_len],  rotation=0)
    plt.legend(prop=myfont)
    plt.title(title, {'fontproperties':myfont, 'size':ptConfig.title_size})
    

def plot_vs(x, y1, y2, y1_label, y2_label,title , ptConfig=PlotConfig):
    '''
    作用：两列数据对比，双坐标轴
    x: x轴数据，如时间
    y1: 第一列，如交易量
    y2: 第二列，如持仓量
    x_label: x轴名称
    y_label: y轴名称
    title: 标题
    ptConfig: 参数，默认为PlotConfig
    '''
    myfont = FontProperties(fname=ptConfig.font_path, size=ptConfig.legend_size)
    fig = plt.figure(figsize=(ptConfig.fig_x, ptConfig.fig_y))
 

    ax1 = fig.add_subplot(111)
    
#     plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
#     plt.gca().xaxis.set_major_locator(mdate.MonthLocator())
    
    
    ## ax1
    plt1 = ax1.plot(x, y1, color=ptConfig.y1_c, label=y1_label, lw=ptConfig.lw)
    ax1.set_title(title, {'fontproperties':myfont, 'size':ptConfig.title_size})
    ax1.set_ylabel(y1_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
    ax1.tick_params(labelsize=ptConfig.tick_size)
    #### 不使用科学计数法
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax1.yaxis.get_major_formatter().set_useOffset(False)
    
    ## ax2
    ax2 = ax1.twinx()
    plt2 = ax2.plot(x, y2, color=ptConfig.y2_c, label=y2_label, lw=ptConfig.lw)
    ax2.set_ylabel(y2_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
    ax2.tick_params(labelsize=ptConfig.tick_size)
    
    ## 坐标轴
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    
    # 图例
    lns = plt1 + plt2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, prop=myfont)
    
def plot_subplots(data, id_list, x_column,y1_column,y2_column,y1_label, y2_label,title,x_num, y_num, ptConfig=PlotConfig):
    '''
    作用：绘制子图
    data: 传入数据
    id_list: 用于对数据进行分类的列，如合约号
    x_column: x轴数据，如时间
    y1_column: 第一列数据，如交易量
    y2_column: 第二列数据，如持仓量
    x: x轴数据，如时间
    y1_label: 第一列的图例
    y2_label: 第二列的图例
    title: 标题
    x_num: x方向子图个数
    y_num: y防线子图个数
    ptConfig: 参数，默认为PlotConfig
    '''
    myfont = FontProperties(fname=ptConfig.font_path, size=ptConfig.legend_size)
    fig = plt.figure(figsize=(ptConfig.fig_x, ptConfig.fig_y))
    plt.subplots_adjust(hspace=0.15, wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)


    for i in enumerate(id_list):
        df = data[data.commid == i[1]]
        x_init = df[x_column]
        x = [datetime.strptime(d, '%Y-%m-%d').date() for d in x_init ] # time: .dt.strftime('%Y-%m-%d')
        y1 = df[y1_column]
        y2 = df[y2_column]
        
        ax1 = fig.add_subplot(x_num*100+ y_num*10+ i[0]+1)
        
        plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdate.YearLocator())  # DayLocator MonthLocator
        
        ## ax1
        plt1 = ax1.plot(x, y1, color=ptConfig.y1_c, label=y1_label, lw=ptConfig.lw)
        ax1.set_title(title, {'fontproperties':myfont, 'size':ptConfig.title_size})
        ax1.set_ylabel(y1_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
        ax1.tick_params(labelsize=ptConfig.tick_size)
        #### 不使用科学计数法
        ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.yaxis.get_major_formatter().set_scientific(False)
        ax1.yaxis.get_major_formatter().set_useOffset(False)
    
        ## ax2
        ax2 = ax1.twinx()
        plt2 = ax2.plot(x, y2, color=ptConfig.y2_c, label=y2_label, lw=ptConfig.lw)
        ax2.set_ylabel(y2_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
        ax2.tick_params(labelsize=ptConfig.tick_size)
    
        ## 坐标轴
        #plt.gcf().autofmt_xdate()  # 自动旋转日期标记，加上该句后，只有最后的子图显示坐标轴
    
        # 图例
        lns = plt1 + plt2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, prop=myfont)


    
def plot_three(x, y1, y2, y3,x_label, y_label,y1_legend, y2_legend,y3_legend,title, ptConfig=PlotConfig):
    myfont = FontProperties(fname=ptConfig.font_path, size=ptConfig.legend_size)
    fig = plt.figure(figsize=(ptConfig.fig_x, ptConfig.fig_y))
    
    ax = fig.add_subplot(111)
    
    
    ## ax
    ax.plot(x, y1, color=ptConfig.y1_c, label=y1_legend, lw=ptConfig.lw)
    ax.plot(x, y2, color=ptConfig.y2_c, label=y2_legend, lw=ptConfig.lw)
    ax.plot(x, y3, color=ptConfig.y3_c, label=y3_legend, lw=ptConfig.lw)

    
    
    ax.set_xlabel(x_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
    ax.set_ylabel(y_label, {'fontproperties':myfont, 'size':ptConfig.label_size})
    
    plt.tick_params(labelsize=ptConfig.tick_size)
    plt.legend(prop=myfont)
    plt.title(title, {'fontproperties':myfont, 'size':ptConfig.title_size})
     

    
def plot_descartes(x, y, ptConfig=PlotConfig):
    '''
    作用：绘制笛卡尔坐标系
    x: x轴数据，如时间
    y: y轴数据，如交易量
    ptConfig: 参数，默认为PlotConfig
    '''
    #创建画布
    fig = plt.figure(figsize=(8, 8))
    #使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)  
    #将绘图区对象添加到画布中
    fig.add_axes(ax)

    #通过set_visible方法设置绘图区所有坐标轴隐藏
    ax.axis[:].set_visible(False)

    #ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"] = ax.new_floating_axis(0,0)
    #给x坐标轴加上箭头
    ax.axis["x"].set_axisline_style("->", size = 1.0)
    #添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1,0)
    ax.axis["y"].set_axisline_style("->", size = 1.0)
    #设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("bottom")
    ax.axis["y"].set_axis_direction("left")

    
    #绘制图形
    plt.scatter(x, y, c=ptConfig.y2_c)
    
    
    
