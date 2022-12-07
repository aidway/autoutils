from sklearn.metrics import *
from sklearn.model_selection import learning_curve 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_confusion_matrix  as plt_matrix



def plot_lgb_importances(model, plot=False, num=10):
    '''
    功能：lgb模型特征重要性画图或输出
    
    参数：
      - model：已训练的模型
      - plot：如果为True，则画图，否则，输出特征重要性
      - num：特征个数
      
    返回：None
    '''

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
        
        
def plot_auc(true_value, y_pred_score):
    '''
    功能: 绘制AUC曲线
    
    参数:
     - true_value: 真实值
     - y_pred_score: 预测的分数
     
    返回: None
    '''
    fpr, tpr, thresholds = roc_curve(true_value, y_pred_score[:, 1])
    AUC = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='AUC = %0.2f'% AUC)
    plt.legend(loc='lower right')
    plt.title('ROC curve')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
def plot_pr(true_value, y_pred_score):
    '''
    功能: 绘制Precision-Recall曲线
    
    参数:
     - true_value: 真实值
     - y_pred_score: 预测的分数
     
    返回: best_threshold, best_f1_score
    '''
    precision, recall, thresholds = precision_recall_curve(true_value, y_pred_score[:,1])
    f1 = 2 * (precision * recall) / (precision + recall)

    ind = np.argmax(f1)
    best_f1_score = f1[ind]
    best_threshold = thresholds[ind]

    plt.plot(thresholds, precision[:-1], label='precision', color='#EE7E2D')
    plt.plot(thresholds, recall[:-1], label='recall', color= '#4273C5')
    plt.plot(thresholds, f1[:-1], label='f1_score', color='#9F59A6')
    plt.scatter(best_threshold, best_f1_score, color='red', s=100)

    plt.xlabel('thresholds')
    plt.legend(loc='best')
    plt.title('P-R curve with thresholds')
    
    return round(best_threshold, 3), round(best_f1_score, 3)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    功能：画出data在某模型上的learning curve.
    
    参数：
      - estimator : 你用的分类器。
      - title : 表格的标题。
      - X : 输入的feature，numpy类型
      - y : 输入的target vector
      - ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
      - cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
      
    返回：None
    """
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()
    
def class_metrics(true_value, pred_value, pred_value_score):
    '''
    功能：输出分类问题的评估指标
    
    参数：
      - true_value：真实值
      - pred_value：预测值
      - pred_value_score：预测的分数
    
    function: class model score, true vs test
    '''
    print('accuracy_score: %.4f' % accuracy_score(true_value, pred_value))
    print('precision_score:%.4f' % precision_score(true_value, pred_value))
    print('recall_score: %.4f' % recall_score(true_value, pred_value))
    print('confusion_matrix:\n' , confusion_matrix(true_value, pred_value))
    print('roc_auc_score:%.4f' % roc_auc_score(true_value, pred_value_score[:,1]))
    print('f1_score:%.4f' %f1_score(true_value, pred_value))
    

def plot_confusion_matrix(model, X, y):
    '''
    功能：绘制confusion matrix
    
    参数：
      - model:
      - X:
      - y:
      
    返回: None
    '''
    plt_matrix(model, X, y)  
    plt.title('Confusion Matrix')        


if __name__ == "__main__" :
    plot_lgb_importances(model, num=30, plot=True)
    plot_auc(true_value, y_pred_score)
    plt_pr(true_value, y_pred_score)
    plot_learning_curve(modle, "logistics learning curve",x_test, true_value, ylim=(0.8, 1.01),train_sizes=np.linspace(.05, 0.2, 5))
    class_metrics(true_value, y_pred, y_pred_score)



