# -*- coding:utf-8 -*-
# @Time : 2022/6/29 8:44 下午
# @Author : Bin Bin Xue
# @File : Chart6_DataExp&Plot
# @Project : Machine Learning Pocket Reference
# 这里介绍以下度量标准和评估工具：混淆矩阵、scikit-learn常用度量标准、分类报告和几种可视化工具

import pandas as pd
import matplotlib.pyplot as plt
# 导入混淆矩阵
from sklearn.metrics import confusion_matrix
# 导入绘制库
from yellowbrick.classifier import (
    ConfusionMatrix
)
# 导入度量标准
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
# 导入yellowbrick的绘图工具
from yellowbrick.classifier import (
    ClassificationReport,
    ROCAUC,
    PrecisionRecallCurve,
    ClassBalance,
    ClassPredictionError,
    DiscriminationThreshold,
)
# 导入scikit-plot库
from scikitplot.metrics import (
    plot_cumulative_gain,
    plot_lift_curve,
)
from sklearn.tree import DecisionTreeClassifier
import util

# 导入预处理好的Taitannike数据
df, X, y, X_train, y_train, X_test, y_test = util.dataProcess()


# 混淆矩阵
def cm(dt):
    # 得到四个类各类的数量
    y_predict = dt.predict(X_test)
    tp = (
            (y_test == 1) & (y_test == y_predict)
    ).sum()
    tn = (
            (y_test == 0) & (y_test == y_predict)
    ).sum()
    fp = (
            (y_test == 1) & (y_test != y_predict)
    ).sum()
    fn = (
            (y_test == 0) & (y_test != y_predict)
    ).sum()
    print(tp, tn, fp, fn)
    # 用skl的方法创建一个混淆矩阵
    metrix = pd.DataFrame(
        confusion_matrix(y_test, y_predict),
        columns=[
            'Predict died',
            'Predict Survive',
        ],
        index=['True Death', 'True Survive'],
    )
    print(metrix)
    # 用yellowbrick绘制混淆矩阵
    mapping = {0: 'died', 1: 'survived'}
    fig, ax = plt.subplots(figsize=(6, 6))
    cm_viz = ConfusionMatrix(
        dt,
        classes=['died', 'survived'],
        label_encorder=mapping,
    )
    cm_viz.score(X_test, y_test)
    cm_viz.poof()
    fig.savefig('images/mlpr_1202.png', dpi=300)


# 分类报告
def report(dt):
    fig, ax = plt.subplots(figsize=(6, 3))
    cm_viz = ClassificationReport(
        dt,
        classes=['died', 'survived'],
    )
    cm_viz.score(X_test, y_test)
    cm_viz.poof()
    fig.savefig('images/mlpr_1203.png')


# 绘制roc曲线
def rocCurve(dt):
    fig, ax = plt.subplots(figsize=(6, 6))
    roc_viz = ROCAUC(dt)
    roc_viz.score(X_test, y_test)
    roc_viz.poof()
    fig.savefig('images/mlpr_1204.png', dpi=300)


# 绘制精确率-召回率曲线
def apCurve():
    fig, ax = plt.subplots(figsize=(6, 4))
    viz = PrecisionRecallCurve(
        DecisionTreeClassifier(max_depth=3)
    )
    viz.fit(X_train, y_train)
    print(viz.score(X_test, y_test))
    viz.poof()
    fig.savefig('images/mlpr_1205.png', dpi=300)


# 绘制累积增益图
def cgPlot(dt):
    fig, ax = plt.subplots(figsize=(6, 6))
    y_probas = dt.predict_proba(X_test)
    plot_cumulative_gain(
        y_test, y_probas, ax=ax
    )
    fig.savefig('images/mlpr_1206.png', dpi=300, bbox_inches='tight', )


# 提升曲线
def liftCurve():
    fig, ax = plt.subplots(figsize=(6, 6))
    y_probas = dt.predict_proba(X_test)
    plot_lift_curve(
        y_test, y_probas, ax=ax
    )
    fig.savefig('images/mlpr_1207.png', dpi=300, bbox_inches='tight')


# 绘制柱形图查看各类数量
def cbPlot():
    fig, ax = plt.subplots(figsize=(6, 6))
    cb_viz = ClassBalance(
        labels=['Died', 'Survived']
    )
    cb_viz.fit(y_test)
    cb_viz.poof()
    fig.savefig('images/mlpr_1208.png', dpi=300)


# 类别预测错误图
def cpePlot(dt):
    fig, ax = plt.subplots(figsize=(6, 3))
    cpe_viz = ClassPredictionError(
        dt, classes=['died', 'survived']
    )
    cpe_viz.score(X_test, y_test)
    cpe_viz.poof()
    fig.savefig('images/mlpr_1209.png', dpi=300)


# 判别阈值图
def dtPlot(dt):
    fig, ax = plt.subplots(figsize=(6, 5))
    dt_viz = DiscriminationThreshold(dt)
    dt_viz.fit(X, y)
    dt_viz.poof()
    fig.savefig('images/mlpr_1210.png', dpi=300)


if __name__ == '__main__':
    # 决策树分类器
    dt = DecisionTreeClassifier(random_state=42, max_depth=3)
    dt.fit(X_train, y_train)
    # 1.混淆矩阵（二分类器的四种结果，表现良好的分类器的真正类和真负类比例较高（预测正确））
    # cm(dt)
    # 2.度量标准（用sklearn.metrics模块实现多种常用标准：accuracy,average_precision,f1,neg_log_loss,precision)
    # 2.1 准确率（accuracy 正确分类的百分比 -> (tp+tn)/(tp+tn+fp+fn)）
    y_predict = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(accuracy)
    # 2.2 召回率（tp/(tp+tn)）
    recall = recall_score(y_test, y_predict)
    print(recall)
    # 2.3 精准率（tp/(tp+fp))
    precision = precision_score(y_test, y_predict)
    print(precision)
    # 2.4 f1（召回率和精准率的调和平均数）
    f1 = f1_score(y_test, y_predict)
    print(f1)
    # 3. 分类报告（展示正类、负类的精准率、召回率和f1值）
    # report(dt)
    # 4.ROC曲线（越凸表示效果越好,或曲线下面积(AUC)越大越好）
    roc_auc = roc_auc_score(y_test, y_predict)
    print(roc_auc)
    # 用yellowbrick绘制roc曲线 ?问题-报错，搞不懂
    # rocCurve(dt)
    # 5.精确率-召回率曲线（ROC曲线对于类不平衡可能较为乐观，评估分类器的另一种方法是精确率-召回率曲线）
    # 召回率上升，精确率通常会下降,所以需要平衡两者
    ap_score = average_precision_score(y_test, y_predict)
    print(ap_score)
    # 用yellowbrick绘制曲线
    # apCurve()
    # 6.累计增益图（取数据集中不同数据量，正负类所占的数量）
    # 如前20%的乘客包含了40%的幸存者（正例）
    # cgPlot(dt)
    # 7.lift(提升)曲线（指比基准的模型提升的幅度）
    # 如我们取概率最高的20%乘客，得到的lift是随机选择幸存者的2.2倍
    # liftCurve()
    # 8.类别平衡问题（各类样本数差别较大时，准确率不是一个很好的度量标准）
    # 使用分层抽样切分训练集和测试集能保证各组中类的比例基本一致
    # cbPlot()
    # 9.混淆矩阵的另一种可视化-类别预测错误图
    # cpePlot(dt)
    # 10.判别阈值图
    dtPlot(dt)
