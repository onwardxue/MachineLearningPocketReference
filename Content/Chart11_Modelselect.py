# -*- coding:utf-8 -*-
# @Time : 2022/6/29 8:44 下午
# @Author : Bin Bin Xue
# @File : Chart6_DataExp&Plot
# @Project : Machine Learning Pocket Reference
# 这里介绍超参数的优化方法，并探讨模型是否需要更多数据来提升效果
# 两个内容：验证曲线和学习曲线

# 导入yellowbrick下的验证曲线和学习曲线方法
from yellowbrick.model_selection import(
    ValidationCurve,
    LearningCurve,
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 自定义的工具包
import util

# 导入预处理好的Taitannike数据
df, X, y, X_train, y_train, X_test, y_test = util.dataProcess()

# 验证曲线
def vc():
    fig,ax = plt.subplots(figsize=(6,4))
    vc_viz = ValidationCurve(
        RandomForestClassifier(n_estimators=100),
        param_name='max_depth',
        param_range=np.arange(1,11),
        cv = 10,
        n_jobs=-1,
    )
    vc_viz.fit(X,y)
    vc_viz.poof()
    fig.savefig('images/mlpr_1101.png',dpi=300)


# 学习曲线
def lc():
    fig,ax=plt.subplots(figsize=(6, 4))
    lc3_viz = LearningCurve(
        RandomForestClassifier(n_estimators=100),
        cv=10,
    )
    lc3_viz.fit(X, y)
    lc3_viz.poof()
    fig.savefig('images/mlpr_1102.png',dpi=300)


if __name__ == '__main__':
    # 1.验证曲线(查看某个参数变化对某个度量值的影响）
    # 举例：看看随机森林模型的超参数max_depth的变化是否影响模型的性能（auc)
    # 可以通过参数scoring设置分类的评价指标(auc,average_precision,f1,f1_micro,recall,precision...)
    vc()
    # 2.学习曲线（查看为模型训练的数据集是否足够，探索偏差和方差的关系（欠拟合和过拟合））
    # 通过学习曲线，可看出验证集分数到达平台期，表明增加更多的数据无助于模型的改善
    lc()
