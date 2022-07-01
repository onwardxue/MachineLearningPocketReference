# -*- coding:utf-8 -*-
# @Time : 2022/6/29 8:44 下午
# @Author : Bin Bin Xue
# @File : Chart6_DataExp&Plot
# @Project : Machine Learning Pocket Reference
# 这里介绍的是一些特征选择的常用方法
# 一些概念：
# 选用不相关的特征会引起负面效果，选用存在有相关关系的特征则可能导致回归不可解释
# 维灾：增加维度，数据变得更稀疏，效果变差
# 删除掉不重要的特征对预测效果影响不大

import matplotlib.pyplot as plt
import pandas as pd
# skl的预处理包
from sklearn import (
    ensemble,
    feature_selection
)
# yellowbrick包
from yellowbrick.features import RFECV

# 自定义的工具包
import util

# 导入预处理好的Taitannike数据
df, X, y, X_train, y_train, X_test, y_test = util.dataProcess()

if __name__ == '__main__':
    # 1. 递归消除特征法进行特征选择（递归删除最弱特征，直到特征数满足指定要求）
    # 结果图表会显示选择的特征数量和分类器性能之间的关系
    # fig, ax = plt.subplots(figsize=(6, 4))
    # rfe = RFECV(
    #     ensemble.RandomForestClassifier(
    #         n_estimators=100
    #     ),
    #     cv=5,
    # )
    # rfe.fit(X,y)
    # print(rfe.rfe_estimator_.ranking_)
    # print(rfe.rfe_estimator_.n_features_)
    # print(rfe.rfe_estimator_.support_)
    # rfe.poof()
    # fig.savefig('images/mlpr_083.png',dpi=300)
    # 找出其中10个最重要的特征
    # model = ensemble.RandomForestClassifier(
    #     n_estimators=100
    # )
    # # 设置要输出的重要性特征数量
    # rfe = feature_selection.RFE(model,n_features_to_select=4,verbose=-1)
    # rfe.fit(X,y)
    # print(X.columns[rfe.support_])
    # 2. 使用互信息查看特征重要性
    mic = feature_selection.mutual_info_classif(
        X,y
    )
    fig,ax = plt.subplots(figsize=(10,8))
    (
        pd.DataFrame(
            {'feature': X.columns, 'vimp':mic}
        )
        .set_index('feature')
        .plot.barh(ax=ax)
    )
    fig.savefig('images/mlpr_084.png')
    # 3.使用PCA查看特征的贡献最大（无监督算法，没有考虑标签y）
    # 在第17章进行讲解