# -*- coding:utf-8 -*-
# @Time : 2022/6/29 8:44 下午
# @Author : Bin Bin Xue
# @File : Chart6_DataExp&Plot
# @Project : Machine Learning Pocket Reference
# 这里介绍的是一些解决分类问题的方法（包括skl的常用分类模型、训练过程、评价指标）

# skl模型的通用方法
# fit(X,y[,sample_weight]) - 训练模型
# predict(X) - 预测类别
# predict_log_proba(X) - 预测样例属于某个类别的概率的对数值
# predict_proba(X) - 预测样例属于某个类别的概率
# score(X,y[,sample_weight]) - 获取模型的准确率

import numpy as np
import matplotlib.pyplot as plt

# skl的逻辑回归分类器
import pandas as pd
from sklearn.linear_model import (
    LogisticRegression,
)
# skl的朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
# skl的svm分类器
from sklearn.svm import SVC
# skl的knn分类器
from sklearn.neighbors import KNeighborsClassifier
# skl的决策树分类器
from sklearn.tree import DecisionTreeClassifier
# skl的随机森林分类器
from sklearn.ensemble import RandomForestClassifier
from rfpimp import permutation_importances
from sklearn.metrics import r2_score
# xgb的xgboost分类器
import xgboost as xgb
import xgbfir
# lgb的lightgbm分类器
import lightgbm as lgb
# TPOT分类器
from tpot import TPOTClassifier

# 绘制决策树
from sklearn import tree
import pydotplus
from io import StringIO
from sklearn.tree import export_graphviz

# yellowbrick的特征重要性
from yellowbrick.features import (
    FeatureImportances
)

# 自定义的工具包
import util

# 导入预处理好的Taitannike数据
df, X, y, X_train, y_train, X_test, y_test = util.dataProcess()


# 逆对数概率函数
def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


# 逻辑回归分类器
def logisticRegressModel():
    # 1.模型初始化（在这里设置模型参数）
    # 12个参数-penalty,dual,C,fit_intercept,intercept_scaling,max_iter,multi_class,class_weight,solver,tol,verbose,warm_start,n_jobs
    lr = LogisticRegression(random_state=42)
    # 2.模型训练（在这里加入训练数据，训练模型）
    lr.fit(X_train, y_train)
    # 3.输出模型结果（预测准确度）
    score = lr.score(X_test, y_test)
    print(score)
    # 4.输出预测结果（预测结果，各类别概率，概率的对数值，
    print(lr.predict(X.iloc[[0]]))
    print(lr.predict_proba(X.iloc[[0]]))
    print(lr.predict_log_proba(X.iloc[[0]]))
    print(lr.decision_function(X.iloc[[0]]))
    # 5.模型训练后的属性
    # 决策函数的截距 - 判正的基础概率
    print(lr.intercept_)
    print(inv_logit(lr.intercept_))
    # 决策函数的系数（权重）- 几次方的系数
    print(lr.coef_)
    # 决策函数的迭代次数
    print(lr.n_iter_)
    # 5.查看各特征的系数（跟预测结果正/负相关）
    cols = X.columns
    for col, val in sorted(
            zip(cols, lr.coef_[0]),
            key=lambda x: x[1],
            reverse=True,
    ):
        print(
            f"{col:10}{val:10.3f}{inv_logit(val):10.3f}"
        )
    # 6.使用yellowbrick将系数可视化
    fig, ax = plt.subplots(figsize=(6, 4))
    fi_viz = FeatureImportances(lr)
    fi_viz.fit(X, y)
    fi_viz.poof()
    fig.savefig("images/mlpr_101.png", dpi=300)


# 朴素贝叶斯分类器
def GaussianNBModel():
    # 初始化模型（设置模型参数）
    nb = GaussianNB(priors=None, var_smoothing=1e-9)
    nb.fit(X_train, y_train)
    score = nb.score(X_test, y_test)
    print(score)
    # 输出预测结果
    print(nb.predict(X.iloc[[0]]))
    print(nb.predict_proba(X.iloc[[0]]))
    print(nb.predict_log_proba(X.iloc[[0]]))
    # 模型训练后的属性
    print(nb.class_prior_)
    print(nb.class_count_)
    print(nb.theta_)
    print(nb.sigma_)
    print(nb.epsilon_)


# SVM分类器
def SVMModel():
    # 有14个参数，其中probability=True开启后会降低速度
    svc = SVC(random_state=42, probability=True)
    svc.fit(X_train, y_train)
    # 输出预测结果
    score = svc.score(X_test, y_test)
    print(score)
    print(svc.predict(X.iloc[[0]]))
    print(svc.predict_proba(X.iloc[[0]]))
    print(svc.predict_log_proba(X.iloc[[0]]))


# K近邻分类器
def knnModel():
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    score = knc.score(X_test, y_test)
    print(score)
    print(knc.predict(X.iloc[[0]]))
    print(knc.predict_proba(X.iloc[[0]]))


# 绘制决策树
def DTGraph(dt):
    dot_data = StringIO()
    tree.export_graphviz(
        dt,
        out_file=dot_data,
        feature_names=X.columns,
        class_names=['Died', 'Survived'],
        filled=True,
    )
    g = pydotplus.graph_from_dot_data(
        dot_data.getvalue()
    )
    g.write_png('images/mlpr_102.png')


# 决策树分类器
def DTModel():
    dt = DecisionTreeClassifier(random_state=42, max_depth=3)
    dt.fit(X_train, y_train)
    score = dt.score(X_test, y_test)
    print(score)
    print(dt.predict(X.iloc[[0]]))
    print(dt.predict_proba(X.iloc[[0]]))
    # print(dt.predict_log_proba(X.iloc[[0]]))
    # 模型训练后的参数R
    # print(dt.classes_)
    # print(dt.feature_importances_)
    # print(dt.n_classes_)
    # print(dt.n_features_)
    # print(dt.tree_)
    # 绘制决策树（还要安装一个配置软件）
    # DTGraph(dt)


# 随机森林的oob分数
def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))


# 随机森林分类器
def RFModel():
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    print(score)
    print(rf.predict(X.iloc[[0]]))
    print(rf.predict_proba(X.iloc[[0]]))
    print(rf.predict_log_proba(X.iloc[[0]]))
    # 特征重要性（gini划分）
    for col, val in sorted(
            zip(X.columns, rf.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
    )[:5]:
        print(f'{col:10}{val:10.3f}')
    # 置换重要性（比特征重要性具有更好的度量）
    perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, r2)
    print(perm_imp_rfpimp)


def XgbModel():
    xgb_class = xgb.XGBClassifier(random_state=42, early_stopping_rounds=10)
    xgb_class.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
    )
    score = xgb_class.score(X_test, y_test)
    print(score)
    print(xgb_class.predict(X.iloc[[0]]))
    print(xgb_class.predict_proba(X.iloc[[0]]))
    # 输出特征重要性(根据节点的平均增益）
    for col, val in sorted(
            zip(
                X.columns,
                xgb_class.feature_importances_,
            ),
            key=lambda x: x[1],
            reverse=True,
    )[:5]:
        print(f"{col:10}{val:10.3f}")
    # 用xgb库绘制特征重要性图（其中importance_type参数标志不同的重要性度量,默认为weight）
    # fig,ax=plt.subplots(figsize=(6,4))
    # xgb.plot_importance(xgb_class,ax=ax)
    # fig.savefig('images/mlpr_1005.png',dpi=300)
    # xgb库绘制单独棵树(也是需要安装graphviz这个配置软件）
    # booster = xgb_class.get_booster()
    # print(booster.get_dump()[0])
    # fig,ax = plt.subplots(figsize=(6,4))
    # xgb.plot_tree(xgb_class,ax=ax,num_trees=0)
    # fig.savefig('images/mlpr_1007.png',dpi=300)
    # 使用xgbfir库实现多种特征重要性度量方法(通过该方法能很好地找到最适合的特征组合）
    xgbfir.saveXgbFI(
        xgb_class,
        feature_names=X.columns,
        OutputXlsxFile='fir.xlsx',
    )
    # pd.read_excel('fir.xlsx').head(3).T


# lightgbm模型
def lgbModel():
    lgbm_class = lgb.LGBMClassifier(random_state=42)
    lgbm_class.fit(X_train, y_train)
    score = lgbm_class.score(X_test, y_test)
    print(score)
    print(lgbm_class.predict(X.iloc[[0]]))
    print(lgbm_class.predict_proba(X.iloc[[0]]))
    # 输出特征重要性（这里的特征重要性默认为'splits',按一个特征的使用次数,可以设置importance_type=gain换掉）
    for col, val in sorted(
        zip(X.columns, lgbm_class.feature_importances_),
        key = lambda x: x[1],
        reverse = True
    )[:5]:
        print(f'{col:10}{val:10.3f}')
    # 用lgbm库绘制特征重要性表
    # fig,ax = plt.subplots(figsize=(6,4))
    # lgb.plot_importance(lgbm_class,ax=ax)
    # fig.tight_layout()
    # fig.savefig('images/mlpr_1008.png',dpi=300)
    # 用lgbm库绘制lgbm树(也是需要安装graphiviz这个软件）
    fig, ax = plt.subplots(figsize=(6,4))
    lgb.plot_tree(lgbm_class,tree_index=0,ax=ax)
    fig.savefig('images/mlpr_1009.png',dpi=300)


# TPOT模型
def TPOTModel():
    tc = TPOTClassifier(generations=2)
    tc.fit(X_train,y_train)
    score = tc.score(X_test,y_test)
    print(score)
    print(tc.predict(X.iloc[[0]]))
    print(tc.predict_proba(X.iloc[[0]]))
    # 导出程序流水线
    tc.export('tpot_exported_pipeline.py')


if __name__ == '__main__':
    # 1.逻辑回归分类器
    # logisticRegressModel()
    # 2.朴素贝叶斯分类器（假设各特征相互独立，适合特征较多的数据，但不能判断特征之间的关系）
    # skl提供了3个贝叶斯类：GaussianNB、MultinomialNB、BernoulliNB
    # 第一个假定高斯分布（适合连续型特征值，且呈高斯分布）
    # 第二个适用于离散型计数特征；第三个适用于离散型布尔特征值
    # GaussianNBModel()
    # 3.支持向量机（skl有三种实现-svc,linearSVC,linear_model.SGDClassifier类）
    # svm可支持线性空间，用核技术还支持非线性空间（默认核为RBF）
    # SVMModel()
    # 4.k近邻（k近邻要选取合适的距离度量标准，高维导致的维灾会使该分类器表现较差）
    # knnModel()
    # 5.决策树（优点是支持非数值型数据，且可处理非线性关系，缺点是容易过拟合，要用max_depth和交叉检验等控制）
    # DTModel()
    # 6.随机森林（多棵决策树集成bagging，解决过拟合，降低方差）
    # RFModel()
    # 7.XGBoost（极限梯度提升树）
    # XgbModel()
    # 8.LightGBM（比xgb更快,性能更高）
    # lgbModel()
    # 9.TPOT（用遗传算法尝试不同的模型和集成方式，会需要较长的时间）
    # TPOT牺牲时间换性能（迭代次数和保留的个体数越多，性能越高）
    TPOTModel()



