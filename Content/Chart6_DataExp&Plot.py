# -*- coding:utf-8 -*-
# @Time : 2022/6/29 8:44 下午
# @Author : Bin Bin Xue
# @File : Chart6_DataExp&Plot
# @Project : Machine Learning Pocket Reference
# 这里介绍的是一些常用于观察数据集的方法

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# seaborn是一个可视化库，相对于对matplotlib的二次封装
import seaborn as sns
# yellowbrick绘制数据图
from yellowbrick.features import (
    JointPlotVisualizer,
    Rank2D,
    RadViz,
)
# 自定义的工具包
import util

# 导入预处理好的Taitannike数据
df, X, y, X_train, y_train, X_test, y_test = util.dataProcess()


# pandas的plot绘制直方图
def histplot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    df.Fare.plot(kind='hist', ax=ax)
    fig.savefig('images/mlpr_060.png', dpi=300)


# seaborn绘制直方图 ？问题，有错误解决不了
def sbhistplot(X_train, y_train):
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = y_train == 1
    ax = sns.displot(X_train[mask].Fare, label='survied')
    ax = sns.displot(X_train[~mask].Fare, label='died')
    ax.set_xlim(-1.5, 1.5)
    ax.legend()
    fig.savefig('images/mlpr_061.png', dpi=300, bbox_inches='tight')

# 绘制散点图
def scaplot(X):
    fig,ax = plt.subplots(figsize=(6,4))
    X.plot.scatter(
        x='Age',y='Fare',ax=ax,alpha=0.3
    )
    print(X.age.corr(X.Fare))
    fig.savefig('images/mlpr_063.png',dpi=300)

# yellowbrick绘制双关系图(两个变量之间的关系）
def yellowplot(X):
    fig,ax = plt.subplots(figsize=(6, 6))
    jpv = JointPlotVisualizer(
        feature='age', target='fare'
    )
    jpv.fit(X['Age'],X['Fare'])
    jpv.poof()
    fig.savefig('images/mlpr_064.png',dpi=300)

# 绘制盒型图（查看数据分布）
def sbplot2(X,y):
    fig,ax=plt.subplots(figsize=(8,6))
    new_df = X.copy()
    new_df['target'] = y
    sns.boxplot(x='target',y='Age',data=new_df)
    fig.savefig('images/mlpr_067.png',dpi=300)

# 绘制小提琴图（查看数据分布）
def sbplot3(X, y):
    fig,ax = plt.subplots(figsize=(8, 6))
    new_df = X.copy()
    new_df['target'] = y
    sns.violinplot(x='target',y='Sex_male',data=new_df)
    fig.savefig('images/mlpr_068.png',dpi=300)

# yellowbrick绘制热力图（看特征之间的关系）
def yellowplot2(X, y):
    fig,ax = plt.subplots(figsize=(6, 6))
    pcv = Rank2D(
        features=X.columns,algorithm='pearson'
    )
    pcv.fit(X,y)
    pcv.transform(X)
    pcv.poof()
    fig.savefig(
        'images/mlpr_0610.png',
        dpi = 300,
        bbox_inches = 'tight',
    )

# seaborne绘制热力图（看特征之间的关系）
def sbplot4(X):
    fig,ax = plt.subplots(figsize=(8, 8))
    ax = sns.heatmap(
        X.corr(),
        fmt='.2f',
        annot=True,
        ax=ax,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
    )
    fig.savefig(
        'images/mlpr_0611.png',
        dpi=300,
        bbox_inches = 'tight',
    )

# 找出数据集中高度相关的列，并删除其中一个
def correlated_columns(df,threshold=0.95):
    return (
        df.corr()
        .pipe(
            lambda df1: pd.DataFrame(
                np.tril(df1,k=-1),
                columns=df.columns,
                index=df.columns,
        )
    )
    .stack()
    .rename('pearson')
    .pipe(
            lambda s:s[
                s.abs() > threshold
            ].reset_index()
        )
    .query('level_0 not in level_1')
    )

# 绘制RadViz图（观察样例石佛按特征类别可分）
def yellowplot3(X, y):
    fig,ax = plt.subplots(figsize=(6,6))
    rv = RadViz(
        classes=['died','survied'],
        features = X.columns,
    )
    rv.fit(X,y)
    rv.transform(X)
    rv.poof()
    fig.savefig('images/mlpr_0612.png',dpi=300)

# 用pandas库绘制RadViz图
def pdplot(X, y):
    fig,ax =plt.subplots(figsize=(6,6))
    new_df = X.copy()
    new_df['target'] = y
    pd.plotting.radviz(
        new_df, 'target', ax=ax, colormap='PiYG'
    )
    fig.savefig('images/mlpr_0613.png',dpi=300)



if __name__ == '__main__':
    # 1.获取数据集中的行数和列数
    # print(df.shape)
    # 2.数据集的汇总统计-每列（或行）的数量均值、标准差、最小值、最大值、四分位数
    # iloc可根据行、列标或列名获得行和列
    # print(df.describe().iloc[:, 0:-1])
    # 3.用plot绘制直方图（查看某个属性的数值分布）
    # histplot(df)
    # 3.1 用seaborn绘制直方图（查看不同价位船乘客的存活情况）
    # sbhistplot(X_train, y_train)
    # 4.绘制散点图（查看数值型列之间的关系-fare和age，并返回相关系数）
    # scaplot(X)
    # 4.1 用yellowbrick绘制双变量关系图（散点+直方）
    # yellowplot(X)
    # 5.seaborn绘制箱型图和小提琴图（查看特征分布情况）
    # sbplot2(X,y)
    # sbplot3(X,y)
    # 6.yellowbrick绘制特征之间的pearson相关系数
    # yellowplot2(X,y)
    # 6.1 seaborn绘制特征关系的热力图（查看特征之间的相关性，如果特征之间存在相关性较强的要进行额外处理）
    # sbplot4(X)
    # 7 删除高度相关的两列中的其中一列（存在高度相关的列不仅不会提供更多信息，还会干扰特征重要性的计算）
    # c_df =correlated_columns(X)
    # print(c_df.style.format({'pearson':'{:.2f}'}))
    # 8 绘制RadViz图（看数据是否能按照特征类别可分-也是观察特征重要性的一种手段）
    # 8.1 用yellowbrick绘制
    # yellowplot3(X,y)
    # 8.2 用pandas绘制
    pdplot(X,y)
    # 9 绘制平行坐标图（感觉没什么用）