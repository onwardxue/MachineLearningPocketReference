# -*- coding:utf-8 -*-
# @Time : 2022/6/29 8:44 下午
# @Author : Bin Bin Xue
# @File : Chart6_DataExp&Plot
# @Project : Machine Learning Pocket Reference
# 介绍降维技术，包括PCA,UMAP,t-SNE,PHATE等降维方法

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 自定义的工具包
import util

# 导入预处理好的Taitannike数据
df, X, y, X_train, y_train, X_test, y_test = util.dataProcess()


# 根据成分的方差解释率绘制碎石图
def screePlot(ratio):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ratio)
    ax.set(
        xlabel='Component',
        ylabel='Percent of Explained variance',
        title='Scree plot',
        ylim=(0, 1),
    )
    fig.savefig(
        'images/mlpr_1701.png',
        dpi=300,
        bbox_inches='tight',
    )


# 绘制方差累计图
def variblePlot(ratio):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        np.cumsum(ratio)
    )
    ax.set(
        xlabel='Component',
        ylabel='Percent of Explained variance',
        title='cumlative variance',
        ylim=(0, 1)
    )
    fig.savefig('images/mlpr_1702.png', dpi=300)


# 查看各特征对主成分的影响
def ralatePlot(pca):
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.imshow(
        pca.components_.T,
        cmap='Spectral',
        vmin=-1,
        vmax=1,
    )
    plt.yticks(range(len(X.columns)), X.columns)
    plt.xticks(range(8), range(1, 9))
    plt.xlabel('principal component')
    plt.ylabel('Contribution')
    plt.title(
        'Contribution of Features to Components'
    )
    plt.colorbar()
    plt.savefig('images/mlpr_1703.png', dpi=300)

# 绘制柱状图
def barPlot(pca):
    fig,ax=plt.subplots(figsize=(8,4))
    pd.DataFrame(
        pca.components_,columns=X.columns
    ).plot(kind='bar',ax=ax).legend(
        bbox_to_anchor=(1,1)
    )
    fig.savefig('images/mlpr_1704.png',dpi=300)



# 筛选出较为重要的特征
def lmfeaPlot(pca):
    comps = pd.DataFrame(
        pca.components_, columns = X.columns
    )
    min_val=0.5
    num_components=2
    pca_cols=set()
    for i in range(num_components):
        parts = comps.iloc[i][
            comps.iloc[i].abs()>min_val
        ]
        pca_cols.update(set(parts.index))
    print(pca_cols)


# pca降维去噪
def pcaTest():
    pca = PCA(random_state=42)
    X_pca = pca.fit_transform(
        StandardScaler().fit_transform(X)
    )
    # 输出每个成分的方差解释率
    ratio = pca.explained_variance_ratio_
    print(ratio)
    # 输出主成分
    print(pca.components_[0])
    # 将方差解释率累计情况绘制成碎石图（显示主成分所含信息量，用肘部方法查看它的拐点，从而决定使用多少个主成分）
    # 由图中可知，仅保留3个特征就能保留90%的有效信息（方差）
    # screePlot(ratio)
    # 将方差解释率绘制成方差累计图
    # variblePlot(ratio)
    # 绘制特征与主成分的关系图，查看每个特征对主成分的影响
    # ralatePlot(pca)
    # 绘制特征与主成分之间关系的柱状图
    # barPlot(pca)
    # 如果有很多特征，我们想要限制特征数，用代码找出前两个主成分中，权重绝对值至少为0.5的特征
    lmfeaPlot(pca)
    #


if __name__ == '__main__':
    # PCA(降维技术，可以用来降维，也可以主要作为预处理步骤，以过滤噪声数据中的随机噪声，处理掉线性数据）
    # skl中的pca是个转换器，可用.fit训练它使它学会获取主成分，然后调用.transform将矩阵转为主成分矩阵
    pcaTest()
