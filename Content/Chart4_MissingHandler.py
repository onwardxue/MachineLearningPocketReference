# -*- coding:utf-8 -*-
# @Time : 2022/6/27 4:44 下午
# @Author : Bin Bin Xue
# @File : Chart4_MissingHandler
# @Project : Machine Learning Pocket Reference

import pandas as pd
import matplotlib.pyplot as plt

# 可视化缺失值库
import missingno as msno

# 插值法库
from sklearn.impute import SimpleImputer


def getData():
    path = '../Data/Taitanike/train.csv'
    df = pd.read_csv(path)
    return df


# 按行统计缺失值个数
def add_indicator(col):
    def wrapper(df):
        return df[col].isna().astype(int)

    return wrapper


if __name__ == '__main__':
    # 1.获取数据
    df = getData()

    # 2. 显示数据缺失情况的多种方式
    # 2.1 获得不同特征的缺失值比例(这个数据集就三个特征有缺失-age,cabin,embarked)
    # pro = df.isnull().mean() * 100
    # print(pro)
    #
    # # 2.2 可视化显示缺失值情况（df.sample为随机取样）
    # ax = msno.matrix(df.sample(500))
    # ax.get_figure().savefig('images/mlpr_040.png')

    # 2.3 使用pd绘制各特征缺失值的百分比
    # fig,ax = plt.subplots(figsize=(6,4))
    # (1-df.isnull().mean()).abs().plot.bar(ax=ax)
    # fig.savefig('images/mlpr_041.png',dpi=300)

    # 2.4 使用msno绘制柱状图
    # ax = msno.bar(df.sample(500))
    # ax.get_figure().savefig("images/mlpr_042.png")

    # 2.5 使用msno绘制热力图，看缺失数据的位置是否有相关性
    # ax = msno.heatmap(df,figsize=(6,6))
    # ax.get_figure().savefig('images/mlpr_043.png')

    # 2.6 用msno绘制创建系统树，看分簇情况
    # ax = msno.dendrogram(df)
    # ax.get_figure().savefig('images/mlpr_044.png')

    # 3.缺失值处理
    # 3.1 删除有缺失值的行或列（该方法慎用）
    # 3.1.1 直接删除有缺失数据的全部行
    # df1 = df.dropna()
    # # 3.1.2 直接删除有缺失数据的全部列
    # df1 = df.dropna(axis=1)
    # # 3.1.3 删除指定列
    # df1 = df.drop(columns='cabin')

    # 3.2 插值法填充（使用slearn下的SimpleImputer能实现均值、中指、最高出现次数值填充缺失值,前两个要求为数值形，后一个要求为字符形）
    # 筛选出数值型的特征
    # num_cols = df.select_dtypes(
    #     include='number'
    # ).columns
    # # 插值（默认使用均值，设置参数strategy='median'或'most_frequent使用中位数和最高频特征值，自定义常数值为'constant',fill_value=-1)
    # im = SimpleImputer()
    # imputed = im.fit_transform(df[num_cols])

    # 3.添加标识列，显示该样本的缺失值个数
    df1 = df.assign(cabin_missing=add_indicator('Cabin'))
    print(df1)
