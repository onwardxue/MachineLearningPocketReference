# -*- coding:utf-8 -*-
# @Time : 2022/6/29 8:44 下午
# @Author : Bin Bin Xue
# @File : Chart6_DataExp&Plot
# @Project : Machine Learning Pocket Reference
# 这里介绍的是一些常用的数据预处理的方法

import pandas as pd
# skl的预处理包
from sklearn import (
    preprocessing,
)

# skl转换器，能将类别特征转为数值特征
# import category_encoders as ce


path = '../Data/Taitanike/train.csv'
df = pd.read_csv(path)

# 本章要使用的数据集
X = pd.DataFrame(
    {
        'a': range(5),
        'b': [-100, -50, 0, 200, 1000]
    }
)
X_cat = pd.DataFrame(
    {
        'name': ['George', 'paul'],
        'inst': ['Bass', 'Guitar'],
    }
)


# 得到某类型特征的所有特征值
def get_title(df):
    return df.Name.str.extract(
        '([A-Za-z]+)\.', expand=False
    )


if __name__ == '__main__':
    print(X)
    # 1.数值特征-数据标准化（使数据的每列均值为0，标准差为1）-一些算法，如支持向量机，在标准化后的数据上表现更好
    # std = preprocessing.StandardScaler()
    # X1 =std.fit_transform(X)
    # print(X1)
    # 1.1显示标准化后的统计值
    # print(std.scale_)
    # print(std.mean_)
    # print(std.var_)
    # 2.数值特征-调整取值范围（将数据转换到0和1之间，但若数据包含异常值最好不要使用）
    # mms = preprocessing.MinMaxScaler()
    # mms.fit(X)
    # print(mms.transform(X))
    # 3.类别特征-one-hot编码（pandas的get_dummies可实现该功能）
    # X_cat1 =pd.get_dummies(X_cat,drop_first=True)
    # print(X_cat1)
    # 3.1 标签编码（比one-hot节省空间，但只能处理一列）
    # lab = preprocessing.LabelEncoder()
    # lab.fit_transform(X_cat)
    # print(X_cat1)
    # 3.2 频数编码
    # 4.统计某个类别特征不同值的频数
    # count = df.Name.str.extract(
    #     '([A-Za-z]+)\.', expand=False
    # ).value_counts()
    # print(count)
    # 5.类别特征的其他编码
    # 5.1 哈希编码器 - 提前不知道有多少类别（该方法适用于在线学习）
    # he = ce.HashingeEncoder(verbose=1)
    # he.fit_transform(X_cat)
    # 5.2 序数编码器 - 用于包含有序信息的类别型数据转换
    # size_df = pd.DataFrame(
    #     {
    #         'name':['Fred','John','Matt'],
    #         'size':['small','med','xxl'],
    #     }
    # )
    # ore = ce.Ordinale(
    #     mapping=[
    #         {
    #             'col':'size',
    #             'mapping':{
    #                 'small':1,
    #                 'med':2,
    #                 'lg':3,
    #             },
    #         }
    #     ]
    # )
    # ore.fit_transform(size_df)
    # 5.3 贝叶斯编码器 - 适用于唯一值较多的特征
    # 6. 联合前面两个步骤-找出类别特征的所有特征值，将其转化为数值型。
    # 用泰坦尼克号的Name属性做演示
    # te = ce.TargetEncoder(cols='Title')
    # te.fit_transform(
    #     df.assign(Title=get_title),df.Survived
    # )['Title'].head()
    # 7.日期特征的处理方法（用不上）
    # 8.特征工程（利用现有特征组合生成新的特征，添加到数据集中）
    # pandas的groupby生成新特征（聚合特征）和merge添加到数据集中
    # 这里用各船舱的最大年龄、平均年龄等聚合成一个新特征     ?问题，这里也无法执行，不知道错在哪里
    # agg = (
    #     df.groupby('cabin')
    #     .agg('min.max,mean,sum'.split(','))
    #     .reset_index()
    # )
    #
    # agg_df = df.merge(agg,on='cabin')
    # print(agg_df.columns)