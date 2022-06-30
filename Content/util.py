# -*- coding:utf-8 -*-
# @Time : 2022/6/30 8:54 上午
# @Author : Bin Bin Xue
# @File : util
# @Project : Machine Learning Pocket Reference

import pandas as pd
from sklearn import (
    model_selection,
    preprocessing
)


# 主要的数据预处理
def dataProcess():
    # 导入数据
    path = '../Data/Taitanike/train.csv'
    df = pd.read_csv(path)
    # 删除无用特征，设置和提取标签列
    df = df.drop(
        columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    # 使用True模式，保证列属性唯一
    df = pd.get_dummies(df, drop_first=True)
    y = df.Survived
    X = df.drop(columns='Survived')
    # 划分数据集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
    # 处理缺失值
    X_train, X_test = dealNone(X_train, X_test)
    # 数据集正则化
    X_train, X_test = dataRegular(X_train, X_test)
    # 重新得到X和y
    X = pd.concat([X_train,X_test])
    y = pd.concat([y_train,y_test])

    return df, X, y, X_train, y_train, X_test, y_test


# 正则化数据集
def dataRegular(X_train, X_test):
    cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
            'Embarked_S']
    sca = preprocessing.StandardScaler()
    X_train = sca.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = sca.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=cols)

    return X_train, X_test

# 处理指定列的缺失值
def dealNone(X_train, X_test):
    # 选择要填充的列
    num_cols = [
        'age',
    ]

    # 1.拟合值插值
    # imputer = impute.IterativeImputer()
    # imputed = imputer.fit_transform(X_train[num_cols])
    # X_train.loc[:, num_cols] = imputed
    # imputed = imputed.transform(X_test[num_cols])
    # X_test.loc[:, num_cols] = imputed

    # 2.中位数插值
    meds = X_train.median()
    X_train = X_train.fillna(meds)
    X_test = X_test.fillna(meds)

    return X_train, X_test
