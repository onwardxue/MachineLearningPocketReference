# -*- coding:utf-8 -*-
# @Time : 2022/6/20 10:18 下午
# @Author : Bin Bin Xue
# @File : Chart2
# @Project : Machine Learning Pocket Reference

# 完整的数据分类任务工作流程（使用泰坦尼克数据集）

# 导入所需要用到的库(pandas-整理数据，scikit-learn机器学习库，Yellowbrick可视化工具库,numpy
import numpy as np
import matplotlib.pyplot as plt
# import janitor as jn
import pandas as pd
# import pandas_profiling
from sklearn import (
    ensemble,
    preprocessing,
    tree,
    model_selection
)
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score
)

from yellowbrick.classifier import (
    ConfusionMatrix,
    ROCAUC,
)
from yellowbrick.model_selection import (
    LearningCurve,
)
from sklearn.experimental import (
    enable_iterative_imputer,
)
from sklearn import impute

# 导入不同的模型库
# 准备交叉验证
from sklearn import model_selection
# 1.基础模型
from sklearn.dummy import DummyClassifier
# 2.线性回归
from sklearn.linear_model import (LogisticRegression, )
# 3.决策树
from sklearn.tree import DecisionTreeClassifier
# 4.K近邻
from sklearn.neighbors import KNeighborsClassifier
# 5.朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
# 6.支持向量机
from sklearn.svm import SVC
# 7.随机森林
from sklearn.ensemble import RandomForestClassifier
# 8.xgboost
import xgboost
# 9.用stacking整合不同分类器
from mlxtend.classifier import StackingClassifier

# 模型保存和加载库
import pickle


# 研究问题：根据乘客的个人特征和旅行相关特征，预测其是否能在泰坦尼克事故中存活下来（二分类问题-生、死）

# 获取数据 http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt 已经失效了，只能使用kaggle的数据集了
def getData():
    path = '../Data/Taitanike/train.csv'
    df = pd.read_csv(path)
    return df


# 清洗数据
def cleanData(df):
    # 显示特征格式
    # print(df.dtypes)
    # 显示行数和列数
    # print(df.shape)
    # 显示整体信息
    # print(df.describe().iloc[:,:2])
    # 显示各列缺失值数量
    # print(df.isnull().sum())
    # 检查哪些行缺数据
    # mask = df.isnull().any(axis=1)
    # print(df[mask)
    # 查看某个类型变量的选项（dropna=false表示将空值也带上）
    print(df.Sex.value_counts(dropna=False))


# 创建特征(删除无用列，根据现有列添加新列）
def createCh(df):
    # 显示列信息
    # name = df.Name
    # print(name.head)
    # 删除行或列（删除分个别的类-名字、和泄漏乘客生存情况的特征、删除高度正负相关的属性-两个性别属性中的一个）
    df = df.drop(
        columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    # 使用True模式，保证列属性唯一
    df = pd.get_dummies(df, drop_first=True)
    # print(df.columns)
    # 将标签列y和非标签列X划分开（或使用pyjanitor库的get_features_targets）
    y = df.Survived
    X = df.drop(columns='Survived')
    return df, X, y


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


def dataRegular(X_trainn, X_test):
    cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
            'Embarked_S']
    sca = preprocessing.StandardScaler()
    X_train = sca.fit_transform(X_trainn)
    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = sca.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=cols)

    return X_train, X_test


def dummyModel(X_train, X_test, y_train, y_test):
    bm = DummyClassifier()
    # 训练集训练模型
    bm.fit(X_train, y_train)
    # 测试模型效果
    auc = bm.score(X_test, y_test)
    # 输出accuracy值（标签重合度除真实标签总数）
    print(auc)
    # 输出混淆矩阵中得到的精确度值（正样率）？问题：不知道为什么这里预测的标签都是0
    # print(bm.predict(X_test))
    # pre = precision_score(y_test,bm.predict(X_test))
    # print(pre)


# 使用多种模型进行预测
def multiModel(X_train, X_test, y_train, y_test):
    # 合并数据集
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    # 逐个模型进行训练
    for model in [
        DummyClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        GaussianNB,
        SVC,
        RandomForestClassifier,
        xgboost.XGBClassifier,
    ]:
        cls = model()
        # 设置交叉检测次数和随机种子
        kfold = model_selection.KFold(
            n_splits=10, shuffle=True, random_state=42
        )
        # 使用指定的模型进行交叉检验
        s = model_selection.cross_val_score(cls, X, y, scoring='roc_auc', cv=kfold)

        # 输出AUC（取10次的均值）和标准差
        print(
            f"{model.__name__:30}     AUC:  "
            f"{s.mean():.3f} STD:  {s.std():.2f}"
        )


# 使用stack方法集成不同模型进行预测
def stackModel(X_train, X_test, y_train, y_test):
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    clfs = [
        x()
        for x in [
            LogisticRegression,
            DecisionTreeClassifier,
            KNeighborsClassifier,
            GaussianNB,
            SVC,
            RandomForestClassifier,
        ]
    ]

    # 模型集成
    stack = StackingClassifier(
        classifiers=clfs,
        meta_classifier=LogisticRegression(),
    )

    # 使用集成模型对数据集进行十折交叉检验
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
    s = model_selection.cross_val_score(stack, X, y, scoring='roc_auc', cv=kfold)
    print(f"{stack.__class__.__name__}"
          f"AUC: {s.mean():.3f}  STD:  {s.std():.2f}"
          )


# 模型评估，重要特征
def modelAccess(X_train, X_test, y_train, y_test):
    # 初始化随机森林模型
    rf = ensemble.RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    # 模型训练
    rf.fit(X_train, y_train)

    # 模型评估
    # 返回各树的准确率
    acu = rf.score(X_test, y_test)
    print(acu)
    # 返回精确值
    pre = precision_score(y_test, rf.predict(X_test))
    print(pre)
    # 查看模型中的特征重要性(默认使用的是"gini"）
    for col, val in sorted(
            zip(
                X_train.columns,
                rf.feature_importances_,
            ),
            key=lambda x: x[1],
            reverse=True,
    )[:5]:
        print(f"{col:10}{val:10.3f}")


# 绘制混淆矩阵
def plotMatrix(rf5, X_test, y_test):
    # 取得预测标签
    y_pred = rf5.predict(X_test)
    # 用真实标签和预测标签得到混淆矩阵
    cfs = confusion_matrix(y_test, y_pred)
    print(cfs)

    mapping = {0: 'died', 1: 'survied'}
    fig, ax = plt.subplots(figsize=(6, 6))
    cm_viz = ConfusionMatrix(
        rf5,
        classes=['died', 'survied'],
        label_encoder=mapping,
    )
    cm_viz.score(X_test, y_test)
    cm_viz.poof()
    fig.savefig(
        "images/mlpr_010.png",
        dpi=300,
        bbox_inches='tight',
    )


# 绘制ROC曲线（越凸效果越好）
def plotRoc(rf5, X_test, y_test):
    y_pred = rf5.predict(X_test)
    roc = roc_auc_score(y_test, y_pred)
    print(roc)

    fig, ax = plt.subplots(figsize=(6, 6))
    roc_viz = ROCAUC(rf5)
    # ？这句显示有问题，不知道哪里出错了，绘制不了
    # roc_viz.score(X_test, y_test)
    roc_viz.poof()
    fig.savefig('images/mlpr_020.png')


# 绘制学习曲线(了解训练数据的量是否足够训练模型）
def plotStu(rf5, X, y):
    fig, ax = plt.subplots(figsize=(6, 4))
    cv = model_selection.StratifiedKFold(12)
    sizes = np.linspace(0.3, 1.0, 10)
    lc_viz = LearningCurve(
        rf5,
        cv=cv,
        train_sizes=sizes,
        scoring='f1_weighted',
        n_jobs=4,
        ax=ax,
    )
    lc_viz.fit(X, y)
    lc_viz.poof()
    fig.savefig('images/mlpr_030.png')


#  模型优化：网格搜索法调整模型训练参数
def modelOpt(X_train, X_test, y_train, y_test):
    # 初始化随机森林
    rf4 = ensemble.RandomForestClassifier()
    # 设置要调整的参数范围
    params = {
        "n_estimators": [15, 200],
        "min_samples_leaf": [1, 0.1],
        "random_state": [42],
    }
    # 使用网格搜索法设置参数
    cv = model_selection.GridSearchCV(
        rf4, params, n_jobs=-1).fit(X_train, y_train)

    # 输出最佳参数值
    print(cv.best_params_)

    # 用最佳参数初始化模型并进行训练 ?不知道为什么这里要在前面加两个*号
    rf5 = ensemble.RandomForestClassifier(
        **{
            "min_samples_leaf": 0.1,
            "n_estimators": 200,
            "random_state": 42
        }
    )
    rf5.fit(X_train, y_train)
    auc = rf5.score(X_test, y_test)
    # print(auc)

    # 绘制混淆矩阵
    # plotMatrix(rf5,X_test,y_test)

    # 绘制ROC曲线
    # plotRoc(rf5, X_test, y_test)

    # 绘制学习曲线(这里的结果表示增加数据也起不到更好的提升效果）
    # plotStu(rf5, X_train, y_train)

    return rf5


# 模型持久化（保存训练好的模型，需要的时候再使用）
def saveModel(rf5, X_test, y_test):
    # 保存模型
    pic = pickle.dumps(rf5)
    # 加载模型
    rf6 = pickle.loads(pic)
    # 用加载的模型预测当前的数据
    y_pred = rf6.predict(X_test)
    roc = roc_auc_score(y_test, y_pred)


if __name__ == '__main__':
    # 1.获取数据
    df = getData()
    # 2.清洗数据
    # cleanData(df)
    # 3.特征筛选：删除和创建特征
    df, X, y = createCh(df)
    # 4.数据集划分
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
    # 5.缺失值处理：对不含标签的数据列插值（拟合插值或中位数插值）
    # 查看含缺失值的列
    # print(X_train.isnull().any(axis=0))
    X_train, X_test = dealNone(X_train, X_test)
    # 6.数据标准化处理（统一取值范围-均值为0，标准差为1）
    #  查看现有的列名
    print(X_train.columns)
    # 标准化
    X_train, X_test = dataRegular(X_train, X_test)
    # 7.使用最简单的dummy模型进行预测
    # dummyModel(X_train, X_test, y_train, y_test)
    # 8.使用8个不同模型进行预测（RF>xgb>KNN>LR>SVM>GNB>DT>Dummy>
    # multiModel(X_train,X_test,y_train,y_test)
    # 9.使用堆叠（Stacking）模型（追求性能，放弃可解释性,标准差会变小）
    # stackModel(X_train, X_test, y_train, y_test)
    # 10.模型评估，特征重要性排序 - 用随机森林举例
    # modelAccess(X_train, X_test, y_train, y_test)
    # 11.模型优化，使用网格搜素法调整模型的训练参数,绘制混淆矩阵、ROC曲线、学习曲线
    rf5 = modelOpt(X_train, X_test, y_train, y_test)
    # 12.部署模型（保存模型）
    saveModel(rf5,X_test,y_test)

