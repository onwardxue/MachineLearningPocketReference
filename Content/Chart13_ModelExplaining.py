# -*- coding:utf-8 -*-
# @Time : 2022/6/29 8:44 下午
# @Author : Bin Bin Xue
# @File : Chart6_DataExp&Plot
# @Project : Machine Learning Pocket Reference
# 这里介绍几种模型可解释性的指标

from lime import lime_tabular
from treeinterpreter import(
    treeinterpreter as ti
)
# 问题？安装不了这个包
# import pdpbox

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import shap
import util

# 导入预处理好的Taitannike数据
df, X, y, X_train, y_train, X_test, y_test = util.dataProcess()

# lime 模型局部解释
def limePlot(dt):
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X.columns,
        class_names=['died','survived']
    )
    exp = explainer.explain_instance(
        X_train.iloc[-1].values, dt.predict_proba
    )
    # 绘图
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig('images/mlpr_1301.png')
    # 从图中可以看出性别特征对决策起到了重要作用（若不分性别，预测的死亡率为48%；若筛选为男性，则预测死亡率为80%）
    data = X_train.iloc[-2].values.copy()
    dt.predict_proba(
        [data]
    )
    data[5]=1
    dt.predict_proba([data])

# 树模型的可解释
def treeExplain(dt):
    instances = X.iloc[:2]
    prediction,bias,contribs = ti.predict(
        dt,instances
    )
    i=0
    print('Instance',i)
    print('Prediction',prediction[i])
    print('Bias (trainset mean)', bias[i])
    print('Feature contributions:')
    for c , feature in zip(
        contribs[i],instances.columns
    ):
        print('{}{}'.format(feature,c))

# 部分依赖图
# def pdboxPlot():
    # rf5 = RandomForestClassifier(
    #     **{
    #         'max_features':'auto',
    #         'min_samples_leaf':'0.1',
    #         'n_estimators':'200',
    #         'random_state':42,
    #     }
    # )
    # rf5.fit(X_train,y_train)
    # feat_name='Age'
    # p = pdp.pdp_iso
    # # 可视化两个特征的交互作用
    # features = ['Fare','Sex_male']
    # p = pdp_interact(
    #     rf5
    # )

# 替代模型
def replaceModel():
    sv = svm.SVC()
    sv.fit(X_train,y_train)
    sur_dt = DecisionTreeClassifier()
    sur_dt.fit(X_test,sv.predict(X_test))
    # 输出特征重要性
    for col, val in sorted(
        zip(
            X_test.columns,
            sur_dt.feature_importances_,
        ),
        key = lambda x: x[1],
        reverse=True,
    )[:7]:
        print(f'{col:10}{val:10.3f}')

# shap值
def shapTest():
    rf5 = RandomForestClassifier(
        **{
            'min_samples_leaf':0.1,
            'n_estimators':200,
            'random_state':42,
        }
    )
    rf5.fit(X_train,y_train)
    s = shap.TreeExplainer(rf5)
    shap_vals = s.shap_values(X_test)
    target_idx = 1
    shap.force_plot(
        s.expected_value[target_idx],
        shap_vals[target_idx][20,:],
        feature_names=X_test.columns,
    )


if __name__ == '__main__':
    dt = DecisionTreeClassifier(random_state=42, max_depth=3)
    dt.fit(X_train, y_train)
    # 1.回归系数
    # 截距和回归系数解释了模型的预测结果以及特征对结果的影响。回归系数为正表示正相关。
    # 2.特征重要性
    # skl的树模型都带有一个.feature_importances_属性，可用于显示特征重要性
    # 3.模型局部解释（LIME包) - 显示每个特征对决策正负的影响大小
    # limePlot(dt)
    # 4.解释树模型（包括决策树、随机森林、极限树）
    # 用treeinterpreter包解释，会给出每个特征给每个类的贡献列表(本例中显示年龄和性别的贡献最大）
    # treeExplain(dt)
    # 5.部分依赖图（查看特征值的变化是如何影响结果的）
    # 例子：用pdbox包绘制年龄对乘客死亡的影响
    # pdboxPlot()
    # 6.替代模型（用可解释的模型替代不可解释的模型）
    # svm和神经网络一般不可解释
    # 例子：用决策树模型替代支持向量机模型进行解释（先用训练集和标签训练svm模型，用测试集数据和svm预测测试集数据得到的标签训练决策树模型）
    # replaceModel()
    # 7.shapley值（SHAP包，能生成shap值，利用特征shaply值的可加性解释模型的预测结果，能可视化任意模型的特征贡献）
    # 例子：
    # shapTest()
    print('sss')