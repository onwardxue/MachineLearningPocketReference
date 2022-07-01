# -*- coding:utf-8 -*-
# @Time : 2022/6/29 8:44 下午
# @Author : Bin Bin Xue
# @File : Chart6_DataExp&Plot
# @Project : Machine Learning Pocket Reference
# 这里介绍的是一些处理类不平衡数据（如标签为1的数据量远少于0的数据量）的方法
# 过采样：较少的类别的数据量加量
# 欠采样：较多的类别的数据流减少
# 过采样+欠采样
import pandas as pd
# skl带有的采样包
from sklearn.utils import resample
# 用imbalanced-learn库实现不放回的随机采样(还可以使用SMOTE和ADASYN使少数类生成新的样本）
from imblearn.over_sampling import(
    RandomOverSampler,
)

path = '../Data/Taitanike/train.csv'
df = pd.read_csv(path)

if __name__ == '__main__':
    # 1.考虑采用不同的度量标准（除了准确率之外的）
    # 2.树模型和集成方法（树模型、随机森林、xgboost对小类别有更好的效果）
    # 3.使用模型中的一些惩罚参数（如skl模型的参数class_weight，balanced；xgb的参数max_delta_step、scale_pos_weight、eval_metric；knn的weights和distance
    # 4.1 采样策略 - 对小众类别过采样
    # mask = df.Survived == 1
    # surv_df = df[mask]
    # death_df = df[~mask]
    # 对生存者用随机过采样（有放回）
    # df_upsample = resample(
    #     surv_df,
    #     replace=True,
    #     # 采样后的数据量与死亡数量相同
    #     n_samples=len(death_df),
    #     random_state=42,
    # )
    # df2 = pd.concat([death_df,df_upsample])
    # print(df2.Survived.value_counts())
    # 对生存者用随机过采样（无放回）
    # ros = RandomOverSampler(random_state=42)
    # X_ros,y_ros = ros.fit_sample(X,y)
    # pd.Series(y_ros).value_counts()

    # 4.2 采样策略 - 对大众类别欠采样（用的是有放回，不能用无放回）
    # 4.2.1 使用skl库实现
    mask = df.Survived == 1
    surv_df = df[mask]
    death_df = df[~mask]
    df_downsample = resample(
        death_df,
        replace=False,
        n_samples=len(surv_df),
        random_state = 42,
    )
    df3 = pd.concat([surv_df,df_downsample])
    print(df3.Survived.value_counts())
    # 4.2.2 使用imblanced-learn库实现
    # 十二类方法实现
    # ClusterCentroids类用K-means算法生成质心点
    # RandomUnderSample类随机取样
    # NearMiss类用最近邻实现下采样
    # TomekLink类通过删除彼此靠的很近的样例实现
    # EditedNearestNeighbours类删除相邻样例不属于同一类别或全部属于同一类别的样例
    # RepeatedNearsetNeighbours类对上面这个类的迭代
    # allknn类在下采样迭代过程中增加近邻的数量
    # CondensedNearestNeighbour类从下采样类别挑选样例
    # OneSidedSelection类删除噪声样例
    # NeighbourhoodCleaningRule类使用Edit..的结果，并用knn处理
    # InstanceHardnessThreshold类训练一个模型，删除概率较小的样例
    #先上采样，后再下采样
    # 使用imblanced-learn库的SMOTEENN和SMOTETomek类实现
