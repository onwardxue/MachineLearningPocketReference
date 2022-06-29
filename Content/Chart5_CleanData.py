# -*- coding:utf-8 -*-
# @Time : 2022/6/27 4:44 下午
# @Author : Bin Bin Xue
# @File : Chart4_MissingHandler
# @Project : Machine Learning Pocket Reference

import pandas as pd
import matplotlib.pyplot as plt

# 专业表格数据清洗工具pyjanitor
import janitor as jn


# 例1用于创建一个数据集
def example_1():
    # 创建一个Dataframe数据
    Xbad = pd.DataFrame(
        {
            "A": [1, None, 3],
            " sales number": [20.0, 30.0, None],
        }
    )
    return Xbad

# 用jn的方法将数据列名进行转换
# 使用jb的方法将列名转换的更友好（列名变成小写，空格变成下划线)
# 但 jn无法去掉列名前后的下划线
def jnTest(Xbad):
    Xbads = jn.clean_names(Xbad)
    print(Xbads)

# 用pandas的方法转换数据集列名
def clean_col(name):
    return(
        name.strip().lower().replace(" ","_")
    )


if __name__ == '__main__':
    # 创建一个测试版数据集
    Xbad = example_1()
    # 1. 列名（特征名）转换
    # 1.1 使用jn的方法转换列名
    # jnTest(Xbad)
    # 1.2 使用pd的方法转换列名 ？这里实现不了书上的
    # Xbad.rename(columns=clean_col)
    #2.替换缺失值
    # 2.1 用dataframe的fillna方法填充缺失值
    # Xbads =Xbad.fillna(10)
    # print(Xbads)
    # 2.2 用pyjanitor的fill_empty方法填充缺失值
    # jn.fill_empty(
    #     Xbad,
    #     columns=["A", " sales numbers "],
    #     value=10,
    # )
    # 检查数据中是否还有缺失值(True为缺失还存在缺失值）
    print(Xbad.isna().any().any())