# 鸢尾花数据集分类于k-means与pyeachrs
import os
import pandas as pd
# 引入pyecharts相关包
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType
# 引入Kmeans API
from sklearn.cluster import KMeans
from sklearn import preprocessing


def getCalyxLAW(data):  # getCalylenghAndWidth 获取花萼的长宽
    return data[["Sepal.Length", "Sepal.Width"]].values


def analSepal(sepdata):  # 基于花萼进行分析
    kmeansTraindata = pd.DataFrame(sepdata)  # 拷贝一份作为训练集
    kModel = KMeans(n_clusters=3)  # 实例化对象
    min_max_scaler = preprocessing.MinMaxScaler()  # 设置定标器
    kmeansTraindata = min_max_scaler.fit_transform(kmeansTraindata)  # 数据归一化
    kModel.fit(kmeansTraindata)  # 进行训练
    cluster = kModel.predict(kmeansTraindata)  # 生成聚类
    sepdata["cluster"] = cluster  # 添加到原数据中
    print(sepdata)
    eachClass = sepdata.groupby("cluster").apply(getCalyxLAW)  # 对数据进行分类
    print(eachClass)


if __name__ == "__main__":
    a = pd.read_csv(os.getcwd() + "\\iris.csv")
    # 基于当前目录读取文件
    print(a.columns)

    # 检查显示行数据
    analSepal(a[["id", "Sepal.Length", "Sepal.Width"]])
