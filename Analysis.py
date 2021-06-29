import os
import pandas as pd
# 引入pyecharts相关包
from pyecharts import options as opts
from pyecharts.charts import Scatter
from pyecharts.globals import ThemeType
# 引入Kmeans API
from sklearn.cluster import KMeans
from sklearn import preprocessing


def getCalyLAW(data):  # getCalyLenghAndWidth 获取花萼的长宽
    return data[["id", "Sepal.Length", "Sepal.Width"]].values


def getPetalLAW(data):  # getpetalLenghAndWidth 获取花萼的长宽
    return data[["id", "Petal.Length", "Petal.Width"]].values


def scatterImgCreater(scatterdata, title):
    sepalScatter = Scatter(init_opts=opts.InitOpts(
        width="1600px", height="1000px"))  # 初始化散点图类
    clusterTotal = len(scatterdata)  # 聚类的总个数
    clusterNum = 0
    for eachcluster in scatterdata:
        eachcluster = sorted(eachcluster, key=lambda x: (x[1], x[2]))
        clusterNum += 1
        transponcluster = list(zip(*eachcluster))  # 转置原数据集以提取长度与宽度
        iddata = list(transponcluster[0])  # 提取id数据 或许不使用
        lenthdata = list(transponcluster[1])  # 提取长度数据
        widthdata = list(transponcluster[2])  # 提取宽度数据
        sepalScatter.add_xaxis(
            xaxis_data=lenthdata,
        )
        sepalScatter.add_yaxis(
            series_name="",
            y_axis=widthdata,
            symbol_size=20,
            label_opts=opts.LabelOpts(is_show=False),
        )
        sepalScatter.set_global_opts(
            yaxis_opts=opts.AxisOpts(
                min_='dataMin'
            ),
            toolbox_opts=opts.ToolboxOpts(),
        )
    sepalScatter.render(title+"分类图.html")


def analSepal(sepdata):  # 基于花萼进行分析
    kmeansTraindata = pd.DataFrame(sepdata)  # 拷贝一份作为训练集
    kModel = KMeans(n_clusters=3)  # 实例化对象
    min_max_scaler = preprocessing.MinMaxScaler()  # 设置定标器
    kmeansTraindata = min_max_scaler.fit_transform(kmeansTraindata)  # 数据归一化
    kModel.fit(kmeansTraindata)  # 进行训练
    cluster = kModel.predict(kmeansTraindata)  # 生成聚类
    sepdata["cluster"] = cluster  # 添加到原数据中
    eachClass = sepdata.groupby("cluster").apply(getCalyLAW)  # 对数据进行分类
    title = "基于花萼进行分析"  # 图片标题
    scatterImgCreater(eachClass, title)


def analPetal(petdata):  # 基于花瓣进行分析
    kmeansTraindata = pd.DataFrame(petdata)  # 拷贝一份作为训练集
    kModel = KMeans(n_clusters=3)  # 实例化对象
    min_max_scaler = preprocessing.MinMaxScaler()  # 设置定标器
    kmeansTraindata = min_max_scaler.fit_transform(kmeansTraindata)  # 数据归一化
    kModel.fit(kmeansTraindata)  # 进行训练
    cluster = kModel.predict(kmeansTraindata)  # 生成聚类
    petdata["cluster"] = cluster  # 添加到原数据中
    eachClass = petdata.groupby("cluster").apply(getPetalLAW)  # 对数据进行分类
    title = "基于花瓣进行分析"  # 图片标题
    scatterImgCreater(eachClass, title)
    # print(petdata)


if __name__ == "__main__":
    a = pd.read_csv(os.getcwd() + "\\iris.csv")
    # 基于当前目录读取文件
    print(a.columns)
    # 检查显示行数据
    analSepal(a[["id", "Sepal.Length", "Sepal.Width"]])
    analPetal(a[["id", "Petal.Length", "Petal.Width"]])
