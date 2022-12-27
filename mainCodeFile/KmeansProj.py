import torch
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class Timer:

    def __new__(cls, fun, *args, **kwargs):
        import time
        start = time.time()
        fun(*args, **kwargs)
        cost = time.time() - start
        return cost * 1e3  # ms


def get_result(train_x, sampleNams, labels):
    n_clusters = np.unique(labels).shape[0]
    clu_res = []
    # 分别绘制每个类别的样本
    for i in range(n_clusters):
        samples = train_x[labels == i]
        clu_res.append(sampleNams[labels == i])
    return clu_res


def drawdendrogram(clu_res, jpeg='clusters.jpg'):
    num = sum([i.shape[0] for i in clu_res])
    cls_num = len(clu_res)
    # 定义初始位置
    y = 0
    x = 20

    # 定义其他参数
    cls_space_x = 0
    cls_space_y = 20
    text_space = 20
    ll = 700
    # 高度和宽度
    h = num * 20 + cls_num * cls_space_y + 50
    w = 1500

    # 新建一个白色背景的图片
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    clu_count = 0
    for clu in tqdm(clu_res):
        x += cls_space_x
        y += cls_space_y
        last_pos = [0, 0]
        count = 0
        if clu_count % 2 == 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        clu_count += 1
        for nam in tqdm(clu):
            top = 8
            y += text_space

            cur_pos = [x, y + top]

            if count != 0:
                draw.line((last_pos[0], last_pos[1], cur_pos[0], cur_pos[1]), fill=color)

            last_pos = [x, y + top]

            draw.line((x, y + top, x + ll, y + top), fill=color)

            draw.text((x + ll, y), nam, color, font=ImageFont.truetype('simhei.ttf', 15))
            count += 1
            pass

    # img.save(jpeg, 'JPEG')
    Image._show(img)


if __name__ == '__main__':

    # 读取数据集
    n=10000
    csv_df=pd.read_csv(r'E:\PythonProject\zhilianDataProcess\data\midFile\embedding.csv')
    sampleNams=csv_df['pharseNam'].tolist()[:n]
    sampleNams=np.array(sampleNams)
    cols= csv_df.columns.tolist()
    features=csv_df[cols[1:]].to_numpy()
    features=csv_df.loc[:n-1,cols[1:]].to_numpy()
    x=torch.tensor(features)
    iris_x=x


    # 使用 KMeans 聚类
    clf = KMeans(n_clusters=100)
    print(f'Kmeans: {Timer(clf.fit, iris_x):.0f} ms')
    clu_res=get_result(iris_x,sampleNams, clf.labels_)
    drawdendrogram(clu_res)

