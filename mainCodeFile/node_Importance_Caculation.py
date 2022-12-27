import csv
from tqdm import tqdm
from mainCodeFile.AGNES_Cluster import readfile
import pandas as pd
from otherCodeFile.HanlpProj import TextRank_sim


def node_importance_cacultion(embeddingFile='',entityXlsxFile=r'',importance_SaveFile=r''):

    ##获取嵌入数据
    rownames, colnames, data=readfile(embeddingFile)

    ##获取实体节点名称

    entityTable=pd.read_excel(entityXlsxFile)
    labels=entityTable.columns.tolist()
    labels.remove('first虚节点')
    all_word_Frequencys=[]
    for label in labels:
        print('开始进行 {} 标签的实体重要性得分计算'.format(label))
        entitylist=entityTable[label].tolist()

        if len(entitylist)>0 and entitylist[0]==entitylist[0]:
            singleLabel_word_Frequencys = []
            for entity in entitylist:
                if entity ==entity:
                    sims=[]
                    for rowname in rownames :
                        sim=TextRank_sim(entity,rowname)
                        if sim >=0.5:
                            sims.append([entity,rowname,sim])
                    # sims=sorted(sims,key= lambda x:x[::-1],reverse=True)
                    # print(sims)
                    singleLabel_word_Frequencys.append([entity,len(sims),0.0])


            ##对词频进行标准化处理，得到相对重要性权重
            Frequencys=[i[1] for i in singleLabel_word_Frequencys]
            sum_Fre=sum(Frequencys)
            if sum_Fre!=0:
                for idx in range(len(singleLabel_word_Frequencys)):
                    fre=singleLabel_word_Frequencys[idx][1]
                    singleLabel_word_Frequencys[idx][2]=round(fre/sum_Fre,3)
                singleLabel_word_Frequencys = sorted(singleLabel_word_Frequencys, key=lambda x: x[::-1], reverse=True)
                print(singleLabel_word_Frequencys,'\n')
            all_word_Frequencys+=singleLabel_word_Frequencys

    #存储重要性节点重要性得分
    with open(importance_SaveFile, 'w', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        # 设置第一行标题头
        writer.writerow(['节点名称','词频','重要性'])
        # 将数据写入
        writer.writerows(all_word_Frequencys)

if __name__ == '__main__':
    # #预算员节点(边)权重计算
    # node_importance_cacultion(embeddingFile=r'../data/midFile/embedding.csv',
    #                           entityXlsxFile=r'C:\Users\glodon\Desktop\知识图谱搭建\预算员图谱构建\预算员实体联系表\预算员相关实体表0916.xlsx',
    #                           importance_SaveFile=r'../data/outFile/importance_File.csv')

    # #造价员节点(边)权重计算
    node_importance_cacultion(embeddingFile=r'C:\Users\glodon\Desktop\知识图谱搭建\预算员图谱构建\embedding.csv',
                              entityXlsxFile=r'C:\Users\glodon\Desktop\知识图谱搭建\简易版图谱\预算员简易图谱\预算员相关实体表1008.xlsx',
                              importance_SaveFile=r'C:\Users\glodon\Desktop\知识图谱搭建\简易版图谱\预算员简易图谱\importance_File.csv')