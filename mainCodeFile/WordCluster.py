from tqdm import tqdm
from PIL import Image
from pyhanlp import *
import csv
import re
from mainCodeFile.AGNES_Cluster import readfile
from mainCodeFile.excelRead import pandas_readXlsx
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time
from mainCodeFile.AGNES_Cluster import pearson
from otherCodeFile.HanlpProj import TextRank_sim
def keywordExtract_TFIDF(xlsxFile=r'C:\Users\glodon\Desktop\jobCollect\zhilian\智联预算员0.xlsx',baseWordType='hanlp'):
    startT=time.time()
    print('\n开始进行分词： ')
    segPharse_SavePath = 'pharse_seg_result.txt'
    seg_base_word_SavePath='baseWord_seg_result.txt'


    ###1.根据it-idf算法获取短语分词

    _,text=pandas_readXlsx(xlsxFile)# 获取职位描述
    # text,_ = pandas_readXlsx(xlsxFile) #获取job名称
    fileTrainRead =text[:]
    # segment pharse with IF-IDF
    fileTrainSeg = []
    for i in range(len(fileTrainRead)):
        if not (fileTrainRead[i]!=fileTrainRead[i]): #去除nan值项
            tfidf = TfidfVectorizer()
            weight = tfidf.fit_transform([fileTrainRead[i]]).toarray()
            word = tfidf.get_feature_names()
            # print(word)
            fileTrainSeg.append(['()'.join(word)])

    # print(len(fileTrainSeg))
    # save the result
    with open(segPharse_SavePath,'wb') as f:
        for i in range(len(fileTrainSeg)):
            f.write(fileTrainSeg[i][0].encode('utf-8'))
            # f.write('\n'.encode('utf-8'))
        f.close()


    ##获取基础分词
    all_base_Words = []
    if baseWordType=='hanlp':
        ###2.1根据hanlp获取基础词的分词
        CoreStopWordDictionary = JClass("com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary")
        for i in tqdm(range(len(fileTrainRead))):
            sentence = fileTrainRead[i]
            if sentence==sentence:
                sentenceSplit=re.split(r'[；。]',sentence)
                for i,line in enumerate(sentenceSplit):
                    res=hanlp_segment(sentence=line,num=3)
                    CoreStopWordDictionary.apply(res)#去除标点、无关词
                    for term in res:
                        all_base_Words.append(term.word)
        all_base_Words = list(set(all_base_Words))
    else:
        ###2.2 根据短语获取字级分词
        allPharse=[]
        for line in fileTrainSeg:
            allPharse+=line[0].split('()')

        for pharse in allPharse:
            all_base_Words+=list(pharse)

        all_base_Words=list(set(all_base_Words))


    with open(seg_base_word_SavePath,'wb') as f:
        seg_base_word_res='()'.join(all_base_Words)
        f.write(seg_base_word_res.encode('utf-8'))
        f.close()


    endT = time.time()
    print('分词时间消耗： {}s'.format(endT-startT))

def hanlp_segment(sentence='',num=3):
    # sentence="大专及以上学历，工程造价专业1年及以上工作经验；2、熟练应用广联达等造价软件，以及office、autocad等相关的办公软件；"

    # #基础分词
    segmentRes=HanLP.segment(sentence)
    #
    # #关键词提取
    # print(HanLP.extractKeyword(sentence, 2))

    #短语提取
    # segmentRes=HanLP.extractPhrase(sentence,num)


    # #基于感知机进行分词
    # PerceptronLexicalAnalyzer=JClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
    # analyzer=PerceptronLexicalAnalyzer()
    # segmentRes=analyzer.analyze(sentence)
    return segmentRes


def wordEmcedding_create( segPharse_Path = 'pharse_seg_result.txt',seg_base_word_Path='baseWord_seg_result.txt' ):
    startT = time.time()
    print('\n开始嵌入生成： ')

    embedding_SaveFile=r'embedding.csv'

    #1.获取短语分词
    all_pharses=[]
    with open(segPharse_Path,'r',encoding='utf-8') as f:
        lines=f.readline()

        all_pharses=lines.split('()')
        f.close()
    all_pharses=list(set(all_pharses))

    #2.获取基础词分词
    all_base_Words=[]
    with open(seg_base_word_Path,'r',encoding='utf-8') as f:
        lines=f.readline()
        all_base_Words+=lines.split('()')
        f.close()


    #3.计算短语对应基础词的词向量，数据格式为m x n, m为短语数量， n为基础词数量，嵌入特征存入csv文件
    m=len(all_pharses)
    n=len(all_base_Words)

    #设置csv表头
    header=['pharseNam']+all_base_Words


    #新建m x n的零矩阵
    emdedding_data=[['0' for j in range(n)] for i in range(m) ]

    #4.统计短语中各个基础词的数量
    for pharse in tqdm(all_pharses):
        for baseWord in all_base_Words:
            val=str(len(pharse.split(baseWord))-1)
            if val !='0':
                pharse_idx=all_pharses.index(pharse)
                base_word_idx = all_base_Words.index(baseWord)
                emdedding_data[pharse_idx][base_word_idx]=val

    #5.完整csv数据data
    data=[]
    for i,pharse in enumerate(all_pharses):
        row=[pharse]+emdedding_data[i]
        data.append(row)

    #6.将完整数据存入CSV文件

    with open(embedding_SaveFile, 'w', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        # 设置第一行标题头
        writer.writerow(header)
        # 将数据写入
        writer.writerows(data)

    print('pharse,{},{}'.format(len(all_pharses),len(list(set(all_pharses)))))
    print('word,{},{}'.format(len(all_base_Words), len(list(set(all_base_Words)))))


    endT = time.time()
    print('词嵌入时间消耗： {}s'.format(endT-startT))
def clear_useless_pharseWithSim(csvfile=r''):
    lines=[line for line in open(csvfile,encoding='utf-8')]
    lines=lines[:]
    # 第一行是列标题
    colnames=lines[0].strip().split(',')[1:]
    rownames=[]
    data=[]
    rownames_data=[]
    for line in lines[1:]:
        p=line.strip().split(',')
        # 每行的第一列是行名
        rownames.append(p[0])
        # 剩余部分就是该行对应的数据
        onerow = [float(x) for x in p[1:]]
        data.append(onerow)
        rownames_data.append([p[0],onerow])

    rownames_sum_sim=[] #记录短语到非自身其他所有相似度总和
    count=0
    max_sim=0
    for i,rowname_i in enumerate(rownames) :
        sum_sim=0.0
        for j,rowname_j in enumerate(rownames):
            if rowname_i!=rowname_j:
                sim=pearson(data[i],data[j])
                sum_sim+=sim
        if max_sim<sum_sim:max_sim=sum_sim
        rownames_sum_sim.append([rowname_i,sum_sim])
        count+=1
        print(round(count/len(rownames)*100,2),'%')
    for i,val in enumerate(rownames_sum_sim):
        rownames_sum_sim[i]+=[round(val[1]/max_sim,4)]


    sims = sorted(rownames_sum_sim, key=lambda t: t[::-1])
    print(sims)





def finalCluster(xlsx=r'',isBaseWordType=False,jpgNam=''):
    startT = time.time()

    if isBaseWordType:
        baseWordType = 'hanlp'
        out_jpg='hanlp_all_{}.jpg'.format(jpgNam)
    else:
        baseWordType = 'nothanlp'
        out_jpg = 'nothanlp_all_{}.jpg'.format(jpgNam)

    keywordExtract_TFIDF(xlsxFile=xlsx,baseWordType=baseWordType)
    wordEmcedding_create(segPharse_Path ='../data/midFile/pharse_seg_result.txt', seg_base_word_Path='../data/midFile/baseWord_seg_result.txt')
    # clear_useless_pharseWithSim(csvfile='embedding.csv')
    # whole_Cluster_process(csvFile='embedding.csv', jpgSave=out_jpg)

    endT = time.time()
    print('总聚类过程时间消耗： {}s'.format(endT-startT))

def clusterIMG_show(imgfile=r'clustrRes1.jpg'):
    img=Image.open(imgfile)
    Image._show(img)
def spiderResult_union():
    #整合该母文件夹下的爬虫excel数据

    motherDir=r'C:\Users\glodon\Desktop\jobCollect\预算员爬虫信息'
    spider_union_xlsx=r'spider_union_result.xlsx'
    files=os.listdir(motherDir)
    small_job_list, job_text_list=[],[]
    res=[]
    count=0
    for file in files:
        init_count=count
        xlsxfile=os.path.join(motherDir,file)
        print('开始处理 ',xlsxfile )
        df = pd.read_excel(xlsxfile)
        small_job,job_text=df['标题'].tolist(),df['字段1'].tolist()
        for i in range(len(small_job)):
            if (job_text[i] not in job_text_list) and (job_text[i]!='') :
            #     small_job_list.append(small_job[i])
            #     job_text_list.append(job_text[i])
                res.append([small_job[i],job_text[i]])
                count+=1

        print('该文件数据量条数: ',count-init_count ,'\n')


    print('总数据量: ',len(res))
    data = pd.DataFrame(data=res)
    data.to_excel(spider_union_xlsx, index=False, header=['标题','字段1'])










if __name__ == '__main__':

    # 爬虫信息xlsx文件整合
    # spiderResult_union()


    # 清除无效短语
    # clear_useless_pharseWithSim(csvfile='embedding.csv')


    # 整个聚类函数
    finalCluster(xlsx='E:\PythonProject\zhilianDataProcess\data\midFile\spider_union_result.xlsx',isBaseWordType=False,jpgNam='猎聘')
    finalCluster(xlsx='E:\PythonProject\zhilianDataProcess\data\midFile\spider_union_result.xlsx', isBaseWordType=True,jpgNam='猎聘')

    # clusterIMG_show(imgfile=r'clustrRes1.jpg') #展示图片
    # clusterIMG_show(imgfile=r'clustrRes2.jpg')




