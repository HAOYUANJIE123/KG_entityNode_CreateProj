from mainCodeFile.excelRead import pandas_readXlsx
from sklearn.feature_extraction.text import TfidfVectorizer


##!/usr/bin/env python
## coding=utf-8

def keywordExtract_TFIDF(xlsxFile=r'C:\Users\glodon\Desktop\jobCollect\zhilian\智联预算员0.xlsx'):
    fileSegWordDonePath ='预算招聘处理原始0.txt'
    # read the file by line
    # _, text = readXls()
    _,text=pandas_readXlsx(xlsxFile)
    fileTrainRead =text[:]
    # segment word with IF-IDF (方法二)
    fileTrainSeg = []
    for i in range(len(fileTrainRead)):
        if not (fileTrainRead[i]!=fileTrainRead[i]): #去除nan值项
            tfidf = TfidfVectorizer()
            weight = tfidf.fit_transform([fileTrainRead[i]]).toarray()
            word = tfidf.get_feature_names()
            print(word)
            fileTrainSeg.append(['()'.join(word)])

    print(len(fileTrainSeg))

    # save the result
    with open(fileSegWordDonePath,'wb') as fW:
        for i in range(len(fileTrainSeg)):
            fW.write(fileTrainSeg[i][0].encode('utf-8'))
            fW.write('\n'.encode('utf-8'))


    # # create wordVector
    # sentences = word2vec.LineSentence('corpusSegDone.txt')
    # model=word2vec.Word2Vec(sentences=sentences, sg=0, hs=1,min_count=1,window=10,vector_size=100)
    #
    # for key in model.wv.similar_by_word('工作经验'.encode('utf-8'), topn =5):
    #     print(key)
def webAdd_create(pageNum=10,ProfessionNam='预算员'):


    for i in range(1,pageNum+1):
        WebAdd='https://www.zhipin.com/web/geek/job?query={}&city=100010000&page={}'.format(ProfessionNam,i)
        print(WebAdd)

if __name__ == '__main__':
    webAdd_create() #爬虫网址批量生成
    # pandas_readXlsx() #读取智联招聘爬虫结果
    # keywordExtract_TFIDF()  #基于TIFIDF算法提取关键信息
    # smallJob,Jobtext=pandas_readXlsx()




