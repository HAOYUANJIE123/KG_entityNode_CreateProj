from pyhanlp import *
from collections import Counter
import re
from tqdm import tqdm
from otherCodeFile.TFIDF_extractKEYword import pandas_readXlsx


def test():
    print(HanLP.segment('你好，欢迎在Python中调用HanLP的API'))
    for term in HanLP.segment('下雨天地面积水'):
        print('{}\t{}'.format(term.word, term.nature)) # 获取单词与词性
    testCases = [
        "商品和服务",
        "结婚的和尚未结婚的确实在干扰分词啊",
        "买水果然后来世博园最后去世博会",
        "中国的首都是北京",
        "欢迎新老师生前来就餐",
        "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
        "随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。"]
    for sentence in testCases: print(HanLP.segment(sentence))
    # 关键词提取
    document = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
               "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
               "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
               "严格地进行水资源论证和取水许可的批准。"
    print(HanLP.extractKeyword(document, 2))
    # 自动摘要
    print(HanLP.extractSummary(document, 3))
    # 依存句法分析


def hanlp_segment(sentence='',num=3):
    # sentence="大专及以上学历，工程造价专业1年及以上工作经验；2、熟练应用广联达等造价软件，以及office、autocad等相关的办公软件；"


    # #分词
    segmentRes=HanLP.segment(sentence)
    #
    # #关键词提取
    # print(HanLP.extractKeyword(sentence, 2))

    #短语提取
    # segmentRes=HanLP.extractPhrase(sentence,num)
    return segmentRes

    # #基于感知机进行分词
    # PerceptronLexicalAnalyzer=JClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
    # analyzer=PerceptronLexicalAnalyzer()
    # print(analyzer.analyze(sentence))

def hanlp_segment_zaojia(xlsxFile=r'C:\Users\glodon\Desktop\jobCollect\zhilian\智联预算员0.xlsx'):
    _,text = pandas_readXlsx(xlsxFile)
    fileTrainRead = text[:]
    fileTrainSeg = []
    keyWords=[]
    allWords=[]
    word2nature={}
    CoreStopWordDictionary = JClass("com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary")
    for i in tqdm(range(len(fileTrainRead))):
        sentence = fileTrainRead[i]
        if sentence==sentence:
            oneSentenceWord=[]
            sentenceSplit=re.split(r'[；。]',sentence)
            # print(sentence+'\n\n')
            for i,line in enumerate(sentenceSplit):

                res=hanlp_segment(sentence=line,num=3)

                CoreStopWordDictionary.apply(res)#去除标点、无关词

                for term in res:
                    allWords.append(term.word)
                    word2nature[term.word]={'nature':str(term.nature).encode('utf-8')}
                    # print('{}\t{}'.format(term.word, term.nature))  # 获取单词与词性
                oneSentenceWord+=res

                # print(i+1,line,'\n',res,'\n\n')
            keyWords.append(oneSentenceWord)
        # if i==1:
        #     break

            # print(word)
            # fileTrainSeg.append([' '.join(word)])
    #频数统计,并取出top的词
    c=Counter(allWords)
    top_c=c.most_common(100)
    clist=list(top_c)
    top_c_name=[k[0] for k in clist]

    print(len(fileTrainSeg))

    #相关性统计
    #读取tf-idf关键词结果
    all_longWords=[]
    with open(r'../预算招聘处理原始0.txt', 'r', encoding='utf-8') as f:
        lines=f.readline()

        all_longWords+=lines.split('()')
        f.close()

    allsims=[]
    for S_Word in tqdm(allWords):
        sims=[]
        for L_Word in all_longWords:
            a=S_Word
            b=L_Word
            sim=TextRank_sim(a,b)
            if sim >0.0 :
                sims.append((a,b,sim))
                allsims.append((a,b,sim))

        sims=sorted(sims, key=lambda t: t[::-1],reverse=True)
        print(sims)

        # break
    allsims=sorted(allsims, key=lambda t: t[::-1],reverse=True)
    print(allsims)

def phraseSimCaculation():
    #读取tf-idf关键词结果
    all_longWords=[]
    with open(r'../预算招聘处理原始0.txt', 'r', encoding='utf-8') as f:
        lines=f.readline()
        all_longWords+=lines.split('()')
        f.close()

    allsims=[]
    allChars=[]
    history_record=[]
    for L_Word1 in tqdm(all_longWords):
        sims=[]
        allChars+=list(L_Word1)
        for L_Word2 in all_longWords:
            a=L_Word1
            b=L_Word2
            if (b,a) not in history_record:
                history_record.append((a,b))

                sim=TextRank_sim(a,b)
                if sim >0.0 and sim!=1.0 :
                    # jiao= len(list(set(a) & set(b)))
                    # jiao=set(a) & set(b)
                    sims.append((a,b,sim))
                    allsims.append((a,b,sim))

        sims=sorted(sims, key=lambda t: t[::-1],reverse=True)
        # print(sims)

        # break
    allsims=sorted(allsims, key=lambda t: t[::-1],reverse=True)
    print(len(allsims),allsims)
    char_c=Counter(allChars)
    print(len(allChars),allChars)
    print(char_c)
    print(len(list(set(allChars))))


def TextRank_sim(a,b):
    # sim=len(list(set(a)&set(b)))/len(list(set(a+b)))
    if a==b:
        sim=1.0
    else:
        sim = len(list(set(a) & set(b))) / min(len(a),len(b))
        # sim = len(list(set(a) & set(b))) / len(a)
        sim=round(sim,3)
    return sim


if __name__ == '__main__':
    # hanlp_segment()
    # hanlp_segment_zaojia()
    phraseSimCaculation()



