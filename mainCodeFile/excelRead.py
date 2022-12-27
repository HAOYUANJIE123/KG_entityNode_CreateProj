import pandas as pd
def pandas_readXlsx(xlsxFile=r'C:\Users\glodon\Desktop\jobCollect\zhilian\智联预算员0.xlsx'):
    df = pd.read_excel(xlsxFile)


    smallJob=df['标题'].tolist()
    smallJob=list(set(smallJob))
    for i,val in enumerate(smallJob):
        if val==val and '（' in val and '）' in val :
            idx_left_kuo=val.index('（')
            idx_right_kuo = val.index('）')
            clear_val=val[idx_left_kuo:idx_right_kuo+1]
            dealed_val=val.replace(clear_val,'')
            smallJob[i]=dealed_val
            pass
    Jobtext=df['字段1'].tolist()
    Jobtext=list(set(Jobtext))
    return smallJob,Jobtext

def pandas_read_major_Xlsx(xlsxFile=r'C:\Users\glodon\Desktop\知识图谱搭建\预算员图谱构建\预算员相关专业课程\陕西高校专业信息终.xlsx'):
    df = pd.read_excel(xlsxFile,sheet_name=1)
    smallMajor= df['专业小类'].tolist()
    smallMajor=list(set(smallMajor))
    for i in smallMajor:
        Res=df[df['专业小类']==i].head()
        majorDes=Res['专业描述'].tolist()
        majorDes=list(set(majorDes))
        with open('专业描述/'+i+'专业描述.txt','wb') as f:
            for j in majorDes:
                f.write((j+'\n\n\n').encode('utf-8'))
            print(i ,j,'\n\n')
            f.close()





if __name__ == '__main__':
    pandas_read_major_Xlsx() #读取学校专业xlsx信息
