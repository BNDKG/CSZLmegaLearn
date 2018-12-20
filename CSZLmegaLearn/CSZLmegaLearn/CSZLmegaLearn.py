#coding=utf-8

import tushare as ts
import pandas as pd
import numpy as np
import os
import random
import copy

#正则表达式
import re
import datetime
import time
import random

#自写的展示类
from CSZLmegaDisplay import CSZLmegaDisplay


#文件夹总路径
cwd = os.getcwd()


def CSZL_History_Read():
    '''
    读取全部历史数据

    '''

    global All_K_Data
    #global DataRecord

    TrainDate=[]


    #txtFile1 = cwd + '\\data\\'+'ALL_History_data.npy'
    HistoryDataSourcePath =cwd + '\\data\\'+'History_data.npy'
    #HistoryDataSourcePath ='D:\CSZLsuper\CSZLsuper\CSZLsuper\data\ALL_History_data.npy'
    databased='D:\CSZLsuper\CSZLsuper\CSZLsuper\output\ALL_History_data_Datebased'

    All_K_Data=np.load(HistoryDataSourcePath)


    startdate=20180326
    enddate=20180526

    z=All_K_Data.shape[2]    

    for ii in  range(z):
        #获取代码名称在codelist列表中
        codelist=All_K_Data[:,0,0]
        #生成一个代码长度相等的序列
        rangez=range(All_K_Data.shape[0])
        #获取某日所有股票的收盘价
        pricelist=All_K_Data[:,3,ii]
        #获取最高价
        vollist=All_K_Data[:,2,ii]
        #如果最高级等于收盘价说明可能有毒放入删除列表
        dellist=np.where(pricelist==vollist)

        #获取所有股票第二日是收盘价
        pricelist2=All_K_Data[:,3,ii+1]

        #计算次日涨幅，防止除数是0加了0.1
        pluslist=(pricelist2-pricelist)/(pricelist+0.1)

        #去除异常值
        pluslistindex=np.where((-0.11>pluslist) | (0.09<pluslist))
        #pluslistindex=np.where((-0.1>pluslist) | (0.1<pluslist))

        #合并涨幅代码现价
        out2test=np.vstack((rangez,pluslist))
        out2test=np.vstack((out2test,codelist))
        out2test=np.vstack((out2test,pricelist))

        concetarry=np.append(pluslistindex,dellist)
        concetarry=np.unique(concetarry)

        #去除刚刚提取的涨幅异常值
        zzzz3=np.delete(out2test,concetarry,1)

        maxlistindex=np.where(zzzz3<0.09)
        zzzz4=np.delete(zzzz3,maxlistindex,1)

        zzzz4=zzzz4[:,0:10]

        #CSZLmegaDisplay.twodim(zzzz4[0],zzzz4[1],title=All_K_Data[1111,6,ii],x_tick=zzzz4[2],y_tick=zzzz4[3])
        #CSZLmegaDisplay.twodim(zzzz3[0],zzzz3[1])
        pointz=my_EM(zzzz3[1])

        labelarange=np.arange(-0.1,0.1,0.01).tolist()
        z2=np.abs(labelarange-pointz)
        z3=np.min(z2)
        z4=np.where(z2<0.01)
        labelarange=np.delete(labelarange,z4)

        labelarange=labelarange.tolist()
        labelarange.append(pointz)
        labelarange=np.array(labelarange)
        CSZLmegaDisplay.onedim(zzzz3[1],labelarange,pointz)

        #这里暂时拿特定一个数据来作为日期检测的种子,之后会寻找更加合适的方法1328
        if(All_K_Data[(1111,6,ii)]>=startdate and All_K_Data[(1111,6,ii)]<=enddate ):
            TrainDate.append(All_K_Data[(1,6,ii)])

        CSZLmegaDisplay.clean()

    return TrainDate


def HistoryDataGet(
    DayEnd=datetime.datetime.now().strftime("%Y-%m-%d"),
    Datas=365,
    Path='History_data.npy',
    ReadPath=cwd+'\\StockList'+'\\today_all_data.csv'):
    """
    截止日期("xxxx-xx-xx")
    获取天数(int)

    获取数据为截止日期前指定交易天数的数据,
    保存到data/History_data中无返回值

    从today_all_data.csv读取数据


    """
    buff_dr_result=pd.read_csv(ReadPath,encoding= 'gbk')

    dwww=cwd

    days2=Datas*1.5+10


    timeArray = time.strptime(DayEnd, "%Y-%m-%d")

    timeNow = datetime.datetime(int(timeArray[0]), int(timeArray[1]), int(timeArray[2]), 12, 0, 0); 
    DayStart = (timeNow - datetime.timedelta(days = days2)).strftime("%Y-%m-%d")
    

    HistoryDataSave=np.zeros((4000,7,Datas),dtype=float)

    if True:
        all=len(buff_dr_result['code'])

        for z in range(all):
            try:
  
                if(z%30==0):
                    print(z/all)
                    print("\n")
              
                #temp=str(buff_dr_result['code'][z],"utf-8")
                temp=str(buff_dr_result['code'][z]).zfill(6)
                #这里注意是字符串temp转为数字保存到HistoryDataSave中可能会有bug
                HistoryDataSave[(z,0,0)]=temp

                kget=ts.get_k_data(temp,start=DayStart, end=DayEnd)

                Kdata=kget.tail(Datas)
                
                datamax=len(Kdata)

                x=0

                if(Kdata.empty==True):
                    continue

                #for x in range(0,datamax):
                for singledatezz in Kdata.date:


                    changedate=time.strptime(singledatezz,"%Y-%m-%d")
                    changedate2=time.strftime("%Y%m%d",changedate)
                    changedate3=int(changedate2)
                    HistoryDataSave[(z,6,x)]=changedate3


                    HistoryDataSave[(z,1,x)]=Kdata.open.data[x]
                    HistoryDataSave[(z,2,x)]=Kdata.high.data[x]
                    HistoryDataSave[(z,3,x)]=Kdata.close.data[x]
                    HistoryDataSave[(z,4,x)]=Kdata.low.data[x]
                    HistoryDataSave[(z,5,x)]=Kdata.volume.data[x]

                    x+=1

                #txtFile = cwd + '\\data\\'+Path
                #np.save(txtFile, HistoryDataSave)

            except Exception as ex:
                sleeptime=random.randint(50,99)
                time.sleep(sleeptime/100)       
                wrongmessage="HistoryRoutine FAIL at : %s \n" % ( time.ctime(time.time()))
                print (wrongmessage)

                print (Exception,":",ex)
                z-=1


        txtFile = cwd + '\\data\\'+Path
        np.save(txtFile, HistoryDataSave)
def CSZL_CodelistToDatelist():
    '''
    将以股票代码为基本单位的列表转换为以日期为基本的列表
    '''

    global All_K_Data

   
    x=All_K_Data.shape[0]    #4000
    y=All_K_Data.shape[1]    #7
    z=All_K_Data.shape[2]    #2000

    # 日期 代码 信息
    DateBasedList=np.zeros((z,x,y),dtype=float)


    bufflist=ts.get_k_data('000001',start='2010-03-01', end='2018-06-13', index=True) 

    datelist=bufflist.date.tail(z)

    searchcounter=0
    updatecounter=0

    i=0
    for singledatezz in datelist:

        changedate=time.strptime(singledatezz,"%Y-%m-%d")
        changedate2=time.strftime("%Y%m%d",changedate)
        changedate3=int(changedate2)
        
        DateBasedList[(i,0,0)]=changedate3

        date_index=0
        for ii in range(x):
            cur_changedata=All_K_Data[ii,6,date_index]
            if(changedate3==cur_changedata):
                DateBasedList[i,ii,:]=All_K_Data[ii,:,date_index]
                DateBasedList[i,ii,6]=All_K_Data[ii,0,0]
                
            else:
                
                bufsearch=All_K_Data[ii,6,:]
                #从历史数据列表中寻找是否有对应值
                buff=np.argwhere(bufsearch==changedate3)
                #如果有指则重新定义历史数据位置
                if(buff!=None):
                    foundindex=int(buff)
                    date_index=foundindex
                    zzz2=All_K_Data[(ii,6,date_index)]

                    DateBasedList[i,ii,:]=All_K_Data[ii,:,date_index]
                    DateBasedList[i,ii,6]=All_K_Data[ii,0,0]
                    
                    searchcounter+=1
                else:
                    
                    updatecounter+=1
                    continue
        i+=1
        if(i>1999):
            break;

    '''
    for i in  range(z):
        print(DateBasedList[i,0,0])
        for ii in  range(x):
            asdad3=DateBasedList[i,ii,3]
            asdad2=DateBasedList[i,ii,6]
            print("%2.4f %d " % (asdad3,asdad2))
            
        print("\n")
    '''
    cwd = os.getcwd()

    txtFileA = cwd + '\\output\\ALL_History_data_Datebased.npy'
    np.save(txtFileA, DateBasedList)


    sdfsdf=5

def anafirsttest():
    '''
    分析数据
    '''

    #datapath=cwd + '\\data\\secret\\A'
    file_dir=r"D:\CSZLsuper\CSZLsuper\CSZLsuper\data\secret\A"
    
    for root, dirs,files in os.walk(file_dir):
        L=[]
        for file in files:  
            if os.path.splitext(file)[1] == '.npy':  
                L.append(os.path.join(root, file))

    #遍历所有文件
    for z_file in L:
        #试试我的正则功力
        nums = re.findall(r"secretA(\d+).",z_file)
        if(nums!=[]):
            cur_date=float(nums[0])
        else:
            continue
        if cur_date>=20180510 and cur_date<=20181010:
            #先测试图表绘制
            print(z_file)

#EM算法
SIGMA = 6

EPS = 0.0001

def my_EM(X):
    k = 1
    N = len(X)
    Miu = np.random.rand(k,1)
    Posterior = np.mat(np.zeros((N,2)))
    dominator = 0
    numerator = 0
    #先求后验概率
    for iter in range(1000):
        for i in range(N):
            dominator = 0
            for j in range(k):
                dominator = dominator + np.exp(-1.0/(2.0*SIGMA**2) * (X[i] - Miu[j])**2)
                #print dominator,-1/(2*SIGMA**2) * (X[i] - Miu[j])**2,2*SIGMA**2,(X[i] - Miu[j])**2
                #return
            for j in range(k):
                numerator = np.exp(-1.0/(2.0*SIGMA**2) * (X[i] - Miu[j])**2)
                Posterior[i,j] = numerator/dominator			
        oldMiu = copy.deepcopy(Miu)
        #最大化	
        for j in range(k):
            numerator = 0
            dominator = 0
            for i in range(N):
                numerator = numerator + Posterior[i,j] * X[i]
                dominator = dominator + Posterior[i,j]
            Miu[j] = numerator/dominator
        print ((abs(Miu - oldMiu)).sum())

        if (abs(Miu - oldMiu)).sum() < EPS:
            print (Miu,iter)
            break
    return Miu[0][0]




if __name__ == '__main__':

    #adwdd=ts.get_k_data("603999",start="2018-10-10", end="2018-12-08")

    #获取历史信息
    HistoryDataGet(Datas=10)


    CSZL_History_Read()



    anafirsttest()

    #CSZLmegaDisplay.close()





    