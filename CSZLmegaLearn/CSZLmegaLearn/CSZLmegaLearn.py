#coding=utf-8


import pandas as pd
import numpy as np
import os
import random
import copy

import tushare as ts

import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold

#正则表达式
import re

import datetime
import time
import random

#自写的展示类
from CSZLmegaDisplay import CSZLmegaDisplay
import gc
from sklearn.externals import joblib

import matplotlib.pyplot as plt
#文件夹总路径
cwd = os.getcwd()

def Get_AllkData():
    global All_K_Data    
    HistoryDataSourcePath =cwd + '\\data\\'+'History_data.npy'
    All_K_Data=np.load(HistoryDataSourcePath)

def Get_Datebased():
    global Datebaseddata

    dbPath =cwd + '\\data\\'+'Datebased_data.npy'
    Datebaseddata=np.load(dbPath)


def CSZL_History_Read():
    '''
    读取全部历史数据

    '''

    global All_K_Data
    #global DataRecord

    TrainDate=[]

    startdate=20180326
    enddate=20180526

    z=All_K_Data.shape[2]    

    for ii in  range(140,z-1,1):
        

        #获取代码名称在codelist列表中
        codelist=All_K_Data[:,0,0]
        #生成一个代码长度相等的序列
        rangez=range(All_K_Data.shape[0])
        #获取某日所有股票的收盘价
        pricelist=All_K_Data[:,3,ii]
        #获取最高价
        highlist=All_K_Data[:,2,ii]
        #获取量
        vollist2=All_K_Data[:,5,ii]
        #如果最高价等于收盘价说明可能有毒放入删除列表
        dellist=np.where(pricelist==highlist)

        #获取所有股票第二日是收盘价
        pricelist2=All_K_Data[:,3,ii+1]
        #获取所有股票前一日收盘价
        pricelist3=All_K_Data[:,3,ii-1]        

        #计算次日涨幅，防止除数是0加了0.1
        pluslist=(pricelist2-pricelist)/(pricelist+0.1)
        #计算当日涨幅
        pluslisttoday=(pricelist-pricelist3)/(pricelist3+0.1)

        #计算当日成交金额
        vollist=(vollist2*(pricelist/10000))

        #去除异常值
        pluslistindex=np.where((-0.11>pluslist) | (0.11<pluslist))
        #pluslistindex=np.where((-0.1>pluslist) | (0.1<pluslist))

        #合并涨幅代码现价量
        out2test=np.vstack((rangez,pluslist))
        out2test=np.vstack((out2test,codelist))
        out2test=np.vstack((out2test,pricelist))
        out2test=np.vstack((out2test,vollist))
        out2test=np.vstack((out2test,pluslisttoday))


        #合并两个要删的list并去重
        concetarry=np.append(pluslistindex,dellist)
        concetarry=np.unique(concetarry)

        #去除刚刚提取的涨幅异常值
        zzzz3=np.delete(out2test,concetarry,1)
        #按照第一行排序(有第0行)
        zzzz3=zzzz3.T[np.lexsort(zzzz3[3,None])].T
        #a.T[np.lexsort(-a[0,None])].T  #按第0行的大小逆序排序
        #a[np.lexsort(a.T[0,None])]   #按第0列的大小排序


        #提取涨停的个股
        #maxlistindex=np.where(zzzz3<0.09)
        #zzzz4=np.delete(zzzz3,maxlistindex,1)

        #zzzz4=zzzz4[:,0:10]
        ssef=range(zzzz3.shape[1])


        todaydata=All_K_Data[(12,6,ii)]
        xxx=todaydata//10000
        xxx2=todaydata//100-xxx*100
        xxx3=int(todaydata)%100

        days2=1
        timeNow = datetime.datetime(int(xxx), int(xxx2), int(xxx3), 12, 0, 0); 
        DayStart = (timeNow + datetime.timedelta(days = days2)).strftime("%Y%m%d")

        color="blue"
        if(int(DayStart)!=All_K_Data[(1111,6,ii+1)]):
            color="red"

        CSZLmegaDisplay.twodim(ssef,zzzz3[5],title=todaydata,colori=color)
        #CSZLmegaDisplay.twodim(zzzz3[0],zzzz3[1])

        ##em算法计算高斯中值
        #pointz=my_EM(zzzz3[1])
        ##设置显示标记
        #labelarange=np.arange(-0.1,0.1,0.01).tolist()
        ##去除标记中最接近中值的两个值防止重叠显示
        #z2=np.abs(labelarange-pointz)
        #z3=np.min(z2)
        #z4=np.where(z2<0.01)
        #labelarange=np.delete(labelarange,z4)

        #labelarange=labelarange.tolist()
        #labelarange.append(pointz)
        #labelarange=np.array(labelarange)
        #CSZLmegaDisplay.onedim(zzzz3[1],labelarange,pointz)

        #这里暂时拿特定一个数据来作为日期检测的种子,之后会寻找更加合适的方法1328
        if(todaydata>=startdate and todaydata<=enddate ):
            TrainDate.append(All_K_Data[(1,6,ii)])

        #CSZLmegaDisplay.clean()

    return TrainDate

def CSZL_HistoryDB_Read():
    '''
    读取全部历史数据

    '''
    global Datebaseddata    
    Get_Datebased()

    z=Datebaseddata.shape[0]    


    #CSZLmegaDisplay.twodim(range(z),Datebaseddata[:,0,0])

    for ii in  range(2,z-1,1):
        #获取代码名称在codelist列表中
        codelist=Datebaseddata[ii,:,6]
        #生成一个代码长度相等的序列
        rangez=range(Datebaseddata.shape[1])
        #获取某日所有股票的收盘价
        pricelist=Datebaseddata[ii,:,3]
        #获取最高价
        highlist=Datebaseddata[ii,:,2]
        #获取量
        vollist2=Datebaseddata[ii,:,5]
        vollistb=Datebaseddata[ii-1,:,5]
        #如果最高价等于收盘价说明可能有毒放入删除列表
        dellist=np.where(pricelist==highlist)

        #获取所有股票第二日收盘价
        pricelist2=Datebaseddata[ii+1,:,2]
        #获取所有股票前一日收盘价
        pricelist3=Datebaseddata[ii-1,:,3]        

        #计算次日涨幅，防止除数是0加了0.1
        pluslist=(pricelist2-pricelist)/(pricelist+0.1)
        #计算当日涨幅
        pluslisttoday=(pricelist-pricelist3)/(pricelist3+0.1)

        #计算当日成交金额
        vollist=(vollist2*(pricelist/10000))
        #计算前一日成交金额
        vollistbefore=(vollistb*(pricelist3/10000))
        #计算当日成交比昨日成交多的百分比
        vollistbefore=(vollist/(vollistbefore+1))

        #去除异常值
        pluslistindex=np.where((-0.11>pluslist) | (0.11<pluslist))
        #pluslistindex=np.where((-0.1>pluslist) | (0.1<pluslist))

        #合并涨幅代码现价量
        out2test=np.vstack((rangez,pluslist))
        out2test=np.vstack((out2test,codelist))
        out2test=np.vstack((out2test,pricelist))
        out2test=np.vstack((out2test,vollistbefore))
        out2test=np.vstack((out2test,pluslisttoday))

        #合并两个要删的list并去重
        concetarry=np.append(pluslistindex,dellist)
        concetarry=np.unique(concetarry)
        #concetarry=np.delete(concetarry,[0])

        #去除刚刚提取的涨幅异常值
        zzzz3=np.delete(out2test,concetarry,1)
        #按照第一行排序(有第0行)
        zzzz3=zzzz3.T[np.lexsort(zzzz3[5,None])].T
        #a.T[np.lexsort(-a[0,None])].T  #按第0行的大小逆序排序
        #a[np.lexsort(a.T[0,None])]   #按第0列的大小排序


        #提取涨停的个股
        #maxlistindex=np.where(zzzz3<0.09)
        #zzzz4=np.delete(zzzz3,maxlistindex,1)

        #zzzz4=zzzz4[:,0:10]
        ssef=range(zzzz3.shape[1])


        todaydata=Datebaseddata[(ii,0,0)]
        xxx=todaydata//10000
        xxx2=todaydata//100-xxx*100
        xxx3=int(todaydata)%100

        days2=1
        timeNow = datetime.datetime(int(xxx), int(xxx2), int(xxx3), 12, 0, 0); 
        DayStart = (timeNow + datetime.timedelta(days = days2)).strftime("%Y%m%d")

        color="blue"
        if(int(DayStart)!=Datebaseddata[(ii+1,0,0)]):
            color="red"

        CSZLmegaDisplay.twodim(ssef,zzzz3[1],title=todaydata,colori=color)

        pass



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
    #z=4   #2000

    # 日期 代码 信息
    DateBasedList=np.zeros((z,x+1,y),dtype=float)


    bufflist=ts.get_k_data('000001',start='2017-03-01', end='2018-12-07', index=True) 

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
                DateBasedList[i,ii+1,:]=All_K_Data[ii,:,date_index]
                DateBasedList[i,ii+1,6]=All_K_Data[ii,0,0]
                
            else:
                
                bufsearch=All_K_Data[ii,6,:]
                #从历史数据列表中寻找是否有对应值
                buff=np.argwhere(bufsearch==changedate3)
                #如果有指则重新定义历史数据位置
                if(buff!=None):
                    foundindex=int(buff)
                    date_index=foundindex
                    zzz2=All_K_Data[(ii,6,date_index)]

                    DateBasedList[i,ii+1,:]=All_K_Data[ii,:,date_index]
                    DateBasedList[i,ii+1,6]=All_K_Data[ii,0,0]
                    
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

    txtFileA = cwd + '\\data\\Datebased_data.npy'
    np.set_printoptions(suppress=True)
    print(DateBasedList)

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



def Get_DateBased_Feature(Start='2015-01-01',end='2018-12-31'):

    global Global_train_data
    #date(日期/主键) 

    #szcz深圳成指(以此的date为key)
    b=ts.get_k_data("399001",start=Start, end=end)
    #主键
    xx=b['date'].values
    Data_Merge(xx)
    #收盘价
    xx=b['close'].values
    Data_Merge(xx)
    #成交量
    xx=b['volume'].values
    Data_Merge(xx)

    #cybz创业板指 159915这类基金也可以获取但一些指数不能获取
    b=ts.get_k_data("399006",start=Start, end=end)

    #收盘价
    xx=b['close'].values
    Data_Merge(xx)
    #成交量
    xx=b['volume'].values
    Data_Merge(xx)

    #收盘价
    xx=b['close'].values
    Data_Merge(xx)
    #成交量
    xx=b['volume'].values
    Data_Merge(xx)


    #Reverse_Repo逆回购利率
    a=ts.get_k_data("131810",start=Start, end=end)

    #先传入一个日期的key，来填补缺失数据
    xx=a['date'].values
    Data_Merge(xx)
    xx=a['high'].values
    Data_Merge(xx)

    print(Global_train_data)
    pass

#先用全局变量应付一下，todo使用class封装起来
Global_train_data=[]

miss_index=[]
def Data_Merge(column):
    '''
    判断行数一样直接加行数不一样加成一样再加
    '''

    global miss_index

    if(len(Global_train_data)==len(column)or len(Global_train_data)==0):
        sef=column
    else:
        if(len(miss_index)==0):
            asdw=Global_train_data.T
            asdw2=asdw[0]
            ii=0
            fesef=[]
            for date in asdw2:
                if(date!=column[ii]):
                    fesef.append(ii)
                else:
                    ii+=1
            miss_index=fesef
            return
        else:
            sef=column
            for empty_index in reversed(miss_index):           

                sef=np.insert(sef,empty_index,-1,axis=0)
            miss_index=[]

    xx2=np.array(sef.T)
    xx2=xx2.reshape([xx2.shape[0],1])    
    _Data_Merge(xx2)

    pass

def _Data_Merge(column):
    '''
    将输入的列合并到总的列中,数量不相等返回错误
    '''
    global Global_train_data
    if(len(Global_train_data)):
        Global_train_data=np.hstack((Global_train_data, column))
        
        pass
    else:
        Global_train_data=column
        pass

def get_codeanddate_feature():

    #mydict={'id':["000002.SZ","000004.SZ","000005.SZ"],'fea':["dwdw","dwd","www"]}
    #test_dict_df = pd.DataFrame(mydict)
    #test_dict_df.set_index('id',inplace=True)

    #print(test_dict_df)

    #读取token
    f = open('token.txt')
    token = f.read()     #将txt文件的所有内容读入到字符串str中
    f.close()


    pro = ts.pro_api(token)

    date=pro.query('trade_cal', start_date='20190102', end_date='20190404')

    date=date[date["is_open"]==1]
    get_list=date["cal_date"]

    df_all=pro.daily(trade_date="20190101")

    zcounter=0
    zall=get_list.shape[0]
    for singledate in get_list:
        zcounter+=1
        print(zcounter*100/zall)

        dec=5
        while(dec>0):
            try:
                time.sleep(1)
                df = pro.daily(trade_date=singledate)

                df_all=pd.concat([df_all,df])

                #df_last
                #print(df_all)
                break

            except Exception as e:
                dec-=1
                time.sleep(5-dec)

        if(dec==0):
            fsefe=1


    df_all=df_all.reset_index(drop=True)

    df_all.to_csv("savetest2019.csv")





    ##df = pro.query('adj_factor',  trade_date='20180315')

    ##df = pro.shibor(start_date='20100101', end_date='20101101')

    #df[["change","pct_chg"]]=df[["change","pct_chg"]].apply(pd.to_numeric, errors='coerce')

    #print(df[df["pct_chg"]>9])

    ##df2 = pd.to_numeric(df2.pct_chg, errors='coerce' )

    #print(df2)

    ##获取基本数据
    #data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,market,name,area,industry,list_date')

    #data.set_index('ts_code',inplace=True)

    #df = pro.query('daily_basic', ts_code='', trade_date='20190307',fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,total_share,float_share,free_share,total_mv,circ_mv,close')

    #df.set_index('ts_code',inplace=True)

    #mix=pd.merge(data, df, how='outer', on=None,  

    #        left_index=True, right_index=True, sort=False,  

    #        suffixes=('_x', '_y'), copy=True, indicator=False)


    #print(mix)




    #data.to_csv("zmc.csv")



    sdads=1


dtypes = {
        'open':                                     'float16',
        'high':                                'float16',
        'low':                                      'float16',
        'close':                                 'float16',
        'pre_close':                               'float16',
        'change':                                     'float16',
        'pct_chg':                                     'float16',
        'vol':                                                  'float32',
        'amount':                                               'float32'
        }

def feature_env_codeanddate():

    df_all=pd.read_csv("savetest.csv",index_col=0,header=0)
    
    #df[["open","high","low" ,"close" ,"pre_close" ,"change","pct_chg","vol","amount"]]=df[["change","pct_chg"]].apply(pd.to_numeric, errors='coerce')

    #df_all["newtest"]=0

    df_empty = pd.DataFrame(columns=['yesterday1','yesterday2','yesterday3','yesterday4','yesterday5', 'tomorrow'])

    counter=0

    for cur_ts_code,group in df_all.groupby('ts_code'):

        print(cur_ts_code)

        bufferpct_chg=group["pct_chg"].reset_index(drop=True)

        #取得正确index的group数据，并且把index保存成一列数据列，同时自身index变为从0开始的顺序序列
        bufferlist3=group["pct_chg"].reset_index()

        dropindex=bufferpct_chg.shape[0]
        if(dropindex<6):
            continue

        for i in range(5):
            cur_pct_chg=bufferpct_chg

            #print(bufferpct_chg)

            #dropindex=cur_pct_chg.shape[0]
            #if(dropindex<6):
            #    continue
            ii=i+1
            while ii>0:
                dropindex=cur_pct_chg.shape[0]    
                cur_pct_chg=cur_pct_chg.drop([dropindex-1,])

                ii-=1

            #新建空的行
            plusrow=bufferpct_chg.copy(deep=True)
            plusrow=plusrow[:(i+1)]
            plusrow[plusrow!=0]=0          

            cur_pct_chg=plusrow.append(cur_pct_chg,ignore_index=True)

            #合并 正确index数列 昨日数列 和明日数列
            bufferlist3=pd.concat([bufferlist3,cur_pct_chg],axis=1,ignore_index=True)



        #获得数据
        #yesterday_chg=group["pct_chg"].reset_index(drop=True)
        tomorrow_chg=bufferpct_chg.copy(deep=True)

        #删除昨日最后一行
        dropindex=tomorrow_chg.shape[0]
        #yesterday_chg=yesterday_chg.drop([dropindex-1,])

        #删除明日第一行
        tomorrow_chg=tomorrow_chg.drop([0])

        #新建空的一行
        plusrow=bufferpct_chg.copy(deep=True)
        plusrow=plusrow[:1]
        plusrow[plusrow!=0]=0

        ##第一行为0，将删除最后一行的昨日添加到0后面，同时清空index
        #yesterday_chg=plusrow.append(yesterday_chg,ignore_index=True)

        #将0添加到明日的最后一行，同时清空index
        tomorrow_chg=tomorrow_chg.append(plusrow,ignore_index=True)

        #取得正确index的group数据，并且把index保存成一列数据列，同时自身index变为从0开始的顺序序列
        #bufferlist3=group["pct_chg"].reset_index()

        #合并 正确index数列 昨日数列 和明日数列
        bufferlist3=pd.concat([bufferlist3,tomorrow_chg],axis=1,ignore_index=True)



        #重新将index回设成df_all的index，并且删掉从0开始的index
        bufferlist3.set_index([0], drop=True, append=False, inplace=True, verify_integrity=False) 


        #删除本身自己的当日数据
        bufferlist3.drop([1],axis=1,inplace=True)


        #重命名列名
        bufferlist3.columns = ['yesterday1','yesterday2','yesterday3','yesterday4','yesterday5', 'tomorrow']


        #添加到集合
        df_empty=df_empty.append(bufferlist3)

        #print(df_empty)

        #bufferlist3.drop('pct_chg',axis=1, inplace=True)

        #df_all[df_all['ts_code']==cur_ts_code]['newtest']=1
        #group["newtest"]=bufferlist

        #print(bufferlist)
        #print(bufferlist2)

        #starter=0
        #last_pct=0
        #cur_pct=0
        #for x in range(len(group.index)):
        #    last2_pct=last_pct
        #    last_pct=cur_pct
        #    cur_pct=group["pct_chg"].iloc[x]

        #    group["newtest"].iloc[x]=last_pct


        #print(group)
        #break
        #if(counter==2):
        #    break

        #counter+=1
        fed=1

    #将生成好的乱序但是index对应数据行正确的数据，左连接到df_all中
    df_all=df_all.join(df_empty,how='left', lsuffix='_caller', rsuffix='_other')

    #grouped=df_all.groupby(['ts_code'])

    #print(grouped)
    #pd.set_option('display.max_rows', 1000)  # 设置显示最大行
    #print(df_all)

    df_all.to_csv("zzz2test.csv")
    dwdw=1

def feature_env_codeanddate2():

    df_all=pd.read_csv("savetest.csv",index_col=0,header=0)
    
    #df[["open","high","low" ,"close" ,"pre_close" ,"change","pct_chg","vol","amount"]]=df[["change","pct_chg"]].apply(pd.to_numeric, errors='coerce')

    #df_all["newtest"]=0

    df_empty = pd.DataFrame(columns=['amount_high','amount_low','amount_avg','high10','low10', 'yeaterday_chg','tomorrow_open'])

    counter=0

    for cur_ts_code,group in df_all.groupby('ts_code'):

        print(cur_ts_code)
        #取amount列
        buffer_amount=group["amount"].reset_index(drop=True)
        #数据量过小丢弃
        dropindex=buffer_amount.shape[0]
        if(dropindex<6):
            continue

        #取得正确index的group数据，并且把index保存成一列数据列，同时自身index变为从0开始的顺序序列
        bufferlist3=group["amount"].reset_index()

        #复制一份list
        bufferamount_sum=bufferlist3

        for i in range(1,10):
            bufferpct_2=buffer_amount.shift(i)
            bufferamount_sum=pd.concat([bufferamount_sum,bufferpct_2],axis=1,ignore_index=True)

        #print(bufferamount_sum)

        #计算10日量最大最小平均
        buffer4=bufferamount_sum[range(2,11)]
        high10=buffer4.max(axis=1)
        low10=buffer4.min(axis=1)
        mean10=buffer4.mean(axis=1)
        buffer4['high10']=high10
        buffer4['low10']=low10
        buffer4['mean10']=mean10

        buffer4.drop(range(2,11),axis=1,inplace=True)

        #取明日数据列
        buffer_amount=group["open"].reset_index(drop=True)
        tomorrow_open=buffer_amount.shift(-1)

        #取昨日数据列
        buffer_amount=group["pct_chg"].reset_index(drop=True)
        yeaterday_chg=buffer_amount.shift(1)

        #取10日最高和最低
        bufferhigh=group["high"].reset_index(drop=True)
        bufferhigh_sum=bufferhigh.shift(1)
        for i in range(2,9):
            bufferpct_2=bufferhigh.shift(i)
            bufferhigh_sum=pd.concat([bufferhigh_sum,bufferpct_2],axis=1,ignore_index=True)

        high10=bufferhigh_sum.max(axis=1)
        #print(high10)

        #取10日最高和最低
        bufferlow=group["low"].reset_index(drop=True)
        bufferhigh_sum=bufferlow.shift(1)
        for i in range(2,9):
            bufferpct_2=bufferlow.shift(i)
            bufferhigh_sum=pd.concat([bufferhigh_sum,bufferpct_2],axis=1,ignore_index=True)

        #print(bufferhigh_sum)
        low10=bufferhigh_sum.min(axis=1)
        #print(low10)



        #合并 正确index数列 昨日数列 和明日数列
        bufferlist3=pd.concat([bufferlist3,buffer4,high10,low10,yeaterday_chg,tomorrow_open],axis=1,ignore_index=True)

        #print(bufferlist3)

        #重新将index回设成df_all的index，并且删掉从0开始的index
        bufferlist3.set_index([0], drop=True, append=False, inplace=True, verify_integrity=False) 

        #删除本身自己的当日数据
        bufferlist3.drop([1],axis=1,inplace=True)

        #print(bufferlist3)

        #重命名列名
        bufferlist3.columns = ['amount_high','amount_low','amount_avg','high10','low10', 'yeaterday_chg','tomorrow_open']

        #print(bufferlist3)

        #添加到集合
        df_empty=df_empty.append(bufferlist3)

        #print(df_empty)

        #bufferlist3.drop('pct_chg',axis=1, inplace=True)

        #df_all[df_all['ts_code']==cur_ts_code]['newtest']=1
        #group["newtest"]=bufferlist

        #print(bufferlist)
        #print(bufferlist2)

        #starter=0
        #last_pct=0
        #cur_pct=0
        #for x in range(len(group.index)):
        #    last2_pct=last_pct
        #    last_pct=cur_pct
        #    cur_pct=group["pct_chg"].iloc[x]

        #    group["newtest"].iloc[x]=last_pct


        #print(group)
        #break
        #if(counter==2):
        #    break

        #counter+=1
        fed=1

    #将生成好的乱序但是index对应数据行正确的数据，左连接到df_all中
    df_all=df_all.join(df_empty,how='left', lsuffix='_caller', rsuffix='_other')

    #grouped=df_all.groupby(['ts_code'])

    #print(grouped)
    #pd.set_option('display.max_rows', 1000)  # 设置显示最大行
    #print(df_all)

    df_all.to_csv("zzz2018test.csv")
    dwdw=1

def feature_env_codeanddate3(year):

    bufferstring='savetest'+year+'.csv'

    df_all=pd.read_csv(bufferstring,index_col=0,header=0)
    #df_all=pd.read_csv(bufferstring,index_col=0,header=0,nrows=100000)
    
    df_all.drop(['change','vol'],axis=1,inplace=True)
    

    #明日幅度
    df_all['tomorrow_chg']=df_all.groupby('ts_code')['pct_chg'].shift(-1)
    df_all['tomorrow_chg_rank']=df_all.groupby('trade_date')['tomorrow_chg'].rank(pct=True)
    df_all['tomorrow_chg_rank']=df_all['tomorrow_chg_rank']*9.9//1
    #是否停
    df_all['high_stop']=0
    df_all.loc[df_all['pct_chg']>9,'high_stop']=1
    df_all.loc[(df_all['pct_chg']<5.5) & (4.5<df_all['pct_chg']),'high_stop']=1


    #真实价格范围
    df_all['price_real_rank']=df_all.groupby('trade_date')['pre_close'].rank(pct=True)
    df_all['price_real_rank']=df_all['price_real_rank']*10//1
    #1日
    df_all['chg_rank']=df_all.groupby('trade_date')['pct_chg'].rank(pct=True)
    df_all['chg_rank']=df_all['chg_rank']*10//1

    #3日
    xxx=df_all.groupby('ts_code')['chg_rank'].rolling(3).sum().reset_index()
    xxx.set_index(['level_1'], drop=True, append=False, inplace=True, verify_integrity=False)
    xxx.drop(['ts_code'],axis=1,inplace=True)

    df_all=df_all.join(xxx, lsuffix='_1', rsuffix='_3')

    df_all['chg_rank_3']=df_all.groupby('trade_date')['chg_rank_3'].rank(pct=True)
    df_all['chg_rank_3']=df_all['chg_rank_3']*10//1

    #print(df_all)

    #10日均量
    xxx=df_all.groupby('ts_code')['amount'].rolling(10).mean().reset_index()
    xxx.set_index(['level_1'], drop=True, append=False, inplace=True, verify_integrity=False)
    xxx.drop(['ts_code'],axis=1,inplace=True)
    df_all=df_all.join(xxx, lsuffix='_1', rsuffix='_10')

    #当日量占比
    df_all['pst_amount']=df_all['amount_1']/df_all['amount_10']
    df_all.drop(['amount_1','amount_10'],axis=1,inplace=True)
    #当日量排名
    df_all['pst_amount_rank']=df_all.groupby('trade_date')['pst_amount'].rank(pct=True)
    df_all['pst_amount_rank']=df_all['pst_amount_rank']*10//1

    #计算三种比例rank
    dolist=['open','high','low']

    for curc in dolist:
        buffer=((df_all[curc]-df_all['close'])*100)/df_all['close']
        df_all[curc]=buffer
        df_all[curc]=df_all.groupby('trade_date')[curc].rank(pct=True)
        df_all[curc]=df_all[curc]*10//1




    df_all.drop(['close','pre_close','pct_chg','pst_amount'],axis=1,inplace=True)
    #暂时不用的列
    df_all=df_all[df_all['high_stop']==0]
    #'tomorrow_chg'
    df_all.drop(['high_stop'],axis=1,inplace=True)



    df_all.dropna(axis=0,how='any',inplace=True)

    print(df_all)
    df_all=df_all.reset_index(drop=True)

    df_all.to_csv('ztrain'+year+'.csv')
    dwdw=1




def feature_env_2_old():
    
    train_data=pd.read_csv("zzztest.csv",index_col=0,header=0)

    dolist=['pct_chg','tomorrow','yesterday']

    for singlecol in dolist:

        buffer=(train_data[singlecol]+10)//1
        buffer[buffer>20]=20
        buffer[buffer<0]=0
        train_data[singlecol]=buffer

    ##将明日正则化(拟改用rank或正太分布间隔)
    #train_data['tomorrow']=(train_data['tomorrow']+10.5)//1
    #train_data[train_data['tomorrow']>20]=20
    #train_data[train_data['tomorrow']<0]=0

    ##将昨日正则化
    #train_data['yesterday']=(train_data['yesterday']+10.5)//1
    #train_data[train_data['yesterday']>20]=20
    #train_data[train_data['yesterday']<0]=0

    dolist=['open','high','low','close']

    for singlecol in dolist:

        buffer=(((train_data[singlecol]-train_data['pre_close'])*100)/train_data['pre_close']+10)//1
        buffer[buffer>20]=20
        buffer[buffer<0]=0
        train_data[singlecol]=buffer

    ##最高正则化
    #buffer=(((train_data['high']-train_data['pre_close'])*100)/train_data['pre_close']+10)//1
    #buffer[buffer>20]=20
    #buffer[buffer<0]=0
    #train_data['high']=buffer
    ##train_data[train_data['high']>20]['high']=20
    ##train_data[train_data['high']<0]['high']=1

    ##最低正则化
    #buffer=(((train_data['low']-train_data['pre_close'])*100)/train_data['pre_close']+10)//1
    #buffer[buffer>20]=20
    #buffer[buffer<0]=0
    #train_data['low']=buffer

    ##测试正则化
    #train_data['close']=(((train_data['close']-train_data['pre_close'])*100)/train_data['pre_close']+10)//1
    #train_data[train_data['close']>20]['close']=20
    #train_data[train_data['close']<0]['close']=12


    #see=train_data[train_data['tomorrow']<-1]

    #pd.set_option('display.max_rows', 10000)  # 设置显示最大行

    #print(see)
    #print(train_data)
    #删除第一天和最后一天
    dropindex=train_data[train_data['trade_date']==20180102].index
    train_data.drop(dropindex,inplace=True)

    dropindex=train_data[train_data['trade_date']==20181228].index
    train_data.drop(dropindex,inplace=True)

    dropindex=train_data[train_data['pct_chg']>18].index
    train_data.drop(dropindex,inplace=True)

    print(train_data)

    print(train_data.describe())
    # 默认统计数值型数据每列数据平均值，标准差，最大值，最小值，25%，50%，75%比例。
    print(train_data.describe(include=['O']))
    # 统计字符串型数据的总数，取不同值数量，频率最高的取值。其中include参数是结果数据类型白名单，O代表object类型，可用info中输出类型筛选。

    print("Before", train_data.shape)

    train_data=train_data.reset_index(drop=True)
    train_data.to_csv("ztrain.csv")
    dwdwd=1

def feature_env_2(year):
    
    bufferstring='zzz'+year+'test.csv'

    train_data=pd.read_csv(bufferstring,index_col=0,header=0)

    listname=['high','low','close','pre_close','change','pct_chg','vol','amount']


    train_data[listname]=train_data.groupby('ts_code')[listname].shift(1)


    #删除无效值
    dropindex=train_data[train_data['high']==train_data['low']].index
    train_data.drop(dropindex,inplace=True)

    #昨日成交与前10日平均比例
    train_data['cur_amount_pct']=(train_data['amount']*10)/train_data['amount_avg']//1

    #昨日排名
    train_data['rank']=train_data.groupby('trade_date')['pct_chg'].rank(pct=True)
    train_data['rank']=train_data['rank']*200//10

    #df_empty = pd.DataFrame(columns=['price_rank'])

    #for cur_trade_date,group in train_data.groupby('trade_date'):
    #    print(cur_trade_date)

    #    #取得正确index的group数据，并且把index保存成一列数据列，同时自身index变为从0开始的顺序序列
    #    bufferlist3=group["pct_chg"].reset_index()

    #    #取pct_chg列
    #    buffer_pct_chg=group["pct_chg"].reset_index(drop=True)
        
    #    buffer_rank=buffer_pct_chg.rank(pct=True)
    #    buffer_rank=buffer_rank*200//10


    #    print(buffer_rank)

    #    #重新将index回设成df_all的index，并且删掉从0开始的index
    #    bufferlist3.set_index([0], drop=True, append=False, inplace=True, verify_integrity=False) 

    #    #删除本身自己的当日数据
    #    bufferlist3.drop([1],axis=1,inplace=True)

    #    #print(bufferlist3)

    #    #重命名列名
    #    bufferlist3.columns = ['amount_high','amount_low','amount_avg','high10','low10', 'tomorrow']

    #    #添加到集合
    #    df_empty=df_empty.append(bufferlist3)

    #    sefse=1

    #训练结果
    train_data['tomorrow']=(((train_data['tomorrow_open']-train_data['open'])*100)/train_data['open'])//1
    
    train_data.loc[train_data['tomorrow']>9,'tomorrow']=15
    train_data.loc[(train_data['tomorrow']<=9) & (5<train_data['tomorrow']),'tomorrow']=6
    train_data.loc[(train_data['tomorrow']<=5) & (2<train_data['tomorrow']),'tomorrow']=3
    train_data.loc[(train_data['tomorrow']<=2) & (0.5<train_data['tomorrow']),'tomorrow']=1
    train_data.loc[(train_data['tomorrow']<=0.5) & (0<train_data['tomorrow']),'tomorrow']=0
    train_data.loc[(train_data['tomorrow']<=0) & (-0.5<train_data['tomorrow']),'tomorrow']=0
    train_data.loc[(train_data['tomorrow']<=-0.5) & (-2<train_data['tomorrow']),'tomorrow']=-1
    train_data.loc[(train_data['tomorrow']<=-2) & (-5<train_data['tomorrow']),'tomorrow']=-3
    train_data.loc[(train_data['tomorrow']<=-5) & (-9<train_data['tomorrow']),'tomorrow']=-6
    train_data.loc[train_data['tomorrow']<=-9,'tomorrow']=-15

    #train_data[2<=train_data['tomorrow']<=5]=3
    #train_data[0<=train_data['tomorrow']<=2]=1
    #train_data[-2<=train_data['tomorrow']<=0]=-1
    #train_data[-5<=train_data['tomorrow']<=-2]=-3
    #train_data[-9<=train_data['tomorrow']<=-5]=-6
    #train_data[train_data['tomorrow']<-9]=-12

    #标准化
    dolist=['pct_chg']

    for singlecol in dolist:

        buffer=(train_data[singlecol])//1
        train_data[singlecol]=buffer

    #dolist=['pct_chg','tomorrow']

    #for singlecol in dolist:

    #    buffer=(train_data[singlecol]+10)//1
    #    buffer[buffer>20]=20
    #    buffer[buffer<0]=0
    #    train_data[singlecol]=buffer


    dolist=['high','low']

    for singlecol in dolist:

        buffer=(((train_data[singlecol]-train_data['pre_close'])*100)/train_data['pre_close'])//1
        buffer[buffer>10]=10
        buffer[buffer<-10]=-10
        train_data[singlecol]=buffer

    dolist=['open']

    for singlecol in dolist:

        buffer=(((train_data[singlecol]-train_data['close'])*100)/train_data['close'])//1
        buffer[buffer>10]=10
        buffer[buffer<-10]=-10
        train_data[singlecol]=buffer


    dolist=['high10','low10']

    for singlecol in dolist:

        buffer=(((train_data[singlecol]-train_data['close'])*100)/train_data['close'])//1
        buffer[buffer>25]=25
        buffer[buffer<-25]=-25
        train_data[singlecol]=buffer

    dolist=['pct_chg']

    for singlecol in dolist:

        buffer=train_data[singlecol]
        buffer[buffer>11]=11
        buffer[buffer<-11]=-11
        train_data[singlecol]=buffer
    
    train_data['low_down']=train_data['low']-train_data['pct_chg']
    train_data['high_up']=train_data['high']-train_data['pct_chg']

    #see=train_data[train_data['tomorrow']<-1]

    #pd.set_option('display.max_rows', 10000)  # 设置显示最大行

    #print(see)
    #print(train_data)
    #删除第一天和最后一天

    startdata=int(year)*10000+120
    enddata=int(year)*10000+1220

    dropindex=train_data[train_data['trade_date']<=startdata].index
    train_data.drop(dropindex,inplace=True)

    dropindex=train_data[train_data['trade_date']>=enddata].index
    train_data.drop(dropindex,inplace=True)

    dropindex=train_data[train_data['open']>=9].index
    train_data.drop(dropindex,inplace=True)



    print(train_data)

    print(train_data.describe())
    # 默认统计数值型数据每列数据平均值，标准差，最大值，最小值，25%，50%，75%比例。
    print(train_data.describe(include=['O']))
    # 统计字符串型数据的总数，取不同值数量，频率最高的取值。其中include参数是结果数据类型白名单，O代表object类型，可用info中输出类型筛选。

    print("Before", train_data.shape)

    train_data.dropna(inplace=True)
    train_data=train_data.reset_index(drop=True)

    bufferstring='ztrain'+year+'.csv'

    train_data.to_csv(bufferstring)
    dwdwd=1

def lgb_train(year):

    readstring='ztrain'+year+'.csv'

    train=pd.read_csv(readstring,index_col=0,header=0,nrows=10000)
    #train=pd.read_csv(readstring,index_col=0,header=0)
    train=train.reset_index(drop=True)
    train2=train.copy(deep=True)


    y_train = np.array(train['tomorrow'])
    train.drop(['tomorrow','ts_code','trade_date','pre_close','change','amount','vol','close','amount_high','amount_low','amount_avg','yeaterday_chg','tomorrow_open'],axis=1,inplace=True)
    #pre_close	change	pct_chg	vol	amount

    #open	high	low	close	pre_close	change	pct_chg	vol	amount	amount_high	amount_low	amount_avg	high10	low10	yeaterday_chg	tomorrow_open	cur_amount_pct	rank	tomorrow	low_down	high_up
    #ts_code	trade_date	open	high	low	close	pre_close	change	pct_chg	vol	amount	amount_high	amount_low	amount_avg	high10	low10	yeaterday_chg	tomorrow_open


    #lgb_model = joblib.load('gbm.pkl')

    #dsadwd=lgb_model.feature_importances_

    #pred_test = lgb_model.predict_proba(train)

    #data1 = pd.DataFrame(pred_test)

    #data1['mix']=0
    #multlist=[-15,-6,-3,-1,0,1,3,6,15]

    #for i in range(9):
    #    buffer=data1[i]*multlist[i]
    #    data1['mix']=data1['mix']+buffer

    #train2=train2.join(data1)
    
    #print(train2)
    #readstring='data'+year+'mixd.csv'
    #train2.to_csv(readstring)




    train_ids = train.index.tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(train_ids, y_train)

    train=train.values

    counter=0

    for train_index, test_index in skf.split(train_ids, y_train):
        
        X_fit, X_val = train[train_index],train[test_index]
        y_fit, y_val = y_train[train_index], y_train[test_index]

        lgb_model = lgb.LGBMClassifier(max_depth=-1,
                                       n_estimators=400,
                                       learning_rate=0.05,
                                       num_leaves=2**8-1,
                                       colsample_bytree=0.6,
                                       objective='multiclass', 
                                       num_class=21,
                                       n_jobs=-1)
                                   

        lgb_model.fit(X_fit, y_fit, eval_metric='multi_error',
                      eval_set=[(X_val, y_val)], 
                      verbose=100, early_stopping_rounds=100)
        
        joblib.dump(lgb_model,'gbm.pkl')


        lgb_model = joblib.load('gbm.pkl')

        pred_test = lgb_model.predict_proba(X_val)

        #np.set_printoptions(threshold=np.inf) 

        #pd.set_option('display.max_rows', 10000)  # 设置显示最大行
        #pd.set_option('display.max_columns', None)
        #print(pred_test)

        data1 = pd.DataFrame(pred_test)
        data1.to_csv('data1.csv')

        gc.collect()
        counter += 1    
        #Stop fitting to prevent time limit error
        if counter == 1 : break


    X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)

def lgb_train_2(year):

    readstring='ztrain'+year+'.csv'

    #train=pd.read_csv(readstring,index_col=0,header=0,nrows=10000)
    train=pd.read_csv(readstring,index_col=0,header=0)
    train=train.reset_index(drop=True)
    train2=train.copy(deep=True)


    y_train = np.array(train['tomorrow_chg_rank'])
    train.drop(['tomorrow_chg','tomorrow_chg_rank','ts_code','trade_date'],axis=1,inplace=True)


    lgb_model = joblib.load('gbm.pkl')

    dsadwd=lgb_model.feature_importances_

    pred_test = lgb_model.predict_proba(train)

    data1 = pd.DataFrame(pred_test)

    data1['mix']=0
    multlist=[-10,-3,-2,-1,0,0,1,2,3,10]

    for i in range(10):
        buffer=data1[i]*multlist[i]
        data1['mix']=data1['mix']+buffer

    train2=train2.join(data1)
    
    print(train2)
    readstring='data'+year+'mixd.csv'
    train2.to_csv(readstring)




    train_ids = train.index.tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
    skf.get_n_splits(train_ids, y_train)

    train=train.values

    counter=0

    for train_index, test_index in skf.split(train_ids, y_train):
        
        X_fit, X_val = train[train_index],train[test_index]
        y_fit, y_val = y_train[train_index], y_train[test_index]

        lgb_model = lgb.LGBMClassifier(max_depth=-1,
                                       n_estimators=400,
                                       learning_rate=0.05,
                                       num_leaves=2**8-1,
                                       colsample_bytree=0.6,
                                       objective='multiclass', 
                                       num_class=21,
                                       n_jobs=-1)
                                   

        lgb_model.fit(X_fit, y_fit, eval_metric='multi_error',
                      eval_set=[(X_val, y_val)], 
                      verbose=100, early_stopping_rounds=100)
        
        joblib.dump(lgb_model,'gbm.pkl')


        lgb_model = joblib.load('gbm.pkl')

        pred_test = lgb_model.predict_proba(X_val)

        #np.set_printoptions(threshold=np.inf) 

        #pd.set_option('display.max_rows', 10000)  # 设置显示最大行
        #pd.set_option('display.max_columns', None)
        #print(pred_test)

        data1 = pd.DataFrame(pred_test)
        data1.to_csv('data1.csv')

        gc.collect()
        counter += 1    
        #Stop fitting to prevent time limit error
        if counter == 1 : break


    X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)

def show_start():
    showsource=pd.read_csv('data2018mixd.csv',index_col=0,header=0)
    databuffer=showsource['trade_date'].unique()

    for curdata in databuffer:

        cur_show=showsource[showsource["trade_date"]==curdata]

        b=cur_show.sort_values(by="mix" , ascending=False) 

        x_axis=range(len(b))
        y_axis=b['tomorrow_chg']

        show(x_axis,y_axis,title=curdata)

        adwda=1


def show(x_axis,y_axis,x_label="xlabel",y_label="ylabel ",title="title",x_tick="",y_tick="",colori="blue"):
        plt.figure(figsize=(19, 11))
        plt.scatter(x_axis, y_axis,s=8)
        #plt.xlim(30, 160)
        #plt.ylim(5, 50)
        #plt.axis()
    
        plt.title(title,color=colori)
        plt.xlabel("x_label")
        plt.ylabel("y_label")

        if(x_tick!=""or y_tick!=""):
            plt.xticks(x_axis,x_tick)
            plt.yticks(y_axis,y_tick)

        #plt.pause(2)
        plt.show()



def get_date_feature():

    #获取基于日期的特征

    #读取token
    f = open('token.txt')
    token = f.read()     #将txt文件的所有内容读入到字符串str中
    f.close()


    pro = ts.pro_api(token)


def get_code_feature():

    #获取基于代码的特征

    #读取token
    f = open('token.txt')
    token = f.read()     #将txt文件的所有内容读入到字符串str中
    f.close()


    pro = ts.pro_api(token)


if __name__ == '__main__':

    #df_all_first=pd.read_csv('savetest_all.csv',index_col=0,header=0)
    #print(df_all_first)

    #year=2013

    #readstring='savetest'+str(year)+'.csv'
    #df_all_first=pd.read_csv(readstring,index_col=0,header=0)
    #print(df_all_first)
    #for i in range(2):
    #    year=2014+i
    #    readstring='savetest'+str(year)+'.csv'
    #    df_all_sec=pd.read_csv(readstring,index_col=0,header=0)
    #    df_all_first=df_all_first.append(df_all_sec)
    #    print(df_all_first)

    #df_all_first=df_all_first.reset_index(drop=True)

    #df_all_first.to_csv('savetest_all.csv')

    #adwdd=ts.get_k_data("603999",start="2018-10-10", end="2018-12-08")

    #获取历史信息
    #HistoryDataGet(Datas=10)
    #Get_AllkData()
    #CSZL_CodelistToDatelist()

    show_start()

    get_codeanddate_feature()


    #feature_env_codeanddate2()
    feature_env_codeanddate3('2019')


    #feature_env_2('2018')

    lgb_train_2('2019')

    #feature_env_codeanddate()




    #pro.trade_cal(exchange='', start_date='20180101', end_date='20181231')
    #df = pro.fina_mainbz(ts_code='000627.SZ', period='20171231', type='P')

    Get_DateBased_Feature()


    CSZL_HistoryDB_Read()

    CSZL_History_Read()



    anafirsttest()

    #CSZLmegaDisplay.close()





    