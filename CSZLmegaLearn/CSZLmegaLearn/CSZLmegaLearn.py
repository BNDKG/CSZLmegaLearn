#coding=utf-8


import pandas as pd
import numpy as np
import os
import random
import copy

import tushare as ts

#正则表达式
import re

import datetime
import time
import random

#自写的展示类
from CSZLmegaDisplay import CSZLmegaDisplay


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

    date=pro.query('trade_cal', start_date='20180102', end_date='20181230')

    date=date[date["is_open"]==1]
    get_list=date["cal_date"]

    df_all=pro.daily(trade_date="20180101")

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

    df_all.to_csv("savetest.csv")





    #df = pro.query('adj_factor',  trade_date='20180315')

    #df = pro.shibor(start_date='20100101', end_date='20101101')

    df[["change","pct_chg"]]=df[["change","pct_chg"]].apply(pd.to_numeric, errors='coerce')

    print(df[df["pct_chg"]>9])

    #df2 = pd.to_numeric(df2.pct_chg, errors='coerce' )

    print(df2)

    #获取基本数据
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,market,name,area,industry,list_date')

    data.set_index('ts_code',inplace=True)

    df = pro.query('daily_basic', ts_code='', trade_date='20190307',fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,total_share,float_share,free_share,total_mv,circ_mv,close')

    df.set_index('ts_code',inplace=True)

    mix=pd.merge(data, df, how='outer', on=None,  

            left_index=True, right_index=True, sort=False,  

            suffixes=('_x', '_y'), copy=True, indicator=False)


    print(mix)




    data.to_csv("zmc.csv")


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

    df_all["newtest"]=0

    for cur_ts_code,group in df_all.groupby('ts_code'):

        print(cur_ts_code)



        bufferlist=group["pct_chg"].reset_index(drop=True)
        bufferlist2=bufferlist

        bufferlist=bufferlist.drop(index=[0])

        dropindex=bufferlist2.shape[0]
        bufferlist2=bufferlist2.drop(index=[dropindex-1,])
        plusrow=bufferlist[:0]
        plusrow["pct_chg"]=0

        bufferlist=plusrow.append(bufferlist,ignore_index=True)
        bufferlist2=bufferlist2.append(plusrow,ignore_index=True)

        df_all[df_all['ts_code']==cur_ts_code]['newtest']=1
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
        print(df_all)

        fed=1

    
    #grouped=df_all.groupby(['ts_code'])

    #print(grouped)

    print(df_all)

    df_all.to_csv("zzztest.csv")
    dwdw=1
    

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

    #adwdd=ts.get_k_data("603999",start="2018-10-10", end="2018-12-08")

    #获取历史信息
    #HistoryDataGet(Datas=10)
    #Get_AllkData()
    #CSZL_CodelistToDatelist()

    feature_env_codeanddate()

    #get_codeanddate_feature()


    #pro.trade_cal(exchange='', start_date='20180101', end_date='20181231')
    #df = pro.fina_mainbz(ts_code='000627.SZ', period='20171231', type='P')

    Get_DateBased_Feature()


    CSZL_HistoryDB_Read()

    CSZL_History_Read()



    anafirsttest()

    #CSZLmegaDisplay.close()





    