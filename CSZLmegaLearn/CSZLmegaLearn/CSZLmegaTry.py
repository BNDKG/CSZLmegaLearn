#coding=utf-8

import pandas as pd
import numpy as np
import os
import random

import tushare as ts

import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold

#正则表达式
import re

import datetime
import time
import random

##自写的展示类
#from CSZLmegaDisplay import CSZLmegaDisplay
import gc
from sklearn.externals import joblib


#文件夹总路径
cwd = os.getcwd()

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

    nowTime=datetime.datetime.now()
    delta = datetime.timedelta(days=23)
    delta_one = datetime.timedelta(days=1)
    month_ago = nowTime - delta
    month_ago_next=month_ago+delta_one
    month_fst=month_ago_next.strftime('%Y%m%d')  
    month_sec=nowTime.strftime('%Y%m%d')  
    month_thd=month_ago.strftime('%Y%m%d')      

    date=pro.query('trade_cal', start_date=month_fst, end_date=month_sec)

    date=date[date["is_open"]==1]
    get_list=date["cal_date"]

    df_all=pro.daily(trade_date=month_thd)

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


    df_all['ts_code']=df_all['ts_code'].map(lambda x : x[:-3])
    df_all.drop(['vol','change'],axis=1,inplace=True)

    df_all.to_csv("real_buffer_2.csv")

    sdads=1

def real_get_change():

    df_history=pd.read_csv("real_buffer_2.csv",index_col=0,header=0)

    codelistbuffer=df_history['ts_code']
    codelistbuffer=codelistbuffer.unique()

    codelist=codelistbuffer.tolist()

    code_counter=0
    bufferlist=[]
    df_real=[]

    printcounter=0.0
    for curcode in codelist:

        curcode_str=str(curcode).zfill(6)
        bufferlist.append(curcode_str)
        code_counter+=1
        if(code_counter>=20):
            if(len(df_real)):
                df_real2=ts.get_realtime_quotes(bufferlist)
                df_real=df_real.append(df_real2)
            else:
                df_real=ts.get_realtime_quotes(bufferlist)
            bufferlist=[]            
            code_counter=0
            sleeptime=random.randint(50,99)
            time.sleep(sleeptime/200)
            print(printcounter/len(codelist))

        printcounter+=1

    #df_real=ts.get_realtime_quotes(['600839','000980','000981'])
    #df_real2=ts.get_realtime_quotes(['000010','600000','600010'])
    #df_real=df_real.append(df_real2)

    #print(df_real)
    #'tomorrow_chg'
    df_real.drop(['name','bid','ask','volume','b1_v','b2_v','b3_v','b4_v','b5_v','b1_p','b2_p','b3_p','b4_p','b5_p'],axis=1,inplace=True)
    df_real.drop(['a1_v','a2_v','a3_v','a4_v','a5_v','a1_p','a2_p','a3_p','a4_p','a5_p'],axis=1,inplace=True)
    df_real.drop(['time'],axis=1,inplace=True)

    df_real['amount'] = df_real['amount'].apply(float)
    df_real['amount']=df_real['amount']/1000

    #df[txt] = df[txt].map(lambda x : x[:-2])

    df_real['date']=df_real['date'].map(lambda x : x[:4]+x[5:7]+x[8:10])
    
    df_real['price'] = df_real['price'].apply(float)
    df_real['pre_close'] = df_real['pre_close'].apply(float)

    df_real['pct_chg']=(df_real['price']-df_real['pre_close'])*100/(df_real['pre_close'])


    df_real=df_real.rename(columns={'price':'close','date':'trade_date','code':'ts_code'})

    df_real.to_csv("real_buffer.csv")
    df_real=pd.read_csv("real_buffer.csv",index_col=0,header=0)
    
    
    cols = list(df_history)

    df_history=df_history.append(df_real,sort=False)

    df_history = df_history.ix[:, cols]


    df_history=df_history.reset_index(drop=True)
    #print(df_history)
    #print(df_real)
    df_history.to_csv("real_now.csv")

    dsfesf=1

def real_FE():

    bufferstring='real_now.csv'

    df_all=pd.read_csv(bufferstring,index_col=0,header=0)
    #df_all=pd.read_csv(bufferstring,index_col=0,header=0,nrows=100000)
    
    #df_all.drop(['change','vol'],axis=1,inplace=True)
    print(df_all)

    #是否停
    df_all['high_stop']=0
    df_all.loc[df_all['pct_chg']>9,'high_stop']=1
    df_all.loc[(df_all['pct_chg']<5.5) & (4.5<df_all['pct_chg']),'high_stop']=1

    df_all['price_real_rank']=df_all.groupby('trade_date')['pre_close'].rank(pct=True)
    df_all['price_real_rank']=df_all['price_real_rank']*10//1

    df_all['chg_rank']=df_all.groupby('trade_date')['pct_chg'].rank(pct=True)
    df_all['chg_rank']=df_all['chg_rank']*10//1


    xxx=df_all.groupby('ts_code')['chg_rank'].rolling(3).sum().reset_index()
    xxx.set_index(['level_1'], drop=True, append=False, inplace=True, verify_integrity=False)
    xxx.drop(['ts_code'],axis=1,inplace=True)

    df_all=df_all.join(xxx, lsuffix='_1', rsuffix='_3')

    df_all['chg_rank_3']=df_all.groupby('trade_date')['chg_rank_3'].rank(pct=True)
    df_all['chg_rank_3']=df_all['chg_rank_3']*10//1

    #print(df_all)


    xxx=df_all.groupby('ts_code')['amount'].rolling(10).mean().reset_index()
    xxx.set_index(['level_1'], drop=True, append=False, inplace=True, verify_integrity=False)
    xxx.drop(['ts_code'],axis=1,inplace=True)
    df_all=df_all.join(xxx, lsuffix='_1', rsuffix='_10')


    df_all['pst_amount']=df_all['amount_1']/df_all['amount_10']
    df_all.drop(['amount_1','amount_10'],axis=1,inplace=True)

    df_all['pst_amount_rank']=df_all.groupby('trade_date')['pst_amount'].rank(pct=True)
    df_all['pst_amount_rank']=df_all['pst_amount_rank']*10//1


    dolist=['open','high','low']

    for curc in dolist:
        buffer=((df_all[curc]-df_all['close'])*100)/df_all['close']
        df_all[curc]=buffer
        df_all[curc]=df_all.groupby('trade_date')[curc].rank(pct=True)
        df_all[curc]=df_all[curc]*10//1




    df_all.drop(['close','pre_close','pct_chg','pst_amount'],axis=1,inplace=True)

    df_all=df_all[df_all['high_stop']==0]

    df_all.drop(['high_stop'],axis=1,inplace=True)



    df_all.dropna(axis=0,how='any',inplace=True)

    print(df_all)
    df_all=df_all.reset_index(drop=True)

    df_all.to_csv('today_train.csv')
    dwdw=1

def real_lgb_predict():
    readstring='today_train.csv'

    #train=pd.read_csv(readstring,index_col=0,header=0,nrows=10000)
    train=pd.read_csv(readstring,index_col=0,header=0)
    train=train.reset_index(drop=True)
    train2=train.copy(deep=True)


    train.drop(['ts_code','trade_date'],axis=1,inplace=True)


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
    readstring='todaypredict.csv'
    train2.to_csv(readstring)

    dawd=1

def show_change1():

    show=pd.read_csv('todaypredict.csv',index_col=0,header=0)
    #datamax=show['trade_date'].max()
    datamax=20190403

    show=show[show['trade_date']==datamax]

    show=show[['ts_code','0','9','mix']]

    #ascending表示升降序
    b=show.sort_values(by="mix" , ascending=False) 
    c=show.sort_values(by="9" , ascending=False) 
    final_mix=b.head(20)
    final_9=c.head(20)

    pd.set_option('display.max_columns', None)
    print('综合成绩')
    print(final_mix)
    print('极限成绩')
    print(final_9)


    fsfef=1

def CSZL_TimeCheck():
    global CurHour
    global CurMinute



    CurHour=int(time.strftime("%H", time.localtime()))
    CurMinute=int(time.strftime("%M", time.localtime()))

    caltemp=CurHour*100+CurMinute

    #return True

    if (caltemp>=1455 and caltemp<=1500):
        return True
    else:
        return False  

if __name__ == '__main__':

    cur_date=datetime.datetime.now().strftime("%Y-%m-%d")

    change_flag=0
    while(True):
        date=datetime.datetime.now()
        day = date.weekday()
        if(day>6):   
            time.sleep(1000)
            continue
            dawd=5
        if(cur_date!=date.strftime("%Y-%m-%d")):
            change_flag=1
            cur_date=date
            #刚切到新的一天时候就下一下数据
            get_codeanddate_feature()
            time.sleep(10)            

        if(change_flag==1):

            if(CSZL_TimeCheck()):       

                #这个可能可以循环取
                real_get_change()

                real_FE()

                real_lgb_predict()

                show_change1()
                change_flag=0
                time.sleep(10000) 
        
        time.sleep(10)