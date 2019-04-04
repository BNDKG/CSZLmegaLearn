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

    date=pro.query('trade_cal', start_date='20190302', end_date='20190403')

    date=date[date["is_open"]==1]
    get_list=date["cal_date"]

    df_all=pro.daily(trade_date="20190301")

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

    ##读取token
    #f = open('token.txt')
    #token = f.read()     #将txt文件的所有内容读入到字符串str中
    #f.close()

    #pro = ts.pro_api(token)


    #df_history=pro.daily(trade_date="20190403")

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
            print("continue")
        fsfef=1
        sleeptime=random.randint(50,99)
        time.sleep(sleeptime/2000)


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
    




    #  属性:0：name，股票名字
    #1：open，今日开盘价
    #2：pre_close，昨日收盘价
    #3：price，当前价格
    #4：high，今日最高价
    #5：low，今日最低价
    #6：bid，竞买价，即“买一”报价
    #7：ask，竞卖价，即“卖一”报价
    #8：volumn，成交量 maybe you need do volumn/100
    #9：amount，成交金额（元 CNY）
    #10：b1_v，委买一（笔数 bid volume）
    #11：b1_p，委买一（价格 bid price）
    #12：b2_v，“买二”
    #13：b2_p，“买二”
    #14：b3_v，“买三”
    #15：b3_p，“买三”
    #16：b4_v，“买四”
    #17：b4_p，“买四”
    #18：b5_v，“买五”
    #19：b5_p，“买五”
    #20：a1_v，委卖一（笔数 ask volume）
    #21：a1_p，委卖一（价格 ask price）
    
    cols = list(df_history)

    df_history=df_history.append(df_real)

    df_history = df_history.ix[:, cols]


    df_history=df_history.reset_index(drop=True)
    #print(df_history)
    #print(df_real)
    df_history.to_csv("real_now.csv")

    dsfesf=1


def real_predict():

    bufferstring='real_now.csv'

    df_all=pd.read_csv(bufferstring,index_col=0,header=0)
    #df_all=pd.read_csv(bufferstring,index_col=0,header=0,nrows=100000)
    
    #df_all.drop(['change','vol'],axis=1,inplace=True)
    print(df_all)


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

    df_all.to_csv('ztrain'+year+'.csv')
    dwdw=1


if __name__ == '__main__':

    #get_codeanddate_feature()

    real_get_change()


    real_predict()