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

def show_all_rate():
    showsource=pd.read_csv('data2017mixd.csv',index_col=0,header=0)
    databuffer=showsource['trade_date'].unique()

    changer=[]
    for curdata in databuffer:

        cur_show=showsource[showsource["trade_date"]==curdata]
        b=cur_show.sort_values(by="mix" , ascending=False)
        #b=cur_show.sort_values(by="9" , ascending=True)
        #d=b.head(10)
        #e=d.sort_values(by="mix" , ascending=True)
        

        #b=cur_show[cur_show['mix']>0.40]
        average=b.head(1)['tomorrow_chg'].mean()
        changer.append(average)

        adwda=1


    days2,show=standard_show(changer,day_interval=1)

    return days2,show




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

    date=pro.query('trade_cal', start_date='20180102', end_date='20190404')

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

    df_all.to_csv("savetest2018.csv")


    sdads=1

def feature_env_codeanddate3(year):

    bufferstring='savetest'+year+'.csv'

    df_all=pd.read_csv(bufferstring,index_col=0,header=0)
    #df_all=pd.read_csv(bufferstring,index_col=0,header=0,nrows=100000)
    #print(df_all)
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


def get_allchange():

    #读取token
    f = open('token.txt')
    token = f.read()     #将txt文件的所有内容读入到字符串str中
    f.close()

    pro = ts.pro_api(token)

    df = pro.index_daily(ts_code='000001.SH', start_date='20180117', end_date='20190404')
    df2 = pro.index_daily(ts_code='399006.SZ', start_date='20180117', end_date='20190404')
    
    b=df.sort_values(by="trade_date" , ascending=True)     
    b2=df2.sort_values(by="trade_date" , ascending=True)     

    changer=b['pct_chg']
    changer2=b2['pct_chg']

    days,show=standard_show(changer,day_interval=1)
    days2,show2=standard_show(changer2,day_interval=1)

    #standard_show(changer2,day_interval=1,label="自己2")



    return days,show,show2
    #plt.show()

def standard_show(changer,first_base_income=100000,day_interval=2,label="自己"):
    
    start_from=first_base_income
    show=[]
    for curchange in changer:
        start_from=start_from+(first_base_income/100/day_interval)*curchange
        show.append(start_from)

    #print(show)
    len_show=len(show)
    days=np.arange(1,len_show+1)

    fig=plt.figure(figsize=(6,3))


    #plt.show()

    return days,show

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




if __name__ == '__main__':

    #days,show1,show2=get_allchange()


    #get_codeanddate_feature()

    #feature_env_codeanddate3('2017')

    #lgb_train_2('2017')

    days,show3=show_all_rate()

    #plt.plot(days,show1,c='blue',label="000001")
    #plt.plot(days,show2,c='red',label="399006")
    plt.plot(days,show3,c='green',label="my model head10mean")

    plt.legend()

    plt.show()

    sdfsdfsf=1

    end=1




