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
import seaborn as sns

#文件夹总路径
cwd = os.getcwd()

def feature_env_codeanddate3(year):

    bufferstring='savetest'+year+'.csv'

    df_all=pd.read_csv(bufferstring,index_col=0,header=0)
    #df_all=pd.read_csv(bufferstring,index_col=0,header=0,nrows=100000)
    
    df_all.drop(['change','vol'],axis=1,inplace=True)
    

    #明日幅度
    #tm1=df_all.groupby('ts_code')['pct_chg'].shift(-1)
    #tm2=df_all.groupby('ts_code')['pct_chg'].shift(-2)
    #tm3=df_all.groupby('ts_code')['pct_chg'].shift(-3)
    #df_all['tomorrow_chg']=((100+tm1)*(100+tm2)*(100+tm3)-1000000)/10000
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


    #6日
    xxx=df_all.groupby('ts_code')['chg_rank'].rolling(6).sum().reset_index()
    xxx.set_index(['level_1'], drop=True, append=False, inplace=True, verify_integrity=False)
    xxx.drop(['ts_code'],axis=1,inplace=True)

    df_all=df_all.join(xxx, lsuffix='', rsuffix='_6')

    df_all['chg_rank_6']=df_all.groupby('trade_date')['chg_rank_6'].rank(pct=True)
    df_all['chg_rank_6']=df_all['chg_rank_6']*10//1

    #10日
    xxx=df_all.groupby('ts_code')['chg_rank'].rolling(10).sum().reset_index()
    xxx.set_index(['level_1'], drop=True, append=False, inplace=True, verify_integrity=False)
    xxx.drop(['ts_code'],axis=1,inplace=True)

    df_all=df_all.join(xxx, lsuffix='', rsuffix='_10')

    df_all['chg_rank_10']=df_all.groupby('trade_date')['chg_rank_10'].rank(pct=True)
    df_all['chg_rank_10']=df_all['chg_rank_10']*10//1

    #3日
    xxx=df_all.groupby('ts_code')['chg_rank'].rolling(3).sum().reset_index()
    xxx.set_index(['level_1'], drop=True, append=False, inplace=True, verify_integrity=False)
    xxx.drop(['ts_code'],axis=1,inplace=True)

    df_all=df_all.join(xxx, lsuffix='', rsuffix='_3')

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
        buffer=((df_all[curc]-df_all['pre_close'])*100)/df_all['pre_close']
        df_all[curc]=buffer
        df_all[curc]=df_all.groupby('trade_date')[curc].rank(pct=True)
        df_all[curc]=df_all[curc]*10//1

    #加入昨日rank
    df_all['yesterday_open']=df_all.groupby('ts_code')['open'].shift(1)
    df_all['yesterday_high']=df_all.groupby('ts_code')['high'].shift(1)
    df_all['yesterday_low']=df_all.groupby('ts_code')['low'].shift(1)
    df_all['yesterday_pst_amount_rank']=df_all.groupby('ts_code')['pst_amount_rank'].shift(1)
    #加入前日open
    df_all['yesterday2_open']=df_all.groupby('ts_code')['open'].shift(2)

    df_all.drop(['close','pre_close','pct_chg','pst_amount'],axis=1,inplace=True)
    #暂时不用的列
    df_all=df_all[df_all['high_stop']==0]
    #'tomorrow_chg'
    df_all.drop(['high_stop'],axis=1,inplace=True)



    df_all.dropna(axis=0,how='any',inplace=True)

    #df_all[df_all['tomorrow_chg_rank']<9]['tomorrow_chg_rank']=0
    #df_all[df_all['tomorrow_chg_rank']>8]['tomorrow_chg_rank']=1
    df_all.loc[df_all['tomorrow_chg_rank']<9,'tomorrow_chg_rank']=0
    df_all.loc[df_all['tomorrow_chg_rank']>8,'tomorrow_chg_rank']=1

    df_all=df_all.reset_index(drop=True)

    df_all.to_csv('ztrain'+year+'.csv')
    dwdw=1

def lgb_train_2(year):

    readstring='ztrain'+year+'.csv'

    #train=pd.read_csv(readstring,index_col=0,header=0,nrows=10000)
    train=pd.read_csv(readstring,index_col=0,header=0)
    train=train.reset_index(drop=True)
    train2=train.copy(deep=True)

    

    y_train = np.array(train['tomorrow_chg_rank'])
    train.drop(['tomorrow_chg','tomorrow_chg_rank','ts_code','trade_date'],axis=1,inplace=True)

    #画数据的热力图
    #corrmat = train.corr()
    #f, ax = plt.subplots(figsize=(12, 9))
    #sns.heatmap(corrmat, vmax=.8, square=True);
    #plt.show()

    lgb_model = joblib.load('gbm.pkl')

    dsadwd=lgb_model.feature_importances_

    pred_test = lgb_model.predict_proba(train)

    data1 = pd.DataFrame(pred_test)

    #data1['mix']=0
    ##multlist=[-12,-5,-3,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0,0.25,0.5,0.75,1,1.5,2,3,5,12]
    #multlist=[-10,-3,-2,-1,0,0,1,2,3,10]

    #for i in range(10):
    #    buffer=data1[i]*multlist[i]
    #    data1['mix']=data1['mix']+buffer

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
                                       objective='binary',                                        
                                       n_jobs=-1)
                                   

        lgb_model.fit(X_fit, y_fit, eval_metric='auc',
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
    showsource=pd.read_csv('data2019mixd.csv',index_col=0,header=0)
    databuffer=showsource['trade_date'].unique()

    for curdata in databuffer:

        cur_show=showsource[showsource["trade_date"]==curdata]

        b=cur_show.sort_values(by="1" , ascending=False) 

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
        plt.xlabel("rank")
        plt.ylabel("chg_pct")

        if(x_tick!=""or y_tick!=""):
            plt.xticks(x_axis,x_tick)
            plt.yticks(y_axis,y_tick)

        #plt.pause(2)
        plt.show()
if __name__ == '__main__':

    show_start()

    feature_env_codeanddate3('2019')


    lgb_train_2('2019')