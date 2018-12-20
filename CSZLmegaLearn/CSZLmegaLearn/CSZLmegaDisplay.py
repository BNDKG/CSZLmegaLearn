#coding=utf-8
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
from scipy import stats, integrate

import seaborn as sns
from seaborn.axisgrid import FacetGrid




class CSZLmegaDisplay(object):
    """description of class"""


    def twodim(x_axis,y_axis,x_label="xlabel",y_label="ylabel ",title="title",x_tick="",y_tick=""):
      
        plt.scatter(x_axis, y_axis,s=1)
        #plt.xlim(30, 160)
        #plt.ylim(5, 50)
        #plt.axis()
    
        plt.title(title)
        plt.xlabel("x_label")
        plt.ylabel("y_label")

        if(x_tick!=""or y_tick!=""):
            plt.xticks(x_axis,x_tick)
            plt.yticks(y_axis,y_tick)

        plt.pause(2)
    def onedim(x,xticks,xvalue):
        ax=sns.set(color_codes=True)
        #test_plot=FacetGrid.set(xticks=np.arange(1,4,1))
        #x= np.random.normal(size=100)
        #x= np.random.random(size=10000)
        sns.distplot(x,bins=100)
        plt.xticks(xticks)
        plt.axvline(xvalue)
        plt.pause(2)

    def close():
        plt.close()
    def clean():
        plt.clf()