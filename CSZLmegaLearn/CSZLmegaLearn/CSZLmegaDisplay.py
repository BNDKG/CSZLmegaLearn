#coding=utf-8
import matplotlib.pyplot as plt


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

        plt.xticks(x_axis,x_tick)
        plt.yticks(y_axis,y_tick)

        plt.pause(2)

    def close():
        plt.close()
    def clean():
        plt.clf()