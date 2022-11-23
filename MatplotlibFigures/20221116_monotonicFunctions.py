#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axisartist.axislines import SubplotZero
import pandas as pd
import random
import math


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

if __name__ == '__main__':
   
    x = np.linspace(-6,6,1000000)
    # Relu function
    y_relu = x.clip(min=0)
    y_sigmoid=(1 / (1 + np.exp(-x)))

    #CHANGE FONT TYPE
    plt.rc('font',family='Linux Biolinum')
    #CHANGE FONT SIZES
    # plt.rc('font', size=12, weight='bold') #controls default text size
    #plt.rc('font', size=12) #controls default text size
    plt.rc('axes', titlesize=14) #fontsize of the title
    plt.rc('axes', labelsize=12) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
    plt.rc('legend', fontsize=12) #fontsize of the legend
    #fig1 = plt.figure()
    # fig1 = plt.figure(figsize=(cm2inch(16.5,12))) # Halfpage
    # fig1 = plt.figure(figsize=(cm2inch(15,6))) 

    # # set up axes
    # ax1 = SubplotZero(fig1,111)
    # fig1.add_subplot(ax1)

    rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
      "xtick.major.size" : 5, "ytick.major.size" : 5,}
with plt.rc_context(rc):
    fig, ax1 = plt.subplots(figsize=(cm2inch(15,8)))

    ax1.spines['left'].set_position('zero')
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    # make arrows
    ax1.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax1.get_yaxis_transform(), clip_on=False)
    ax1.plot((0), (1), ls="", marker="^", ms=10, color="k",
            transform=ax1.get_xaxis_transform(), clip_on=False)



    x_non = pd.DataFrame({"X":np.random.uniform(0,10,100)})
    x_non["Parabola"] = (x_non.X-5)**2
    x_non["Sin"] = np.sin(x_non.X/10*2*np.pi)
    x_non["Frac"] = (x_non.X-5)/((x_non.X-5)**2+1)
    x_non["Rand"] = np.random.uniform(-1,1,100)
    x_non["decre"]=np.exp(-x_non.X)
    x_non["Sq+"] = x_non.X*0.1
    print(x_non.head())

    ax1.scatter(x_non["X"], x_non["Rand"],linewidth=0.5, edgecolors='black',label=r'Random points  $\rho$ = 0.03')
    ax1.scatter(x_non["X"], x_non["decre"],linewidth=0.5, edgecolors='black',label=r'$e^{-x}$                    $\rho$ = -1')
    ax1.scatter(x_non["X"], x_non["Sq+"],linewidth=0.5, edgecolors='black',label=r'$x \cdot 0.1$               $\rho$ = 1')
    # plt.step([-4,-1,0,0,1,4],[0,0,0,1,1,1],linewidth=2, label='sgn(x)', linestyle='solid')
    # plt.step(x,y_relu,linewidth=2, label='ReLU(x)',linestyle='dotted')
    # plt.step(x,y_sigmoid,linewidth=2, label='sigmoid(x)',linestyle='dashed')

    # plt.xlim(-4, 4.05)
    # plt.ylim(0,4) 
    
    # plt.plot(0,d,'ro', label="0 d") 
    # plt.plot(-d/m,0,'ro', label = "-d/m, 0") 
    


    ax1.legend()
    #ax1.legend([line1, line2, line3], ['label1', 'label2', 'label3'])
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid()
    # plt.title("Linear Regression")
    #plt.savefig("linechart_Duration_env1.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
    # plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/02_TheoreticalBackground/fig/MachineLearning/activationFunctions.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
    plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/02_TheoreticalBackground/fig/DataPreparation/spearman.pdf", format='pdf',bbox_inches='tight') # saved as eps for high quality pictures
    # plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/02_TheoreticalBackground/fig/DataPreparation/spearman.svg", format='svg',bbox_inches='tight') # saved as eps for high quality pictures
    plt.show()