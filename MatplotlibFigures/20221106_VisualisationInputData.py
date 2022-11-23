#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
import numpy as np
from csvpy import *
from sklearn import linear_model

from cProfile import label
from sklearn.model_selection import train_test_split
from shapely.geometry import LineString
import math
import sys

sys.path.insert(0, 'D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/Python/Maschine Learning')

from plotpy import *

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


if __name__ == '__main__':
    #INPUT
    inputfilepath = "./AlSi10Mg_collected_data.csv"
    invcol =[5,6,4] #First two numbers represent the read ouf col as input value and thrid number is the desired value (relative density)
    randomstate= 42
    trainingdatasize= 0.8
    scoring = 'neg_mean_absolute_percentage_error'#,'neg_root_mean_squared_error']
    T = 0.2
    X_pre_lin= np.linspace(250,800, 10)
    f1vt, u1 = np.meshgrid(X_pre_lin,1)
    f1v= f1vt.T

    idf=readcsvcol(inputfilepath,invcol)

    colheadersidf=list(idf.columns.values) # gets a list of df header strings

    #Spilts the Pandas dataframe in to numpy array Inputs X and result y 
    Xm = idf.iloc[:,0:len(invcol)-1].values
    ym = idf.iloc[:,len(invcol)-1:len(invcol)].values.ravel() #ravel funktion converts the array to a (n,) array instead of an (n,1) vector
    path = 'D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/03_methodology/fig/InputDataVisualization'
    cp1a =10
    



    #Divide data in TRAINING DATA and TEST DATA
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=trainingdatasize, random_state=randomstate, shuffle=True)
    
    #Train a given model
    model = linear_model.LinearRegression()
    model_sel = model.fit(Xm,ym)
    print("Coefficients",model_sel.coef_[0])
    print("Bias",model_sel.intercept_)

    
    
    #CHANGE FONT TYPE
    plt.rc('font',family='Linux Biolinum')
    #CHANGE FONT SIZES
    # plt.rc('font', size=12, weight='bold') #controls default text size
    #plt.rc('font', size=12) #controls default text size
    plt.rc('axes', titlesize=14) #fontsize of the title
    plt.rc('axes', labelsize=11) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=11) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=11) #fontsize of the y tick labels
    plt.rc('legend', fontsize=11) #fontsize of the legend
    #fig1 = plt.figure()
    # fig1 = plt.figure(figsize=(cm2inch(16.5,12))) # Halfpage
    fig = plt.figure(figsize=(cm2inch(7.5,7.5))) 


    ax = plt.axes(projection='3d')
    maxVal =np.max(ym)
    minVal =np.min(ym)
    sc =ax.scatter(Xm.T[0], Xm.T[1], ym,  linewidth=0.5, edgecolors='black',label='Input Data Points');
    # sc.set_clim(minVal,maxVal) #used to set colorbounds of the colormap
    ax.legend()
    # cb=plt.colorbar(sc,ticks=np.linspace(minVal,maxVal,cp1a),format='%.2f')
    # fig.suptitle('3D Visualisation of Input Data Points', fontweight ="bold")
    ax.set_xlabel(colheadersidf[0])
    ax.set_ylabel(colheadersidf[1])
    ax.set_zlabel(colheadersidf[len(colheadersidf)-1]);
    # cb.set_label(colheadersidf[len(colheadersidf)-1]);
    plt.savefig(path+'.pdf', format='pdf', bbox_inches='tight') # saved as eps for high quality pictures
    # set up axes

    
    # plt.title("Linear Regression")
    #plt.savefig("linechart_Duration_env1.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
    # plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/02_TheoreticalBackground/fig/MachineLearning/LinearRegressionChart.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
    # plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/02_TheoreticalBackground/fig/MachineLearning/LinearRegressionChart.svg", format='svg',bbox_inches='tight') # saved as eps for high quality pictures
    
    # plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/02_TheoreticalBackground/fig/MachineLearning/LinearRegressionChart.pdf", format='pdf',bbox_inches='tight') # saved as eps for high quality pictures
    plt.show()