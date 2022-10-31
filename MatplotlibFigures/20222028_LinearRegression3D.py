#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
import numpy as np
from csvpy import *
from sklearn import linear_model

from cProfile import label
from sklearn.model_selection import train_test_split


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)



if __name__ == '__main__':
    #INPUT
    inputfilepath = "./Thesis_Data_Set.csv"
    invcol =[5,6,4] #First two numbers represent the read ouf col as input value and thrid number is the desired value (relative density)
    randomstate= 42
    trainingdatasize= 0.8
    scoring = 'neg_mean_absolute_percentage_error'#,'neg_root_mean_squared_error']
    T = 0.2
    X1_lin= np.linspace(300,800, 10)
    X2_lin= np.linspace(1000,2000, 10)
    f1vt, u1 = np.meshgrid(X1_lin,1)
    f1v= f1vt.T

    idf=readcsvcol(inputfilepath,invcol)

    colheadersidf=list(idf.columns.values) # gets a list of df header strings

    #Spilts the Pandas dataframe in to numpy array Inputs X and result y 
    X = idf.iloc[:,0:len(invcol)-1].values
    y = idf.iloc[:,len(invcol)-1:len(invcol)].values.ravel() #ravel funktion converts the array to a (n,) array instead of an (n,1) vector



    


    #Divide data in TRAINING DATA and TEST DATA
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=trainingdatasize, random_state=randomstate, shuffle=True)
    
    #Train a given model
    model = linear_model.LinearRegression()
    model_sel = model.fit(X,y)

    X1_pre = np.asarray(X1_lin)
    X2_pre = np.asarray(X2_lin)
    print("X1_Predict",X1_pre)
    print("X1_Predict shape",X1_pre.shape)
    #predict the points
    
    f1min = 300
    f1max = 800
    f1num = 100
    f2min = 1000
    f2max = 2000
    f2num = 100
    #creates a set of values according to the input parameters
    f1 = np.linspace(f1min, f1max, f1num)
    f2 = np.linspace(f2min, f2max, f2num)
    # print("f1",f1)
    # print("shape f1",f1.shape)
    # print("f2",f2)
    #Transforms the the lines to vectors for further processing
    f1vt, u1 = np.meshgrid(f1,1)
    f2vt, u2 = np.meshgrid(f2,1)
    f1v= f1vt.T
    f2v= f2vt.T
    # print("\n")
    # print(f1vt)
    # print(u1)
    #Fill an numpy X array in the right form
    numrows=f1num*f2num
    X_pre_o = np.zeros((numrows,2))
    l = 0
    for i in range(f1num):
        for k in range(f2num):
            X_pre_o[l][0]=f1v[i]
            X_pre_o[l][1]=f2v[k]
            l+=1
   
    y_pred = model.predict(X_pre_o)


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
    # fig1 = plt.figure(figsize=(cm2inch(16.5,12))) #15,11
    fig1 = plt.figure(figsize=(cm2inch(8.25,8.25))) #15,11

    # set up axes

    ax1 = plt.axes(projection='3d')
    #plt.grid(linestyle='--', linewidth=1)

    # font = {'family' : 'normal',
    #         'weight' : 'bold',
    #         'size'   : 12}
    # plt.rc('font', **font)
    #plt.rcParams.update({'font.size': 12})



    sc =ax1.scatter(X.T[0],X.T[1], y,  linewidth=0.5, edgecolors='black',label='Showcase Data Set')
    plt.plot(X_pre_o.T[0],X_pre_o.T[1], y_pred, color = "r", label='Regression Model')




    #plt.plot([180000,(2*180000),(3*180000),(4*180000),(5*180000),(6*180000),(7*180000),(8*180000),(9*180000),(10*180000)],[90.63,96.45,104.27,111.47,120.15,128.42,134.09,142.21,151.09,158.38] , linestyle='-', marker='o', color='green', label='Greedy Search')
    ax1.legend()
    #ax1.legend([line1, line2, line3], ['label1', 'label2', 'label3'])
    plt.xlabel("Laser Power [W]")
    plt.ylabel("Relative Density [%]")
    # plt.title("Linear Regression")
    #plt.savefig("linechart_Duration_env1.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
    plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/02_TheoreticalBackground/fig/MachineLearning/LinearRegressionChart3D.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
    # plt.savefig("linechart_duration_lp.svg", format='svg',bbox_inches='tight' )
    plt.show()