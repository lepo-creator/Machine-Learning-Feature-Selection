#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
import numpy as np
from csvpy import *
from sklearn import linear_model

from cProfile import label
from sklearn.model_selection import train_test_split
from shapely.geometry import LineString
import math


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def getIntersecctionPoint(m,d,P0):
    P0x=P0[0]
    P0y=P0[1]
    print("P0x",P0x)
    print("P0y",P0y)
    Px= (m*(m*P0x-P0y))/(m**2+1)
    Py=-(m*P0x-P0y)/(m**2+1)

    Px2= (-m*d)/(m**2+1)
    Py2=(d)/(m**2+1)
    return (Px,Py),(Px2,Py2)

def solveLotsystem(m,d,P):
    Px = P[0]
    Py = P[1]
    left_side = np.array([[-d/m, -m],[-d,-1]])
    right_side = np.array([Px,(Py-d)])
    sol = np.linalg.inv(left_side).dot(right_side)

    Psx= (-d/m)*sol[0]
    Psy= (-d)*sol[0]+d

    X_s= np.array([Px,Psx])
    y_s= np.array([Py,Psy])
    return X_s ,y_s

if __name__ == '__main__':
    #INPUT
    inputfilepath = "./Thesis_Data_Set.csv"
    invcol =[5,4] #First two numbers represent the read ouf col as input value and thrid number is the desired value (relative density)
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
    X = idf.iloc[:,0:len(invcol)-1].values
    y = idf.iloc[:,len(invcol)-1:len(invcol)].values.ravel() #ravel funktion converts the array to a (n,) array instead of an (n,1) vector
 

    P1=(X[0][0],y[0])
    P2=(X[1][0],y[1])
    P3=(X[2][0],y[2])
    print("Points",P1,P2,P3)



    #Divide data in TRAINING DATA and TEST DATA
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=trainingdatasize, random_state=randomstate, shuffle=True)
    
    #Train a given model
    model = linear_model.LinearRegression()
    model_sel = model.fit(X,y)
    print("Coefficients",model_sel.coef_[0])
    print("Bias",model_sel.intercept_)

    m = model_sel.coef_[0]
    d = model_sel.intercept_
    print("D ist",d)
    P1s,P1s2 =getIntersecctionPoint(m,d,P1)

    print("Schnittpunkt1",P1s)
    print("Schnittpunkt2",P1s2)
    print("Px",P1[0])


    #predict the points
    y_pred = model.predict(f1v)
    X_pre = np.asarray(X_pre_lin)

    #plot straight 
    x_s = np.linspace(-25000,800,100)
    y_s = m*x_s+d

    X_s1, y_s1= solveLotsystem(m,d,P1)
    X_s2, y_s2= solveLotsystem(m,d,P2)
    X_s3, y_s3= solveLotsystem(m,d,P3)

    
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
    fig1 = plt.figure(figsize=(cm2inch(8.25,8.25))) 

    # set up axes

    ax1 = fig1.add_subplot(111)
    #plt.grid(linestyle='--', linewidth=1)

    # font = {'family' : 'normal',
    #         'weight' : 'bold',
    #         'size'   : 12}
    # plt.rc('font', **font)
    #plt.rcParams.update({'font.size': 12})

    print("Input X ",X_s1)
    print("Input Y ",y_s1)
    print("Input X  pred",X_pre)
    print("Input Y pred ",y_pred)

    sc =ax1.scatter(X.T[0], y,  linewidth=0.5, edgecolors='black',label='Showcase Data Set')
    # plt.plot(X_pre, y_pred, color='r', label='Regression Model')
    plt.plot(x_s, y_s, 'g', label='y=mx+d')
    plt.plot(X_s1, y_s1, 'b', label='z')
    plt.plot(X_s2, y_s2, 'b', label='z')
    plt.plot(X_s3, y_s3, 'b', label='z')
    plt.plot(0,d,'ro', label="0 d") 
    plt.plot(-d/m,0,'ro', label = "-d/m, 0") 
    k1 = -(d/m)
    print("-d/m",-d/m)
    print("-d",-d)
    betrag = math.sqrt(k1**2+(-d)**2)
    k1_n = k1/betrag
    k2_n = (-d)/betrag
    test = k1_n+k2_n
    print("hallo test",test)
    V = np.array([[(k1/betrag),(-d)/betrag]])
    origin = np.array([[0],[d]]) # origin point
    # plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=1)
    ax1.arrow(0,d, k1_n*200,k2_n*200 , head_width=0.5, head_length=0.5)
    ax1.arrow(0,d, m,-1 , head_width=0.5, head_length=0.5)
    ax1.arrow(0,d, k2_n,-k1_n , head_width=0.5, head_length=0.5)
    print("-d/m",-d/m)
    
    ax1.legend()
    #ax1.legend([line1, line2, line3], ['label1', 'label2', 'label3'])
    plt.xlabel("Laser Power [W]")
    plt.ylabel("Relative Density [%]")
    # plt.title("Linear Regression")
    #plt.savefig("linechart_Duration_env1.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
    plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/02_TheoreticalBackground/fig/MachineLearning/LinearRegressionChart.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
    plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/02_TheoreticalBackground/fig/MachineLearning/LinearRegressionChart.svg", format='svg',bbox_inches='tight') # saved as eps for high quality pictures
    plt.show()