from asyncio.windows_events import NULL
from fileinput import filename
from traceback import print_tb
from weakref import ref
from sklearn import linear_model
from sklearn.model_selection import train_test_split



from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
from scipy.spatial import ConvexHull

#own imports
from csvpy import *

#Sets Plotdesigndata on global Level
 #CHANGE FONT TYPE
plt.rc('font',family='Linux Biolinum')
#CHANGE FONT SIZES
# plt.rc('font', size=12, weight='bold') #controls default text size
#plt.rc('font', size=12) #controls default text size
plt.rc('figure', titlesize=17.28) #fontsize of the title
plt.rc('axes', titlesize=14.4) #fontsize of the title
plt.rc('axes', labelsize=12) #fontsize of the x and y labels
plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
plt.rc('legend', fontsize=12) #fontsize of the legend
plt.set_loglevel("error") # just shows important error. Ignores warnings.





def preVal(model,f1min,f1max,f1num,f2min,f2max,f2num, automaticfeatsel, X_sel):
    
    if automaticfeatsel == 0:
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
        X_pre = np.zeros((numrows,2))
        l = 0
        for i in range(f1num):
            for k in range(f2num):
                X_pre[l][0]=f1v[i]
                X_pre[l][1]=f2v[k]
                l+=1
        # print("X Predict werte",X_pre)

       

        # #DEBUGG print D
        # print(D)
        # print(D.shape)

    elif automaticfeatsel == 1:
        maximum_element_col = np.max(X_sel, 0)
        minimum_element_col = np.min(X_sel, 0)
        a =[]
        for i in range(len(maximum_element_col)):
            a.append(np.linspace(minimum_element_col[i], maximum_element_col[i], f1num)) # adds a linspace array for each feature
        X_l = list(product(*a)) # creates a list with all combinations of features
        X_pre = np.asarray(X_l) # turns the list in a numpy array

    # Calculate preditions and add them together to one array
    y_pre = model.predict(X_pre)
    y_prevt, u1 = np.meshgrid(y_pre,1)
    num_col = np.atleast_2d(X_pre).shape[1] # gets the number of columns of a numpy array
    D = np.insert(X_pre,num_col, y_prevt, axis=1) # inserts a col vector in a numpy array at position num_col      
   
    return D 

def priProWin(D,colheadersidf,desden,cp1a,cp2a):
    
    
    #Plots 2 Windows next to each other uses window pixel size
    start_x, start_y, dx, dy = (0, 30, 960, 1080)
    for i in range(2):
        if i%3 == 0:
            x = start_x
            y = start_y  + (dy * (i//3) )
        fig=plt.figure()
        if i == 0:
            # fig = plt.figure(num=None, figsize=(11.55, 13), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.axes(projection='3d')
            # print(D)
            # maxVal= D.max(axis=1)
            maxVal =np.max(D[:,2])
            minVal =np.min(D[:,2])
            # print("MaxValue",maxVal)
            # print("MinValue",minVal)
            sc =ax.scatter(D.T[0], D.T[1], D.T[2], c=D.T[2], cmap='RdYlGn', linewidth=0.5, edgecolors='black',label='Predicted Process Points');
            # sc.set_clim(minVal,maxVal) #used to set colorbounds of the colormap
            ax.legend()
            cb=plt.colorbar(sc,ticks=np.linspace(minVal,maxVal,cp1a,dtype='float32'),format='%.2f')
            fig.suptitle('3D Visualisation of Predicted Process Points', fontweight ="bold")
            # print("Test123",np.linspace(minVal,maxVal,cp1a,dtype='str'))
            # cb.ax.set_yticklabels(np.linspace(minVal,maxVal,cp1a,dtype='str'))
            ax.set_xlabel(colheadersidf[0])
            ax.set_ylabel(colheadersidf[1])
            ax.set_zlabel(colheadersidf[2]);
            cb.set_label(colheadersidf[2]);
            plt.savefig("./Visual/3DVisualisation_temp.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
            plt.savefig("./Visual/3DVisualisation_temp.svg", format='svg',bbox_inches='tight')
            plt.savefig("./Visual/3DVisualisation_temp.png", bbox_inches='tight')


            # ax.set_title('Visualisation Dataset')
        
        if i == 1:
            
            #Moifies the Dataset
            D[D[:,2]>desden]
            D_dm = D[D[:,2]>desden] # deletes each row with a density lower then the desired density desden
            D_wd = np.delete(D_dm, 2, 1) # deletes the density column out of the dataset
            maxVal2 =np.max(D_dm[:,2])#Searches for the maximal relative density in the dataset
            minVal2 =np.min(D_dm[:,2])#Searches for the minimal relative density in the dataset

            #Second 2D plot
            ax = plt.axes()
            sc2=ax.scatter(D_dm.T[1], D_dm.T[0], c=D_dm.T[2], cmap='RdYlGn', linewidth=0.5, edgecolors='black',label='Predicted Process Points')

            #Creates a Hull from the points
            hull = ConvexHull(D_wd,incremental=False) # defines a hull from the points
            hullverm= np.append(hull.vertices,hull.vertices[0]) # adds the first point to the array to have a closed convex hull
            plt.plot(D_wd[hullverm,1], D_wd[hullverm,0], 'k', lw=2, label='Predicted Process Window') #plots the hull
            ax.legend()#plots a legend
            cb = plt.colorbar(sc2,ticks=np.linspace(minVal2,maxVal2,cp2a,dtype='float32'),format='%.2f')#plots a colorbar

            #Defines Titels of the plot
            fig.suptitle('Predicted Process Window with a {} higher than {}. '.format(colheadersidf[2],desden), fontweight ="bold")
            ax.set_xlabel(colheadersidf[1])
            ax.set_ylabel(colheadersidf[0])
            cb.set_label(colheadersidf[2]);

            #saves the plot
            plt.savefig("./Visual/2DVisualisation_temp.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
            plt.savefig("./Visual/2DVisualisation_temp.svg", format='svg',bbox_inches='tight')
            plt.savefig("./Visual/2DVisualisation_temp.png", bbox_inches='tight')



        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(x, y, dx, dy)
        x += dx
 
    plt.show()



def plotinputdata(Xm,ym,colheadersidf,cp1a):
    #CHANGE FONT TYPE
    plt.rc('font',family='Linux Biolinum')
    #CHANGE FONT SIZES
    # plt.rc('font', size=12, weight='bold') #controls default text size
    #plt.rc('font', size=12) #controls default text size
    plt.rc('figure', titlesize=17.28) #fontsize of the title
    plt.rc('axes', titlesize=14.4) #fontsize of the title
    plt.rc('axes', labelsize=12) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
    plt.rc('legend', fontsize=12) #fontsize of the legend
    plt.set_loglevel("error") # just shows important error. Ignores warnings.

    #Plots 2 Windows next to each other uses window pixel size
    start_x, start_y, dx, dy = (0, 30, 1920, 1080)
    for i in range(1): # determins the number of plotted windows
        if i%3 == 0:
            x = start_x
            y = start_y  + (dy * (i//3) )

        fig=plt.figure()
        if i == 0:
            ax = plt.axes(projection='3d')
            maxVal =np.max(ym)
            minVal =np.min(ym)
            sc =ax.scatter(Xm.T[0], Xm.T[1], ym, c=ym, cmap='RdYlGn',  linewidth=0.5, edgecolors='black',label='Input Data Points');
            # sc.set_clim(minVal,maxVal) #used to set colorbounds of the colormap
            ax.legend()
            cb=plt.colorbar(sc,ticks=np.linspace(minVal,maxVal,cp1a),format='%.2f')
            fig.suptitle('3D Visualisation of Input Data Points', fontweight ="bold")
            ax.set_xlabel(colheadersidf[0])
            ax.set_ylabel(colheadersidf[1])
            ax.set_zlabel(colheadersidf[2]);
            cb.set_label(colheadersidf[2]);
            plt.savefig("./Visual/3DVisualisationInputData_temp.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
            plt.savefig("./Visual/3DVisualisationInputData_temp.svg", format='svg',bbox_inches='tight')
            plt.savefig("./Visual/3DVisualisationInputData_temp.png", bbox_inches='tight')
            

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(x, y, dx, dy)
        x += dx
    plt.show()


def plotinputdataML(X_train,y_train,X_test, y_test,colheadersidf,testdatasize):

    #Plots 2 Windows next to each other uses window pixel size
    start_x, start_y, dx, dy = (0, 30, 1920, 1080)
    for i in range(1): # determins the number of plotted windows
        if i%3 == 0:
            x = start_x
            y = start_y  + (dy * (i//3) )

        fig=plt.figure()
        if i == 0:
            ax = plt.axes(projection='3d')
            # maxVal =np.max(ym)
            # minVal =np.min(ym)
            sc =ax.scatter(X_train.T[0], X_train.T[1], y_train, c='black', linewidth=0.5,label='Trainings Data Points');
            sc2 =ax.scatter(X_test.T[0], X_test.T[1], y_test, c='red', linewidth=0.5, label='Test Data Points');
            # sc.set_clim(minVal,maxVal) #used to set colorbounds of the colormap
            ax.legend()
            # cb=plt.colorbar(sc,ticks=np.linspace(minVal,maxVal,cp1a),format='%.2f')
            trainingdatasize = (1-testdatasize)*100
            fig.suptitle('3D Visualisation of Input Data Divided in {:.2f} % Training and {:.2f} % Test Data Set '.format(trainingdatasize,testdatasize*100), fontweight ="bold")
            ax.set_xlabel(colheadersidf[0])
            ax.set_ylabel(colheadersidf[1])
            ax.set_zlabel(colheadersidf[2]);
            # cb.set_label(colheadersidf[2]);
            plt.savefig("./Visual/3DVisualisationTestandTrainingData_temp.eps",bbox_inches='tight', format='eps') # saved as eps for high quality pictures
            plt.savefig("./Visual/3DVisualisationTestandTrainingData_temp.svg",bbox_inches='tight', format='svg')
            plt.savefig("./Visual/3DVisualisationTestandTrainingData_temp.png",bbox_inches='tight')
            

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(x, y, dx, dy)
        x += dx
    plt.show()



def plotpermutationimportance(X_train,y_train,X_test, y_test,colheadersidf,testdatasize):

    #Plots 2 Windows next to each other uses window pixel size
    start_x, start_y, dx, dy = (0, 30, 1920, 1080)
    for i in range(1): # determins the number of plotted windows
        if i%3 == 0:
            x = start_x
            y = start_y  + (dy * (i//3) )

        fig, ax = plt.subplots(figsize =(16, 9))
        if i == 0:
            ax.barh(colheadersidf,)
            # maxVal =np.max(ym)
            # minVal =np.min(ym)
            sc =ax.scatter(X_train.T[0], X_train.T[1], y_train, c='black', linewidth=0.5,label='Trainings Data Points');
            sc2 =ax.scatter(X_test.T[0], X_test.T[1], y_test, c='red', linewidth=0.5, label='Test Data Points');
            # sc.set_clim(minVal,maxVal) #used to set colorbounds of the colormap
            ax.legend()
            # cb=plt.colorbar(sc,ticks=np.linspace(minVal,maxVal,cp1a),format='%.2f')
            trainingdatasize = (1-testdatasize)*100
            fig.suptitle('3D Visualisation of Input Data Divided in {:.2f} % Training and {:.2f} % Test Data Set '.format(trainingdatasize,testdatasize*100), fontweight ="bold")
            ax.set_xlabel(colheadersidf[0])
            ax.set_ylabel(colheadersidf[1])
            ax.set_zlabel(colheadersidf[2])
            # cb.set_label(colheadersidf[2]);
            plt.savefig("./Visual/3DVisualisationTestandTrainingData_temp.eps",bbox_inches='tight', format='eps') # saved as eps for high quality pictures
            plt.savefig("./Visual/3DVisualisationTestandTrainingData_temp.svg",bbox_inches='tight', format='svg')
            plt.savefig("./Visual/3DVisualisationTestandTrainingData_temp.png",bbox_inches='tight')
            

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(x, y, dx, dy)
        x += dx
    plt.show()


if __name__ == "__main__":
    idf=readcsvcol("./InputData/collected_data.csv",[3,4,5])
    colheadersidf=list(idf.columns.values) # gets a list of df header strings

    #Spilts the Pandas dataframe in to numpy array Inputs X and result y 
    X = idf.iloc[:,0:2].values
    y = idf.iloc[:,2:3].values.ravel() #ravel funktion converts the array to a (n,) array instead of an (n,1) vector
  

    # plotinputdata(X,y,colheadersidf,8)
    # Divide data in TRAINING DATA and TEST DATA
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)

    # Train LINEAR REGRESSON MODEL
    l_reg = linear_model.LinearRegression()
    model = l_reg.fit(X_train, y_train)

    


    D = preVal(model,0,100,4,0,100,4,1,X)
    # priProWin(D,colheadersidf,95,8,8)