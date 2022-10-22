#----MachineLearning_ProcessWindow_Prediction.py-------------------------------------------------------------------------------
#----Machine Learning Algorithm to predict a process window of Laser Powder Bed Fusion processes with physics simulation-------
#----input data from Autodesk Netfabb Simulations------------------------------------------------------------------------------ 
#----It is built in simple 2-dimensional space using 2D raycasting tool--------------------------------------------------------
#----Author: Leon Pohl, Bachelor's Student TUHH Mechatronics ------------------------------------------------------------------
#----Date: From mid of September 2022 to mid of Dezember 2022------------------------------------------------------------------
#----Research Project Mechatronics---------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os


#own imports
from plotpy import *
from csvpy import *
from featureselection import *



if __name__ == "__main__":

    #--------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------INPUT----------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------


    maxf1val =100     #maximal possible value of feature 1
    minf1val =800   #minimal possible value of feature 1
    f1p=100          #Number of points, which are plotted in the borders of min and max value of f1
    maxf2val =1000     #maximal possible value of feature 2
    minf2val =2000  #minimal possible value of feature 2
    f2p=100          #Number of points, which are plotted in the borders of min and max value of f2
    invcol =[1,2,3,4,6,7,8,5] #Selection of specific colum of of csv file last entry needs to be desired quantity (e.g. relative Density)
    #The search_space defines the possible hyperparameters of the multilayer perceptron (automatic search and the best performing parameters get elected)
    search_spache = {
        "hidden_layer_sizes":[(1,),(10,),(30,),(100,)],#1,10,100,1000
        "activation":['relu','identity','logistic','tanh'],#'identity','logistic','tanh','relu'
        "solver":['lbfgs']#'sgd','adam'
    }
    inputfilepath = "./InputData/collected_data.csv"
    outputfilepath = "./results/cv_results_ModelTraining.csv"
    DesDen = 95 # minimal possible density, which appear in the process window
    NT3d = 9 #number of ticks on the colorbar on the 3D plot
    NT2d = 9 #number of ticks on the colorbar on the 2D plot
    randomstate = 42 # sets a random state for the random split funktion to reproduce results #40 30 29 | 42 34 | 28 | 27 2 states
    testdatasize = 0.2 # defines the trainingdatasize of the Trainingsdatasplit in Test and Trainingsdata
    T = 0.2 #Threshold for hierarchical clustering visable in Dendrogram
    scoring = 'neg_mean_absolute_percentage_error'#,'neg_root_mean_squared_error'] # defines the used scoring method
    automaticfeatsel = 0 # Defines, wether the feature selection appears automatic or manual
    incolfea = [6,7,5] # defines the selected features, if automaticfeatsel is 0

    
    
    #--------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------CODE-----------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------
      
    
    
    if automaticfeatsel == 1:
        idf=readcsvcol(inputfilepath,invcol) # gets the pandas dataframe#
        le = len(invcol)
    elif automaticfeatsel == 0:
        idf=readcsvcol(inputfilepath,incolfea) # gets the pandas dataframe
        le = len(incolfea)
    else:
        raise ValueError("Variable automaticfeatsel needs to be 0 or 1.")


    colheadersidf=list(idf.columns.values) # gets a list of df header strings

    #Spilts the Pandas dataframe in to numpy array Inputs X and result y 
    X = idf.iloc[:,0:le-1].values
    y = idf.iloc[:,le-1:le].values.ravel() #ravel funktion converts the array to a (n,) array instead of an (n,1) vector

    # Normalizses the data for sperman correlation feature selection
    scaler = MinMaxScaler()
    X_minmax = scaler.fit_transform(X)

    #Remove features that have a close monotonic correlations
    if automaticfeatsel == 1:
        X_sel_o,colheadersidf_sel = getfeaturecorrelation(X_minmax,T,colheadersidf,scaler)
    else:
        X_sel_o = X
        colheadersidf_sel = colheadersidf

    #Normalizses selected features for training ML Model
    scaler_c = MinMaxScaler()
    X_sel = scaler_c.fit_transform(X_sel_o)

    #Divide data in TRAINING DATA and TEST DATA
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_sel_o,y, test_size=testdatasize, random_state=randomstate, shuffle=True) # orgiginal Dataset for Visualisation purposes
    X_train, X_test, y_train, y_test = train_test_split(X_sel,y, test_size=testdatasize, random_state=randomstate, shuffle=True) #normalized dataset for ML model training purpose


    # #StratifiedShuffleSplit
    # sss = StratifiedShuffleSplit(n_splits=1 , test_size=0.9, random_state=randomstate)
    # for train_index, test_index in sss.split(X, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    # # #Plot Input Data Points
    plotinputdata(X_sel_o,y,colheadersidf_sel,NT3d)
    plotinputdataML(X_train_o,y_train_o,X_test_o, y_test_o,colheadersidf_sel,testdatasize)
    

    # Train LINEAR REGRESSON MODEL
    l_reg = linear_model.LinearRegression()
    model = l_reg.fit(X_train, y_train)


    #Train MUTLILAYER PERCEPTRON MODEL
    # net = MLPRegressor(hidden_layer_sizes=(30,),activation='relu', solver = 'lbfgs', max_iter= 10000000)
    net = MLPRegressor(hidden_layer_sizes=(30,),activation='relu', solver = 'lbfgs', max_iter= 10000000, random_state=randomstate)
    net2 = MLPRegressor(max_iter= 10000000,random_state=randomstate)
    model2 = net.fit(X_train, y_train)

    # GRID SEARCH tune hyperparameters
    GS = GridSearchCV(estimator=net2,param_grid=search_spache,scoring=scoring, refit=scoring,cv = 5, verbose=4, n_jobs=-1) #sklearn.metrics.SCORERS.keys()
    GS.fit(X_train,y_train)
    if automaticfeatsel == 1:
        model4 = GS.best_estimator_

        # gets the most important features for the trained grid search model
        X_sel,colheadersidf_sel, scaler_p = getpermutationimportance(model4,X_test,y_test,randomstate, scoring,colheadersidf_sel,idf)
        #Divide data in TRAINING DATA and TEST DATA
        X_train_sel2, X_test_sel2, y_train_sel2, y_test_sel2 = train_test_split(X_sel,y, test_size=testdatasize, random_state=randomstate, shuffle=True)

        #Train same model again
        GS.fit(X_train_sel2,y_train_sel2)
    elif automaticfeatsel == 0:
        scaler_p = scaler_c
    
    model3 = GS.best_estimator_

    # # GRID SEARCH 2 tune hyperparameters
    # GS2 = GridSearchCV(estimator=net2,param_grid=search_spache,scoring="neg_mean_absolute_percentage_error", refit="neg_mean_absolute_percentage_error",cv = 5, verbose=4, n_jobs=-1) #sklearn.metrics.SCORERS.keys()
    # GS2.fit(X_train_sel2,y_train)
    # model3 = GS2.best_estimator_

    

    # #Predict the Outputs of BOTH models
    # y_pre = model.predict(X_test_sel2)
    # y_pre2 = net.predict(X_test_sel2)
    # y_pre3 = model3.predict(X_test_sel2)


    # #calculate error of BOTH MODELS
    # mse1=mean_squared_error(y_test,y_pre)
    # mape1=mean_absolute_percentage_error(y_test,y_pre)

    # mse2=mean_squared_error(y_test,y_pre2)
    # mape2=mean_absolute_percentage_error(y_test,y_pre2)

    # mse3=mean_squared_error(y_test,y_pre3)
    # mape3=mean_absolute_percentage_error(y_test,y_pre3)

    # #Crossvalidation of BOTH MODELS
    # scores = cross_val_score(model,X,y, cv= 5, scoring='neg_mean_absolute_percentage_error')
    # scores2 = cross_val_score(model2,X,y, cv= 5, scoring='neg_mean_absolute_percentage_error')
    # scores3 = cross_val_score(model3,X,y, cv= 5, scoring='neg_mean_absolute_percentage_error')


    #Print OUTPUT DATA
    print("Grid Search results BEST MODEL\n",GS.best_estimator_)
    print("Grid Search results BEST PARAMETERS\n",GS.best_params_)
    print("Grid Search results BEST SCORE\n",GS.best_score_)


    #Saves the GridSearch results in a csv file
    df = pd.DataFrame(GS.cv_results_)
    df = df.sort_values("rank_test_score")
    df.to_csv(outputfilepath)

    # # Print OUTPUT of 2 Model simple execution
    # print("Test Train Values\n",X_test)
    # print("\nModel 1 \n")
    # print("Predicitions 1\n",y_pre)
    # print("Mean squared error 1\n", mse1)
    # print("Mean absolute percentage error 1\n", mape1)
    params1 = model.get_params()
    print("Model 1 Parameter", params1)

    # print("\nModel 2 \n")
    # print("Predicitions 2\n",y_pre2)
    # print("Mean squared error 2\n", mse2)
    # print("Mean absolute percentage error 2\n", mape2)
    params2 = model2.get_params()
    print("Model 2 Parameter", params2)

    # print("\nModel 3 \n")
    # print("Predicitions 3\n",y_pre3)
    # print("Mean squared error 3\n", mse3)
    # print("Mean absolute percentage error 3\n", mape3)
    params3 = model3.get_params()
    print("Model 3 Parameter", params3)

    #Creates Predicted Data and plots the ProcessWindow
    D3= preVal(model3,minf1val,maxf1val,f1p,minf2val,maxf2val,f2p,automaticfeatsel, X_sel, scaler_p) # gerates matrix of predited values and input features
    priProWin(D3,colheadersidf_sel,DesDen,NT3d,NT2d)



