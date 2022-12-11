#----MachineLearning_ProcessWindow_Prediction.py-------------------------------------------------------------------------------
#----Machine Learning Algorithm to predict a process window of Laser Powder Bed Fusion processes with physics simulation-------
#----input data from Autodesk Netfabb Simulations------------------------------------------------------------------------------ 
#----Author: Leon Pohl, Master's Student TUHH Mechatronics --------------------------------------------------------------------
#----Date: From mid of September 2022 to mid of Dezember 2022------------------------------------------------------------------
#----Research Project Mechatronics---------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from itertools import zip_longest
import pandas as pd
import numpy as np

#own imports
from plotpy import *
from csvpy import *
from featureselection import *



if __name__ == "__main__":

    #--------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------INPUT----------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------

    ###### Not really used, because algorithm selects the borders by itself according to the input data set. 
    maxf1val =100     #maximal possible value of feature 1
    minf1val =800   #minimal possible value of feature 1
    f1p=100       #Number of points, which are plotted in the borders of min and max value of f1
    maxf2val =1000     #maximal possible value of feature 2
    minf2val =2000  #minimal possible value of feature 2
    f2p=100        #Number of points, which are plotted in the borders of min and max value of f2
    ########
    invcol =[0,1,2,3,5,6,7,9,4] #Selection of specific colum of of csv file last entry needs to be desired quantity (e.g. relative Density)
    # Important that constant features like a constant layer thickness need to be ignored. Otherwise this leads to an error.

    #The search_space defines the possible hyperparameters of the multilayer perceptron (automatic search and the best performing parameters get elected)
    search_spache = {
        "hidden_layer_sizes":[(10,),(30,),(100,),(10,10,),(30,30,),(100,100,)],#1,10,100,1000
        "activation":['relu','identity','logistic','tanh'],#'identity','logistic','tanh','relu'
        "solver":['lbfgs'],#'sgd','adam'
        "alpha":[0.00001,0.0001,0.001]
    }
     # inputfilepath = "./InputData/prepared/AlSi10Mg_collected_data.csv"
    inputfilepath = "./InputData/prepared/AlSi10Mg_collected_data.csv"
    outputfilepath = "./results/ModelTraining/Final/"
    DesDen = 95 # minimal possible density, which appear in the process window
    NT3d = 9 #number of ticks on the colorbar on the 3D plot
    NT2d = 9 #number of ticks on the colorbar on the 2D plot
    randomstate = 44 # sets a random state for the random split funktion to reproduce results #40 30 29 | 42 34 | 28 | 27 2 states
    testdatasize = 0.2 # defines the trainingdatasize of the Trainingsdatasplit in Test and Trainingsdata
    T = 0.2 #Threshold for hierarchical clustering visable in Dendrogram
    scoring = 'neg_root_mean_squared_error' #'neg_mean_absolute_percentage_error'#,'neg_root_mean_squared_error'] # defines the used scoring method
    automaticfeatsel =  1 # Defines, wether the feature selection appears automatic or manual
    incolfea = [5,6,4] # [0,1,2,3,5,6,7,9,4],[5,6,4] defines the selected features, if automaticfeatsel is 0
    numberintervals = 5 # defines the number intervalls with are used for the stratified shuffle split
    pathInputData = './Visual/3DVisualisationInputData_temp'
    
    
    #--------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------CODE-----------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------
    randomstatelist= list(range(4,5,1))
    print("The random state list:",randomstatelist)
    print("Number of random states:",len(randomstatelist))
    errList = []
    errList2 = []
    errList3 = []
    errList4 = []
    FeatCorrNamesNum= []
    FeatPermuNamesNum = []
    FeatCorrNames= []
    FeatPermuNames = []
    
    for ele in randomstatelist:
        randomstate = ele


        if automaticfeatsel == 1:
            idf=readcsvcol(inputfilepath,invcol) # gets the pandas dataframe
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
        # Normalizses the data for spearman correlation feature selection
        scaler = MinMaxScaler()
        X_pc = scaler.fit_transform(X)

        #Remove features that have a close monotonic correlations
        if automaticfeatsel == 1:
            X_sel_o,colheadersidf_sel = getFeatureCorrelation(X_pc,T,colheadersidf,scaler)
            FeatCorrNamesNum.append(len(colheadersidf_sel)-1)
            FeatCorrNames.append(colheadersidf_sel[:-1])
        else:
            X_sel_o = X
            colheadersidf_sel = colheadersidf

        #Scales the selected features for training ML Model
        scaler_c = MinMaxScaler()
        X_sel = scaler_c.fit_transform(X_sel_o)


        if automaticfeatsel == 1:
            #StratifiedShuffleSplit
            group_label, intervalpoints = creategroups(idf,numberintervals) # places in the regression data points relative density to speific groups to apply stratification
            sss = StratifiedShuffleSplit(n_splits=1 , test_size=testdatasize, random_state=randomstate) #StratifiedShufflesplit applied 

            for train_index, test_index in sss.split(X_sel, group_label): # splits the normalised data into groups using the fakt that the test and trainingsdata need in members out of every group
                # print("n_split","TRAIN:", train_index, "TEST:", test_index)
                # splits the normalised data into groups using the fakt that the test and trainingsdata need in members out of every group
                X_train, X_test = X_sel[train_index], X_sel[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # splits the original data the same way for visualization purporses 
                X_train_o, X_test_o = X_sel_o[train_index], X_sel_o[test_index]
                y_train_o, y_test_o = y[train_index], y[test_index]
        elif automaticfeatsel == 0:
            #Divide data in TRAINING DATA and TEST DATA
            X_train, X_test, y_train, y_test = train_test_split(X_sel,y, test_size=testdatasize, random_state=randomstate, shuffle=True)
            X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_sel_o,y, test_size=testdatasize, random_state=randomstate, shuffle=True)


        

        #Plot Input Data Points only 3D
        plotinputdata(X_sel_o,y,colheadersidf_sel,NT3d,pathInputData)
        plotinputdataML(X_train_o,y_train_o,X_test_o, y_test_o,colheadersidf_sel,testdatasize)


        #Train MUTLILAYER PERCEPTRON MODEL
        net = MLPRegressor(max_iter= 10000000,random_state=randomstate)


        # GRID SEARCH tune hyperparameters
        GS = GridSearchCV(estimator=net,param_grid=search_spache,scoring=scoring, refit=scoring,cv= 5, verbose=4, n_jobs=-1) #sklearn.metrics.SCORERS.keys()

        GS.fit(X_train,y_train)


        if automaticfeatsel == 1:
            model4 = GS.best_estimator_

            # gets the most important features for the trained grid search model
            X_sel2,colheadersidf_sel, scaler_p, X_sel2_o = getPermutationImportance(model4,X_test,y_test,randomstate, scoring,colheadersidf_sel,idf)
            
            if X_sel2 == '':
                errList.append('')
                errList2.append('')
                errList3.append('')
                errList4.append('')
                FeatPermuNamesNum.append(0)
                FeatPermuNames.append('')
            else:
                FeatPermuNamesNum.append(len(colheadersidf_sel)-1)
                FeatPermuNames.append(colheadersidf_sel[:-1])
                for train_index, test_index in sss.split(X_sel2, group_label):
                    X_train, X_test = X_sel2[train_index], X_sel2[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                #Train same model again
                GS.fit(X_train,y_train)
            print("Number of final features",FeatPermuNamesNum)
        elif automaticfeatsel == 0:
            X_sel2 = X_sel
            scaler_p = scaler_c
        

        if X_sel2 != '':

           
            model = GS.best_estimator_
            

            # Predict the Outputs of BOTH models
            y_test_pre = model.predict(X_test)
            y_train_pre = model.predict(X_train)

            

            #Saves the GridSearch results in a csv file
            ydf=pd.DataFrame(list(zip_longest(y_train,y_train_pre,y_test,y_test_pre)))
            ydf.columns=["y_train","y_train_pre","y_test","y_test_pre"]
            ydf.to_csv(str(outputfilepath)+'PredictionResults/RS_{}.csv'.format(randomstate))


            # #calculate error of BOTH MODELS
            rmse=(mean_squared_error(y_test,y_test_pre)**0.5)
            print("The RMSE is", rmse)
            mape = mean_absolute_percentage_error(y_test,y_test_pre)
            print("The MAPE is", mape)
            mse=mean_squared_error(y_test,y_test_pre)
            print("The MSE is", mse)
            print("The training error is",GS.best_score_)
            errList.append(rmse)
            errList2.append(mape)
            errList3.append(mse)
            errList4.append(GS.best_score_*-1)


            #Print OUTPUT DATA
            print("Grid Search results BEST MODEL\n",GS.best_estimator_)
            print("Grid Search results BEST PARAMETERS\n",GS.best_params_)


            #Saves the GridSearch results in a csv file
            df = pd.DataFrame(GS.cv_results_)
            df = df.sort_values("rank_test_score")
            df.to_csv(str(outputfilepath)+'GSResults/RS_{}.csv'.format(randomstate))

            params = model.get_params()
            print("GS Model \n", params)

            # Plot Modelstructure

            listhid = []
            
            for ele in params['hidden_layer_sizes']:
                listhid.append(ele)

            hiddarr = np.asarray(listhid)
            #Plots the neuronal Network
            a= np.array(len(colheadersidf_sel)-1)
            c = np.array(1)
            d = np.append(a,hiddarr)
            layers= np.append(d,c)
        
            drawNN('./Results/NNModel.pdf',layers) # function which draws the neuronal network

            D3= preVal(model,minf1val,maxf1val,f1p,minf2val,maxf2val,f2p,automaticfeatsel, X_sel2, scaler_p) # gerates matrix of predited values and input features
            
            priProWin(D3,colheadersidf_sel,DesDen,NT3d,NT2d) # plots the process window
        
    
    if FeatCorrNames != []:
        dft=pd.DataFrame(list(zip(randomstatelist,errList,errList2,errList3,errList4,FeatCorrNamesNum,FeatPermuNamesNum,FeatCorrNames,FeatPermuNames)))
        dft.columns=["Randomstate","RMSE","MAPE","MSE","Training RMSE","Feat Corr Num","Feat Permu Num","Feat Corr Name","Feat Permu Name"]
        dft.to_csv(str(outputfilepath)+'TrainTestErrorResults.csv')
    else:
        dft=pd.DataFrame(list(zip(randomstatelist,errList,errList2,errList3,errList4)))
        dft.columns=["Randomstate","RMSE","MAPE","MSE","Training RMSE"]
        dft.to_csv(str(outputfilepath)+'TrainTestErrorResults.csv')

    
    



