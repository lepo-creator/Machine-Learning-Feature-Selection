#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd

#import matplotlib.font_manager
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def Average(lst):
    return sum(lst) / len(lst)
  
def Median(lst):
    lst.sort()
    mid = len(lst) // 2
    return (lst[mid] + lst[~mid]) / 2


#Read out the Data
def plotoverfitting(Inputpath,Outputpath):
    wcTbm=pd.read_csv("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/Python/Maschine Learning/Results/ModelTraining/"+Inputpath, encoding='latin1')


    y_train=wcTbm['y_train'].tolist()
    y_train_predict=wcTbm['y_train_pre'].tolist()
    y_test=wcTbm['y_test'].tolist()
    y_test_predict=wcTbm['y_test_pre'].tolist()
    




    # lis = []
    # i=1
    # for ele in RSList:
    #     lis.append(i)
    #     i+=1


    # csfont = {'fontname':'LinuxBiolinum'}
    # hfont = {'fontname':'Helvetica'}
    #CHANGE FONT SIZES
    plt.rc('font',family='Linux Biolinum')
    # plt.rc('font', size=12, weight='bold') #controls default text size
    #plt.rc('font', size=12) #controls default text size
    #plt.rc('axes', titlesize=15) #fontsize of the title
    plt.rc('axes', labelsize=12) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
    # plt.rc('legend', fontsize=15) #fontsize of the legend


    #fig1 = plt.figure()
    fig1 = plt.figure(figsize=(cm2inch(7,5)))

    # set up axes

    ax1 = fig1.add_subplot(111)
    plt.grid(True, which='both')
    #plt.grid(linestyle='--', linewidth=1)
    for axis in [ax1.xaxis, ax1.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))


    plt.grid(True)
    ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, color ='royalblue')
    plt.scatter(y_train,y_train_predict,linestyle='-', color='limegreen',label="Training Set", s=30)

    plt.scatter(y_test,y_test_predict,linestyle='-', color='salmon',label="Test Set", s= 30)



    ax1.legend()


    plt.xlabel("Predicted Relative Density [%]")
    plt.ylabel("Relative Density [%]")


    plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/08_validation/fig/"+Outputpath, format='pdf',bbox_inches='tight') # saved as eps for high quality pictures


    # plt.show()

if __name__ == '__main__':

    ### BASIC MODEL ##############################################################################################################
    # Test Set
    plotoverfitting('BasicModel/PredictionResults/RS_32.csv','BasicModel/Overfittingtest/Test/Worestcase.pdf') 
    plotoverfitting('BasicModel/PredictionResults/RS_86.csv','BasicModel/Overfittingtest/Test/Bestcase.pdf')
    # Training Set
    plotoverfitting('BasicModel/PredictionResults/RS_76.csv','BasicModel/Overfittingtest/Training/Worestcase.pdf')
    plotoverfitting('BasicModel/PredictionResults/RS_35.csv','BasicModel/Overfittingtest/Training/Bestcase.pdf')

    ### SIMULATION DATA MODEL ##############################################################################################################
    # Test Set
    plotoverfitting('SimulationDataModel/PredictionResults/RS_69.csv','SimulationDataModel/Overfittingtest/Test/Worestcase.pdf') 
    plotoverfitting('SimulationDataModel/PredictionResults/RS_86.csv','SimulationDataModel/Overfittingtest/Test/Bestcase.pdf')
    # Training Set
    plotoverfitting('SimulationDataModel/PredictionResults/RS_48.csv','SimulationDataModel/Overfittingtest/Training/Worestcase.pdf')
    plotoverfitting('SimulationDataModel/PredictionResults/RS_74.csv','SimulationDataModel/Overfittingtest/Training/Bestcase.pdf')

    ### SIMULATION DATA MODEL ##############################################################################################################
    # Test Set
    plotoverfitting('FeatureSelectionAlgorithm/PredictionResults/RS_32.csv','FeatureSelectionAlgorithm/Overfittingtest/Test/Worestcase.pdf') 
    plotoverfitting('FeatureSelectionAlgorithm/PredictionResults/RS_9.csv','FeatureSelectionAlgorithm/Overfittingtest/Test/Bestcase.pdf')
    # Training Set
    plotoverfitting('FeatureSelectionAlgorithm/PredictionResults/RS_31.csv','FeatureSelectionAlgorithm/Overfittingtest/Training/Worestcase.pdf')
    plotoverfitting('FeatureSelectionAlgorithm/PredictionResults/RS_81.csv','FeatureSelectionAlgorithm/Overfittingtest/Training/Bestcase.pdf')
