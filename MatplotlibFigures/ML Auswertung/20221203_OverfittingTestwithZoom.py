#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

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
    
    maxValReaLis = []
    maxValReaLis.append(max(y_train))
    maxValReaLis.append(max(y_test))
    maxValReaLis.append(min(y_train))
    maxValReaLis.append(min(y_test))
    MaxValRea = max(maxValReaLis)
    MaxValRea +=1
    MinValRea = min(maxValReaLis)
    MinValRea -=1

    maxValPreLis = []
    maxValPreLis.append(max(y_train_predict))
    maxValPreLis.append(max(y_test_predict))
    maxValPreLis.append(min(y_train_predict))
    maxValPreLis.append(min(y_test_predict))
    MaxValPre = max(maxValPreLis)
    MaxValPre +=1
    MinValPre = min(maxValPreLis)
    MinValPre -=1





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
    plt.rc('axes', labelsize=14) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
    # plt.rc('legend', fontsize=15) #fontsize of the legend


    #fig1 = plt.figure()
    fig1 = plt.figure(figsize=(cm2inch(15,7)))

    # set up axes
    plt.xlabel("Predicted Relative Density [%]")
    plt.ylabel("Relative Density [%]")

    ax1 = fig1.add_subplot(111)
    plt.grid(True, which='both')
    #plt.grid(linestyle='--', linewidth=1)
    for axis in [ax1.xaxis, ax1.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))


    plt.grid(True)
    ax1.plot([0, 150], [0, 150], color ='royalblue')

    plt.scatter(y_train_predict,y_train,linestyle='-', color='limegreen',label="Training Set", s=30)

    plt.scatter(y_test_predict,y_test,linestyle='-', color='salmon',label="Test Set", s= 30)



    axins = inset_axes(ax1, 1, 1, loc = 1, bbox_to_anchor=(0.47, 0.85),#0.41, 0.74 #0.85, 0.52 #0.40, 0.85 #0.47, 0.85
                   bbox_transform = ax1.figure.transFigure)


    axins.scatter(y_train_predict,y_train,linestyle='-', color='limegreen',label="Training Set")
    axins.scatter(y_test_predict,y_test,linestyle='-', color='salmon',label="Test Set")
    # axins.plot([0, 1], [0, 1], transform=axins.transAxes, color ='royalblue')
    axins.plot([0, 150], [0, 150], color ='royalblue')
    axins.set_xlim(95, MaxValRea)
    axins.set_ylim(95, MaxValPre)
    
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec = "0.4")
    axins.set_xlabel("")
    axins.set_ylabel("")
    ax1.legend(loc='lower center')
    axins.grid(True)
    # ax1.set_xlim(MinValRea-3, MaxValRea+2.7)
    # ax1.set_ylim(MinValPre-3, MaxValPre+2.7)
    ax1.set_xlim(40, 103)
    ax1.set_ylim(40, 103)
    

    # ax1.set_xlabel("Predicted Relative Density [%]")
    # ax1.set_ylabel("Relative Density [%]")


    plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/08_validation/fig/"+Outputpath, format='pdf',bbox_inches='tight') # saved as eps for high quality pictures


    # plt.show()

if __name__ == '__main__':

    ### BASIC MODEL ##############################################################################################################
    # Test Set
    plotoverfitting('BasicModel/PredictionResults/RS_32.csv','BasicModel/Overfittingtest/Test/Zoom/WorestcaseZoom.pdf') 
    plotoverfitting('BasicModel/PredictionResults/RS_86.csv','BasicModel/Overfittingtest/Test/Zoom/BestcaseZoom.pdf')
    # Training Set
    plotoverfitting('BasicModel/PredictionResults/RS_76.csv','BasicModel/Overfittingtest/Training/Zoom/WorestcaseZoom.pdf')
    plotoverfitting('BasicModel/PredictionResults/RS_35.csv','BasicModel/Overfittingtest/Training/Zoom/BestcaseZoom.pdf')

    ### SIMULATION DATA MODEL ##############################################################################################################
    # Test Set
    plotoverfitting('SimulationDataModel/PredictionResults/RS_69.csv','SimulationDataModel/Overfittingtest/Test/Zoom/WorestcaseZoom.pdf') 
    plotoverfitting('SimulationDataModel/PredictionResults/RS_86.csv','SimulationDataModel/Overfittingtest/Test/Zoom/BestcaseZoom.pdf')
    # Training Set
    plotoverfitting('SimulationDataModel/PredictionResults/RS_48.csv','SimulationDataModel/Overfittingtest/Training/Zoom/WorestcaseZoom.pdf')
    plotoverfitting('SimulationDataModel/PredictionResults/RS_74.csv','SimulationDataModel/Overfittingtest/Training/Zoom/BestcaseZoom.pdf')

    ### SIMULATION DATA MODEL ##############################################################################################################
    # Test Set
    plotoverfitting('FeatureSelectionAlgorithm/PredictionResults/RS_32.csv','FeatureSelectionAlgorithm/Overfittingtest/Test/Zoom/WorestcaseZoom.pdf') 
    plotoverfitting('FeatureSelectionAlgorithm/PredictionResults/RS_9.csv','FeatureSelectionAlgorithm/Overfittingtest/Test/Zoom/BestcaseZoom.pdf')
    # Training Set
    plotoverfitting('FeatureSelectionAlgorithm/PredictionResults/RS_31.csv','FeatureSelectionAlgorithm/Overfittingtest/Training/Zoom/WorestcaseZoom.pdf')
    plotoverfitting('FeatureSelectionAlgorithm/PredictionResults/RS_81.csv','FeatureSelectionAlgorithm/Overfittingtest/Training/Zoom/BestcaseZoom.pdf')
