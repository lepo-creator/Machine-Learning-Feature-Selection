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

idbm=pd.read_csv("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/Python/Maschine Learning/Results/ModelTraining/BasicModel/MatplotlibDaten100.csv", encoding='latin1')
idsm=pd.read_csv("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/Python/Maschine Learning/Results/ModelTraining/SimulationDataModel/MatplotlibDaten100.csv", encoding='latin1')
idfs=pd.read_csv("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/Python/Maschine Learning/Results/ModelTraining/FeatureSelectionAlgorithm/MatplotlibDaten100.csv", encoding='latin1')


RSList=idbm['Randomstate'].tolist()
RMSEListbm=idbm['Training RMSE'].tolist()
RMSEListsm=idsm['Training RMSE'].tolist()
RMSEListfs=idfs['Training RMSE'].tolist()
# MAPEList=idbm['MAPE'].tolist()
# MSEList=idbm['MSE'].tolist()
print("Länge List",len(RMSEListfs))



#calculates the averages
RMSEbmav = Average(RMSEListbm)
RMSEsmav = Average(RMSEListsm)
RMSEfsav = Average(RMSEListfs)
print("\nDEr Basic Model Aver ist",RMSEbmav)
print("Das Simulation data Model Aver ist",RMSEsmav)
print("Das Feature Selection Aver ist",RMSEfsav)

## Median
BMListcpy=RMSEListbm.copy()
SMListcpy=RMSEListsm.copy()
FSListcpy=RMSEListfs.copy()
MedBM=Median(BMListcpy)
MedSM=Median(SMListcpy)
MedFS=Median(FSListcpy)

print("\nDEr Basic Model Median ist",MedBM)
print("Das Simulation data Model Aver ist",MedSM)
print("Das Feature Selection Median ist",MedFS)

#Max Value
print("\nMaximaler Fehler Basic Model",max(RMSEListbm))
print("Maximaler Fehler Simulation Data Model",max(RMSEListsm))
print("Maximaler Fehler Feature Selection Model",max(RMSEListfs))


#Min Value
print("\nMinimaler Fehler Basic Model",min(RMSEListbm))
print("Minimaler Fehler Simulation Data Model",min(RMSEListsm))
print("Minimaler Fehler Feature Selection Model\n",min(RMSEListfs))


lis = []
i=1
for ele in RSList:
    lis.append(i)
    i+=1


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
fig1 = plt.figure(figsize=(cm2inch(17,10)))

# set up axes

ax1 = fig1.add_subplot(111)
plt.grid(True, which='both')
#plt.grid(linestyle='--', linewidth=1)
for axis in [ax1.xaxis, ax1.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 12}
# plt.rc('font', **font)
#plt.rcParams.update({'font.size': 12})
# ax1.set_yscale('log')
plt.grid(True)
# plt.xticks([180000,(2*180000),(3*180000),(4*180000),(5*180000),(6*180000),(7*180000),(8*180000),(9*180000),(10*180000)])
plt.xticks(lis)

#plt.yticks([0,103.58,2000,3009.10,4000,6000,8000,10000,11111.64])

# ax1.set_xticklabels(['360','180','120','90','72','60','51','45','40','36'])
ax1.set_xticklabels("")
ax1.set_xlim(lis[0], lis[-1])
ax1.set_ylim(0,13.5)
#plt.plot(x,y, linestyle='--')
# lis2=[180000,(2*180000),(3*180000),(4*180000),(5*180000),(6*180000),(7*180000),(8*180000),(9*180000),(10*180000)]
# print("LIst 1",lis)
# print("LIst 2",lis2)
# mark_n=[0,1,2,3,7,8,9]
#HORIZONTAL LINE
#plt.hlines(10800, 180000, (10*180000+60000), color='black', linestyle='dashed', label ='Timeout')
#BLACK STYLE
# plt.plot(lis,[94.21,103.58,118.64,136.54,158.63,206.77,3009.10,256.91,2882.27,11111.64] , linestyle='-', marker='o', color='black', label='Env. 1')
# plt.plot(lis,[45.39,82.10,70.20,87.41,0.00,0.00,0.00,609.01,11029.12,11078.97] , linestyle='dashdot', marker='o', color='black', label='Env. 2', markevery=mark_n)
# #plt.plot([(5*180000),(6*180000),(7*180000)],[0.00,0.00,0.00], marker='x', color='red',linestyle='') # red X as Marker
# plt.plot(lis,[126.84,132.65,149.26,167.51,215.06,222.40,11036.71,273.53,11108.83,777.97] , linestyle='dashed', marker='o', color='black', label='Env.3')

#COLOUR STYLE
# plt.plot(lis,RMSEList , linestyle='-', marker='o', color='black', label='RMSE')
plt.plot(lis,RMSEListbm , linestyle='-', color='black', label='Basic Model')


# plt.plot(lis,MAPEList , linestyle='-', marker='v', color='lime', label='MAPE')
plt.plot(lis,RMSEListsm , linestyle='-', color='limegreen', label='Simulation Data Model')

# plt.hlines(RMSEsmav, 1, 101, color='lime', linestyle='--')

# plt.plot(lis,MSEList , linestyle='-', marker='s', color='darkred', label='MSE')
plt.plot(lis,RMSEListfs , linestyle='-', color='salmon', label='Feature Selection Model')

# plt.hlines(RMSEfsav, 1, 101, color='darkred', linestyle='--')


# plt.hlines(RMSEbmav, 1, 101, color='black', linestyle='--',label='Averages')

#Plot horizontal Lines
# plt.yticks([0.001,0.01,0.1,1,10,100])

#plt.plot([180000,(2*180000),(3*180000),(4*180000),(5*180000),(6*180000),(7*180000),(8*180000),(9*180000),(10*180000)],[90.63,96.45,104.27,111.47,120.15,128.42,134.09,142.21,151.09,158.38] , linestyle='-', marker='o', color='green', label='Greedy Search')
ax1.legend(loc='upper left')
#ax1.legend([line1, line2, line3], ['label1', 'label2', 'label3'])

plt.xlabel("Random states")
plt.ylabel("RMSE training set")
#plt.title("With Labels")
plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/08_validation/fig/Errors/TrainingRMSE.pdf", format='pdf',bbox_inches='tight') # saved as eps for high quality pictures


plt.show()