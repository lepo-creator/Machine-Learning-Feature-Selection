#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import re

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

def CountSpecificWord(word,lst):
    k=0
    for ele in lst:
        res=re.findall(word,ele)
        # print(ele)
        # print(res)
        if len(res) == 1:
            k+=1
        elif len(res) != 0:
            print("ERROR")
        
    return k




#Read out the Data


idfs=pd.read_csv("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/Python/Maschine Learning/Results/ModelTraining/FeatureSelectionAlgorithm/MatplotlibDaten100.csv", encoding='latin1')


notcleaned=idfs['Feat Permu Name'].tolist()
FeatPerNam = [x for x in notcleaned if str(x) != 'X']
# MAPEList=idbm['MAPE'].tolist()
# MSEList=idbm['MSE'].tolist()
print("List of ele ments",FeatPerNam)
for ele in FeatPerNam:
    res = re.findall("Laser",ele)
   

Cout = []
CouX = 8 # X were manually counted 
#Cout of features
CouDur=CountSpecificWord('Duration',FeatPerNam)
print("Number of Duration is",CouDur)
CouAvInt=CountSpecificWord('Average',FeatPerNam)
print("Number of Average Interlayertemperature is",CouAvInt)
CouLasPow=CountSpecificWord('Laser',FeatPerNam)
print("Number of Laser Power is",CouLasPow)
CouScaSpe=CountSpecificWord('Scan',FeatPerNam)
print("Number of Scan Speed is",CouScaSpe)
CouHatDis=CountSpecificWord('Hatch',FeatPerNam)
print("Number of Hatch Distance is",CouHatDis)

Cout.append(CouX)
Cout.append(CouDur)
Cout.append(CouAvInt)
Cout.append(CouHatDis)
Cout.append(CouScaSpe)
Cout.append(CouLasPow)





# csfont = {'fontname':'LinuxBiolinum'}
# hfont = {'fontname':'Helvetica'}
#CHANGE FONT SIZES
plt.rc('font',family='Linux Biolinum')
# plt.rc('font', size=12, weight='bold') #controls default text size
plt.rc('font', size=12) #controls default text size
#plt.rc('axes', titlesize=15) #fontsize of the title
plt.rc('axes', labelsize=14) #fontsize of the x and y labels
plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
# plt.rc('legend', fontsize=15) #fontsize of the legend


#fig1 = plt.figure()
fig1 = plt.figure(figsize=(cm2inch(14,10)))

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
plt.grid(axis='y',alpha=0.7)
# plt.xticks([180000,(2*180000),(3*180000),(4*180000),(5*180000),(6*180000),(7*180000),(8*180000),(9*180000),(10*180000)])
# plt.xticks(lis)

#plt.yticks([0,103.58,2000,3009.10,4000,6000,8000,10000,11111.64])

# ax1.set_xticklabels(['360','180','120','90','72','60','51','45','40','36'])

# ax1.set_xlim(lis[0], lis[-1])

# ax1.set_ylim(0,13.5)
plt.barh(range(len(Cout)),Cout)

for index, value in enumerate(Cout):
    plt.text(value, index,
             str(value))
#plt.plot([180000,(2*180000),(3*180000),(4*180000),(5*180000),(6*180000),(7*180000),(8*180000),(9*180000),(10*180000)],[90.63,96.45,104.27,111.47,120.15,128.42,134.09,142.21,151.09,158.38] , linestyle='-', marker='o', color='green', label='Greedy Search')
# ax1.legend(loc='upper left')
#ax1.legend([line1, line2, line3], ['label1', 'label2', 'label3'])
ax1.set_yticklabels([r'No Feat. Sel.$X$','No Feat. Sel.',r'Dur. $t$',r'$AvInt$',r'Hat. Dis. $h_D$',r'Scan Spe. $v_S$',r'Laser Pow. $P_L$' ])
plt.xlabel("Number of Selections")
plt.ylabel("Feature Names")
#plt.title("With Labels")
plt.savefig("D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/LateX/08_validation/fig/FeatureSelectionAlgorithm/PermutationImportance/Barchart.pdf", format='pdf',bbox_inches='tight') # saved as eps for high quality pictures


plt.show()