from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import csv
import numpy as np

import pyvista as pv


filename = "./InputData/collected_data.csv"
#Load features and labels
# initializing the titles and rows list
fields = []
rows = []



# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

    # # get total number of rows
    # print("Total no. of rows: %d"%(csvreader.line_num))


X = np.zeros((csvreader.line_num-1,2))
y = np.zeros((csvreader.line_num-1))
i= 0
for row in rows[:]:
    X[i][0]=row[6]
    X[i][1]=row[7]
    y[i]=row[5]
    i+=1

#Print Debugg
print(X)
print(X.shape)
print("\n")
print(y)
print(y.shape)
print("\n")

#Add a new column for data input
k=0
Xrow, Xcol = X.shape
new_column = np.zeros((Xrow,1))
for ele in new_column:
    new_column[k][0]=y[k]
    k+=1

poi = np.append(X,new_column, axis=1)


# X.shape = (54,3)
# M = np.expand_dims(X, axis=  )
print("New points", poi)
# arr = np.append(X,new_column, axis=1)
# for ele in X:
#     X[k].append(y[k])

# print("X Geändert", X)





# #Display Datapoints
# plt.scatter(X.T[0],y)
# plt.show()
# plt.scatter(X.T[1],y)
# plt.show()

# def f(x, y):
#     return np.sin(np.sqrt(x ** 2 + y ** 2))

# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)

# X1, X2 = np.meshgrid(X[0], X[1])
# X3 = f(X1, X2)


fig = plt.figure()
ax = plt.axes(projection='3d')




# print("X DAten:\n", X1)

# print("Y DAten:\n", X2)

# print("Z DAten:\n", X3)


#CHANGE FONT TYPE
plt.rc('font',family='Linux Biolinum')
#CHANGE FONT SIZES
# plt.rc('font', size=12, weight='bold') #controls default text size
#plt.rc('font', size=12) #controls default text size
plt.rc('axes', titlesize=14.4) #fontsize of the title
plt.rc('axes', labelsize=12) #fontsize of the x and y labels
plt.rc('xtick', labelsize=11) #fontsize of the x tick labels
plt.rc('ytick', labelsize=11) #fontsize of the y tick labels
plt.rc('legend', fontsize=12) #fontsize of the legend
plt.set_loglevel("error") # just shows important error. Ignores warnings.


# ax.plot_wireframe(X.T[0], X.T[1], y, color='black')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.plot_surface(X1, X2, X3, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none');


# PLOT
ax.scatter(X.T[0], X.T[1], y, c=y, cmap='RdYlGn', linewidth=0.5);

ax.set_xlabel('Laser Power [W]')
ax.set_ylabel('Scan Speed [mm/s]')
ax.set_zlabel('Relative Density [%]');
# ax.set_title('Visualisation Dataset')
plt.savefig("./Visual/VisualisationDataset_LP_SS.eps", format='eps',bbox_inches='tight') # saved as eps for high quality pictures
plt.savefig("./Visual/VisualisationDataset_LP_SS.svg", format='svg',bbox_inches='tight')
# plt.savefig("VisualisationDataset.png", bbox_inches='tight')

plt.show()


#Interpolation of Datapoints
cloud = pv.PolyData(poi)
cloud.plot()


volume = cloud.delaunay_3d(alpha=30.)
shell = volume.extract_geometry()
shell.plot()