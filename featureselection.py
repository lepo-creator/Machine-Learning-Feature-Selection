from cProfile import label
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import os
from collections import defaultdict
from collections import Iterable
from sklearn.preprocessing import MinMaxScaler

#own imports
from csvpy import *
from plotpy import *




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

def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item






def getFeatureCorrelation(X,T,colheadersidf,scaler):

    #Plots 2 Windows next to each other uses window pixel size
    start_x, start_y, dx, dy = (0, 30, 1920, 1080)
    for i in range(1): # determins the number of plotted windows
        if i%3 == 0:
            x = start_x
            y = start_y  + (dy * (i//3) )

        fig, (ax1, ax2) = plt.subplots(1, 2,figsize =(16, 9))
        if i == 0:
            # #Get feature correlation
            corr = spearmanr(X).correlation
            # Ensure the correlation matrix is symmetric
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1)
            # creates an output csv file
            cmdf = pd.DataFrame(corr, columns=colheadersidf[:len(colheadersidf)-1], index=colheadersidf[:len(colheadersidf)-1])
            cmdf.to_csv('./Results/CorrelationMatrix.csv')
            # We convert the correlation matrix to a distance matrix before performing
            # hierarchical clustering using Ward's linkage.
            distance_matrix = 1 - np.abs(corr)
            # creates an output csv file
            dmdf = pd.DataFrame(distance_matrix, columns=colheadersidf[:len(colheadersidf)-1], index=colheadersidf[:len(colheadersidf)-1])
            dmdf.to_csv('./Results/DistanceMatrix.csv')
            dist_linkage = hierarchy.ward(squareform(distance_matrix))
            # print("Dist linkage",dist_linkage)
            dendro = hierarchy.dendrogram(
                dist_linkage,labels=colheadersidf[:len(colheadersidf)-1], ax=ax1, leaf_rotation=90
            )
            dendro_idx = np.arange(0, len(dendro["ivl"]))


            fig.suptitle('Dendrogram of Spearman Correlation Hierarchical Clustering', fontweight ="bold")
            ax1.set_ylabel("Distance")
            ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
            ax2.set_xticks(dendro_idx)
            ax2.set_yticks(dendro_idx)
            ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
            ax2.set_yticklabels(dendro["ivl"])
            fig.tight_layout()
            plt.savefig("./Visual/Dendogram_temp.eps",bbox_inches='tight', format='eps') # saved as eps for high quality pictures
            plt.savefig("./Visual/Dendogram_temp.svg",bbox_inches='tight', format='svg')
            plt.savefig("./Visual/Dendogram_temp.png",bbox_inches='tight')
                   

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(x, y, dx, dy)
        x += dx
    plt.show()

    
    # plt.show()


    #Select a feature for each cluster
    cluster_ids = hierarchy.fcluster(dist_linkage, t=T, criterion="distance")
    print("Cluster_ids", cluster_ids)
    cluster_id_to_feature_ids = defaultdict(list)
    print("cluster_id_to_feature_ids be3fore ", cluster_id_to_feature_ids)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    print("cluster_id_to_feature_ids after ", cluster_id_to_feature_ids)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    print("selected_features", selected_features)

    X_o = scaler.inverse_transform(X)
    X_sel_o = X_o[:, selected_features]

    colheadersidf_arr=np.asarray(colheadersidf) # turns a list into a np array
    selected_features.append(len(colheadersidf)-1)
    colheadersidf_sel = list(flatten(colheadersidf_arr[selected_features])) # flattens the list maybe not necessary

    
    return X_sel_o,colheadersidf_sel






def getPermutationImportance(model_sel,X_test,y_test,randomstate,scoring,colheadersidf_selin,idf):
    r = permutation_importance(model_sel, X_test, y_test,
                               n_repeats=30,
                               random_state=randomstate, scoring=scoring, n_jobs=-1)

    
    # for i in r.importances_mean.argsort()[::-1]:
    #     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    #         print(f"{colheadersidf[i]:<8}"
    #               f"         {r.importances_mean[i] :.3f}"
    #               f" +/- {r.importances_std[i] :.3f}")
    
    colheadersidf_sel=np.asarray(colheadersidf_selin[:len(colheadersidf_selin)]) # turns the list into a numpy array
    
    #Sorts Data
    sorted_meanimportances_a =np.sort(r.importances_mean, axis = 0) # Sorts importances_mean in a descending way
    sorted_stdimportances_a=r.importances_std.ravel()[r.importances_mean.argsort(axis=None).reshape(r.importances_mean.shape)] #Sorts importances standartderivation like importances_mean in a descending way
    sorted_names_a=colheadersidf_sel[:].ravel()[r.importances_mean.argsort(axis=None).reshape(r.importances_mean.shape)] #Sorts the names like importances_mean in a descending way

    sorted_meanimportances = sorted_meanimportances_a[::-1] # change sorting direction to ascending
    sorted_stdimportances = sorted_stdimportances_a[::-1] # change sorting direction to ascending
    sorted_names = sorted_names_a[::-1] # change sorting direction to ascending

    #Stores the sorted mean permutation importance and standart derivation of each feature in one csv file
    dmdf = pd.DataFrame(sorted_meanimportances.reshape(1,-1),columns=colheadersidf_selin[:len(colheadersidf_selin)-1], index=["Mean Permutation Importance (MPI)"])
    dmdf2=dmdf.append(pd.DataFrame(sorted_stdimportances.reshape(1,-1), columns=list(dmdf), index=["MPI Standart Derivation"]))
    dmdf2.to_csv('./Results/PermutationImportances.csv')

    # Counts the values which lie in the given threshold of r.importances_mean[i] - 2 * r.importances_std[i] > 0 to seperate the sorted array
    k=0
    for i in range(len(sorted_meanimportances)):
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            k+=1
    
    #get the final feature importances
    idf_sel = idf[sorted_names[:k]]
    X_sel2_o = idf_sel.iloc[:,:].values

    num_col = np.atleast_2d(X_sel2_o).shape[1] # gets the number of columns of a numpy array
    
    if num_col == 0:
        print("\n")
        print("-----INFORMATION AUTOMATIC FEATURE SELECTION-----")
        print("No suitable features were found by the automatic feature selection. Please change the input data, boundary parameters or the random state. ")
        print("Program closed.")
        print("-------------------------------------------------")
        quit()


    #normalise the selected data
    scaler_p = MinMaxScaler()
    X_sel2 = scaler_p.fit_transform(X_sel2_o)

    colheadersidf_sel3=np.asarray(list(idf_sel.columns.values)) # gets a list of df header strings
    colheadersidf_sel2=np.append(colheadersidf_sel3,colheadersidf_sel[len(colheadersidf_sel)-1])



    #Plots 2 Windows next to each other uses window pixel size
    start_x, start_y, dx, dy = (0, 30, 1920, 1080)
    for i in range(1): # determins the number of plotted windows
        if i%3 == 0:
            x = start_x
            y = start_y  + (dy * (i//3) )

        fig, ax = plt.subplots(figsize =(16, 9))
        if i == 0:
            ax.barh(sorted_names[:k],sorted_meanimportances[:k], xerr=sorted_stdimportances[:k], label = 'Acceptable Mean Permutation Importance', color="forestgreen")
            ax.barh(sorted_names[k:],sorted_meanimportances[k:], xerr=sorted_stdimportances[k:], label = 'Nonacceptable Mean Permutation Importance', color="firebrick")
            # ax.errorbar(label='Standart Derivation')
            
            #Creates space for the labels on the left side
            plt.gcf().subplots_adjust(left=0.20)
            # Plots a black vertical line a zero points to visualise negative entries
            plt.axvline(x = 0, color = 'black',linewidth = 0.5)

            # # Remove axes splines
            # for s in ['top', 'bottom', 'left', 'right']:
            #     ax.spines[s].set_visible(False)
            
            # # Remove x, y Ticks
            # ax.xaxis.set_ticks_position('none')
            # ax.yaxis.set_ticks_position('none')
            
            # # Add padding between axes and labels
            # ax.xaxis.set_tick_params(pad = 5)
            # ax.yaxis.set_tick_params(pad = 10)
            # Add x, y gridlines
            ax.grid(b = True, color ='grey',
                    linestyle ='-.', linewidth = 0.5,
                    alpha = 0.8)
            
            # Show top values
            ax.invert_yaxis()
            
            # # # Add annotation to bars
            # for i in ax.patches:
            #     print(i.get_width(),i.get_y())
            #     plt.text(i.get_width()+0.2, i.get_y()+0.5,
            #             str(round((i.get_width()), 20)),
            #             fontsize = 12, fontweight ='bold',
            #             color ='grey')




            legend_elements = [mpatches.Patch(color='forestgreen', label='Acceptable Mean Permutation Importance'),
                    mpatches.Patch(color='firebrick', label='Not Acceptable Mean Permutation Importance'),
                   Line2D([0], [0], color='black', ls='-',lw=1, label='Standart Derivation of Importances')]

            ax.legend(handles=legend_elements)
            # ax.legend()

            # cb=plt.colorbar(sc,ticks=np.linspace(minVal,maxVal,cp1a),format='%.2f')
            fig.suptitle('Visualisation of Mean Permutation Importance of each Feature ', fontweight ="bold")
            ax.set_xlabel("Mean Permutation Importance")
            # ax.set_ylabel(colheadersidf[1])
            # ax.set_zlabel(colheadersidf[2])
            # cb.set_label(colheadersidf[2]);
            plt.savefig("./Visual/3DVisualisationPermutationImportance_temp.eps",bbox_inches='tight', format='eps') # saved as eps for high quality pictures
            plt.savefig("./Visual/3DVisualisationPermutationImportance_temp.svg",bbox_inches='tight', format='svg')
            plt.savefig("./Visual/3DVisualisationPermutationImportance_temp.png",bbox_inches='tight')
            

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(x, y, dx, dy)
        x += dx
    plt.show()
        
    return X_sel2,colheadersidf_sel2, scaler_p


def creategroups(idf,numberintervals):
    numberpoints= numberintervals+1

    E_idf = idf.loc[:,['Laser Energy Density E [J/mm^3]']] # gets the specific energy density column
    E_arr=E_idf.iloc[:,:].values # creates an numpy array out of it
    E_min = np.min(E_arr) # searches for minimal value
    E_max = np.max(E_arr) # searches for the maximal value in the energy densities
    intervalpoints = np.linspace(E_min, E_max, num=numberpoints) # Creates an intervalls out of the min and max value with a spefic a
    print("Intervall points", intervalpoints)
    i = 0
    group_label=np.zeros(E_arr.shape) # creates a array with the correct size with zeros
    for ele in E_arr:
        k = 0
        while True:
            if k <len(intervalpoints)-1: # used to sort out the last case of the intervalls,because then the following if command is out of range
                if intervalpoints[k]<=ele[0]<intervalpoints[k+1]: # validates if a point lies inside an interval p1 <= element < p2
                    group_label[i][0]=k # writes the groupnumber in a pre defined array
                    break 
                k+=1
            else: # for the last case it just add the last element to the previous intervall,because groups with one element are senseless regarding the stratification
                group_label[i][0]=k-1 
                break
        i+=1
    print("The group labels",group_label)
    return group_label, intervalpoints



if __name__ == '__main__':
    #INPUT
    inputfilepath = "./InputData/collected_data.csv"
    invcol =[1,2,3,4,6,7,8,5] #First two numbers represent the read ouf col as input value and thrid number is the desired value (relative density)
    randomstate= 42
    trainingdatasize= 0.8
    scoring = 'neg_mean_absolute_percentage_error'#,'neg_root_mean_squared_error']
    T = 0.2

    idf=readcsvcol(inputfilepath,invcol)

    colheadersidf=list(idf.columns.values) # gets a list of df header strings

    #Spilts the Pandas dataframe in to numpy array Inputs X and result y 
    X = idf.iloc[:,0:len(invcol)-1].values
    y = idf.iloc[:,len(invcol)-1:len(invcol)].values.ravel() #ravel funktion converts the array to a (n,) array instead of an (n,1) vector

    #Remove features that have a close monotonic correlations
    X_sel,colheadersidf_sel = getfeaturecorrelation(X,T,colheadersidf)


    


    #Divide data in TRAINING DATA and TEST DATA
    X_train, X_test, y_train, y_test = train_test_split(X_sel,y, test_size=trainingdatasize, random_state=randomstate, shuffle=True)
    
    #Train a given model
    net = MLPRegressor(hidden_layer_sizes=(30,),activation='relu', solver = 'lbfgs', max_iter= 10000000, random_state=randomstate)
    model_sel = net.fit(X_train,y_train)

    X_sel2,colheadersidf_sel2 = getpermutationimportance(model_sel,X_test,y_test,randomstate,scoring,colheadersidf_sel,idf)
    print("X selected 2",X_sel2)
    print("Colnames 2",colheadersidf_sel2)

      




   



