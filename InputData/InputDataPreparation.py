import pandas as pd
import numpy as np

if __name__ == "__main__":
    df=pd.read_csv("./original/collected_data2.csv", encoding='latin1')
    
    df2=df[df.columns[[4,5,6,7,35,36,37]]]
    df2['Relative Density [%]'] = df2['Relative density'].fillna(df2['Relative density (Archimedis)']) # fills the na values in relative density with the values from relative density archimedis
    df3 = df2.dropna(subset=['Relative Density [%]']) # deletes all rows, where Reltaive Density [%] has a na value
    del df3['Relative density'] # delets the relative density column 
    del df3['Relative density (Archimedis)'] # delets the relative density archimedis column 
    # df3.to_csv("./prepared/TEst.csv")
    gb = df3.groupby('Pulver') # groups the data after pulver type
    listgroups =[gb.get_group(x) for x in gb.groups] # creates a list with groups
    dfAlMgty80 = listgroups[0] #gets first group and defines new df
    del dfAlMgty80['Pulver'] # deletes pulver column
    dfAlMgty90 = listgroups[1]#gets second group and defines new df
    del dfAlMgty90['Pulver'] # deletes pulver column
    dfAlMgty80.to_csv("./prepared/AlMgty80_collected_data.csv") #stores the dataframe in csv
    dfAlMgty90.to_csv("./prepared/AlMgty90_collected_data.csv") #stores the dataframe in csv

