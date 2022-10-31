import pandas as pd

def readcsvcol(filepath,cols):

    # reading csv file
    df=pd.read_csv(filepath, encoding='latin1')
    # print(df.head())
    df2=df[df.columns[cols]]
    # print(df2)
    # ID = df[["Laser Power [W]", "Scan Speed [mm/s]", "Relative Density [%]"]].to_numpy()
    # print(ID)
    return df2

if __name__ == "__main__":
    idf=readcsvcol("./InputData/collected_data.csv",[6,7,5])
    # print(idf.head())
    X = idf.iloc[:,0:2].values
    y = idf.iloc[:,2:3].values.ravel() #ravel funktion converts the array to a (n,) array instead of an (n,1) vector
    # print("X",X.shape)
    # print("y",y.shape)
    colheaders=list(idf.columns.values)
    print("Coloum headers",list(idf.columns.values))
    print("col head 1", colheaders[0])