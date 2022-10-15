
import csv
import numpy as np

def readcsv(filename,fcn1,fcn2,rcn):

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
        X[i][0]=row[fcn1]
        X[i][1]=row[fcn2]
        y[i]=row[rcn]
        i+=1

    # #Print Debugg
    # print(X)
    # print(X.shape)
    # print("\n")
    # print(y)
    # print(y.shape)
    # print("\n")
    return X, y