# importing csv module
import csv


def readvalue(filename,i,c):
    
    # initializing the titles and rows list
    fields = []
    rows = []
    
    # reading csv file
    with open(filename, 'r',encoding="utf8",errors="ignore") as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        fields = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)
    
        # # get total number of rows
        # print("Total no. of rows: %d"%(csvreader.line_num))
        
    
    # # printing the field names
    # print('Field names are:' + ', '.join(field for field in fields))
    density=float(rows[i-2][c]) #- 2 needs to be subtracted
    
    return density

def readcsv(filename):
    
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
        
    
    # # printing the field names
    # print('Field names are:' + ', '.join(field for field in fields))
    
    #Defining Sums of wanted columns
    SumIT = 0
    SumLF = 0
    SumHS = 0
    ltime = rows[1][0]
    for row in rows[1:]:
        # parsing each column of a row
        SumIT+= float(row[4])
        SumLF+= float(row[5])
        SumHS+= float(row[6])

        #DEBUGGING
        # print(float(row[4]))
        # print("\n")
        # print(float(row[5]))
        # print("\n")
        # print(float(row[6]))
        # print("\n")
        

    AvIT= SumIT/(csvreader.line_num-2)
    AvLF= SumLF/(csvreader.line_num-2)
    AvHS= SumHS/(csvreader.line_num-2)

    print("\nColumnsums")
    print(SumIT)
    print(SumLF)
    print(SumHS)

    print("\nAverage")
    print(AvIT)
    print(AvLF)
    print(AvHS)

    print("\nLast time step")
    print(ltime)
    row = [i,ltime,AvIT,AvLF,AvHS]
    return row
# csv file name
rows=[]
i = 6
while i <= 59:
    if i < 10:
        filename = "D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/experimentelle Daten/Simulation/0{}/0{}_t.csv".format(i,i)
    else:
       filename = "D:/Benutzerdateien/OneDrive - tuhh.de/TUHH/Semester 9/Studienarbeit/experimentelle Daten/Simulation/{}/{}_t.csv".format(i,i) 
    row = readcsv(filename)
    density = readvalue('Data_AlSi10Mg.csv',i,33)
    row.append(density)
    sourcepower = readvalue('Data_AlSi10Mg.csv',i,6)
    scanspeed = readvalue('Data_AlSi10Mg.csv',i,7)
    hatch_distance= readvalue('Data_AlSi10Mg.csv',i,9)
    row.append(sourcepower)
    row.append(scanspeed)
    row.append(hatch_distance)

    rows.append(row)
    i+=1


###WRITE CSV
# field names
fields = ['Number', 'Duration [s]', 'Average Interlayertemperature [°C]', 'Average Lack of fusion volume below 580 °C [%]', 'Average Hot spot volume above 880 °C [%]', "Relative Density [%]","Laser Power [W]","Scan Speed [mm/s]","Hatch distance [mm]"]
 
# # data rows of csv file
# rows = [ ['Nikhil', 'COE', '2', '9.0'],
#         ['Sanchit', 'COE', '2', '9.1'],
#         ['Aditya', 'IT', '2', '9.3'],
#         ['Sagar', 'SE', '1', '9.5'],
#         ['Prateek', 'MCE', '3', '7.8'],
#         ['Sahil', 'EP', '2', '9.1']]
 
# name of csv file
filename = "D:\Benutzerdateien\OneDrive - tuhh.de\TUHH\Semester 9\Studienarbeit\Python\Maschine Learning\InputData\collected_data.csv"
 
with open(filename,'w', newline='') as csvfile:
    #creating  a csv writer object
    csvwriter = csv.writer(csvfile)
    #writing the fields
    csvwriter.writerow(fields)
    # writing the data rows
    csvwriter.writerows(rows)


