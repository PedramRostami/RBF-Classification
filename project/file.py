import csv
import xlrd
def readCSV(address) :
    data = []
    with open(address) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if row:
                data.append(row)
    return data

def readXLSX(address) :
    data = []
    loc = (address)
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_name('Sheet1')
    for i in range(sheet.nrows):
        r_data = []
        for j in range(sheet.ncols):
            r_data.append(sheet.cell_value(i, j))
        data.append(r_data)
    return data

def writeCSV(address, data):
    f = open(address, 'w')
    for i in range(len(data)):
        for j in range(len(data[i])):
            if j < len(data[i]) - 1:
                f.write(str(data[i, j]) + ',')
            elif i < len(data) - 1:
                f.write(str(data[i, j]) + '\n')
            else:
                f.write(str(data[i, j]))
    f.close()
    return None