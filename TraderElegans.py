
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


#CLASSES----------------------------------------------------------------------------------------------------------------
class Constants:
    #Class Variables (static)
#    file_path = 'C:\\Users\\mbergbauer\\Desktop\\NN\\TraderElegans\\CSV_M1\\'
    file_path = 'C:\\Users\\bergbmi\Desktop\\NN\\TraderElegans\\Data\\M1_Raw\\'
    file_ext = '.csv'
    filename_EURUSD = 'DAT_ASCII_EURUSD_M1_'
    filename_GBPUSD = 'DAT_ASCII_GBPUSD_M1_'
    filename_USDCHF = 'DAT_ASCII_USDCHF_M1_'
    start_y = 2000
    end_y = 2017
    end_m = 3
    start_time = 80000
    end_time = 120000
#-----------------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, constants):
        self.constants = constants
        self.files = [constants.filename_EURUSD, constants.filename_GBPUSD, constants.filename_USDCHF]

        if constants.end_m is None:
            self.out_file = constants.file_path + 'training_set_' + str(constants.start_y) + '_' + str(constants.end_y) + '_' + str(
                constants.start_time) + '_' + str(constants.end_time) + '.txt'
        else:
            self.out_file = constants.file_path + 'training_set_' + str(constants.start_y) + '_' + str(constants.end_y) + str(constants.end_m).zfill(2) + '_' + str(
                constants.start_time) + '_' + str(constants.end_time) + '.txt'

        self.EURUSD = {}
        self.GBPUSD = {}
        self.USDCHF = {}

    def get_out_file_name(self):
        return self.out_file

    def transform_data_file(self):
        self.out = open(self.out_file, 'w')
        for year in range(self.constants.start_y, self.constants.end_y + 1):

            if year == self.constants.end_y:
                for month in range(1, self.constants.end_m + 1):
                    file_EURUSD = self.constants.file_path + self.constants.filename_EURUSD + str(year) + str(month).zfill(2) + self.constants.file_ext
                    file_GBPUSD = self.constants.file_path + self.constants.filename_GBPUSD + str(year) + str(month).zfill(2) + self.constants.file_ext
                    file_USDCHF = self.constants.file_path + self.constants.filename_USDCHF + str(year) + str(month).zfill(2) + self.constants.file_ext
                    f1 = open(file_EURUSD, 'r')
                    f2 = open(file_GBPUSD, 'r')
                    f3 = open(file_USDCHF, 'r')

                    for line1, line2, line3 in zip(f1, f2, f3):
                        line1 = line1.replace(" ", "")
                        line2 = line2.replace(" ", "")
                        line3 = line3.replace(" ", "")
                        line1 = line1.split(';')
                        line2 = line2.split(';')
                        line3 = line3.split(';')

                        if self.constants.start_time <= int((line1[0])[8:14]) <= self.constants.end_time:
                            self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
                        if self.constants.start_time <= int((line2[0])[8:14]) <= self.constants.end_time:
                            self.GBPUSD.update({line2[0]: line2[1:(len(line2) - 1)]})
                        if self.constants.start_time <= int((line3[0])[8:14]) <= self.constants.end_time:
                            self.USDCHF.update({line3[0]: line3[1:(len(line3) - 1)]})

            else:
                file_EURUSD = self.constants.file_path + self.constants.filename_EURUSD + str(year) + self.constants.file_ext
                file_GBPUSD = self.constants.file_path + self.constants.filename_GBPUSD + str(year) + self.constants.file_ext
                file_USDCHF = self.constants.file_path + self.constants.filename_USDCHF + str(year) + self.constants.file_ext
                f1 = open(file_EURUSD, 'r')
                f2 = open(file_GBPUSD, 'r')
                f3 = open(file_USDCHF, 'r')

                for line1, line2, line3 in zip(f1, f2, f3):
                    line1 = line1.replace(" ", "")
                    line2 = line2.replace(" ", "")
                    line3 = line3.replace(" ", "")
                    line1 = line1.split(';')
                    line2 = line2.split(';')
                    line3 = line3.split(';')

                    if self.constants.start_time <= int((line1[0])[8:14]) <= self.constants.end_time:
                        self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
                    if self.constants.start_time <= int((line2[0])[8:14]) <= self.constants.end_time:
                        self.GBPUSD.update({line2[0]: line2[1:(len(line2) - 1)]})
                    if self.constants.start_time <= int((line3[0])[8:14]) <= self.constants.end_time:
                        self.USDCHF.update({line3[0]: line3[1:(len(line3) - 1)]})

        for key in sorted(self.EURUSD):

            tmp = key + ',' + str(self.EURUSD.get(key)).rstrip()
            if key in self.GBPUSD:
                tmp = tmp + ',' + str(self.GBPUSD.get(key)).rstrip()
            else:
                continue
            if key in self.USDCHF:
                tmp = tmp + ',' + str(self.USDCHF.get(key)).rstrip()
            else:
                continue
            tmp = tmp.replace("'", "")
            tmp = tmp.replace("[", "")
            tmp = tmp.replace("]", "")
            tmp = tmp.replace(" ", "")
            tmp = tmp + '\n'
            self.out.write(tmp)
        self.out.close()
#FUNCTIONS--------------------------------------------------------------------------------------------------------------
def read_data(file):
    raw_data = []
    f = open(file, 'r')
    countday = 0
    prevday = ''
    for line in f:
        day = line[:8]
        tmp = line[8:]
        if day == prevday:
            raw_data[countday].append(tmp)
        else:
            countday += 1
            raw_data[countday].append(tmp)
        prevday = day





    return raw_data

#-----------------------------------------------------------------------------------------------------------------------


#MAIN-------------------------------------------------------------------------------------------------------------------
data = Data(Constants())
#data.transform_data_file()
print(data.get_out_file_name())
raw_data = read_data(data.get_out_file_name())

pass

