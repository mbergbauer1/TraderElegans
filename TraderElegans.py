
import numpy as np
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
    start_y = 2015
    end_y = 2016
    end_m = 0
    start_time = 80000
    end_time = 120000
    create_file = True
    create_scaled = False
    scale_min_max = True
    lookback = 20
    lookahead = 10
#-----------------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self):
        self.Constants = Constants
        self.files = [Constants.filename_EURUSD, Constants.filename_GBPUSD, Constants.filename_USDCHF]

        if Constants.end_m == 0:
            if Constants.create_scaled == True:
                self.out_file = Constants.file_path + 'training_set_' + str(Constants.start_y) + '_' + str(Constants.end_y) + '_' + str(
                Constants.start_time) + '_' + str(Constants.end_time) + '_Scaled.txt'
            else:
                self.out_file = Constants.file_path + 'training_set_' + str(Constants.start_y) + '_' + str(Constants.end_y) + '_' + str(
                Constants.start_time) + '_' + str(Constants.end_time) + '.txt'
        else:
            if Constants.create_scaled == True:
                self.out_file = Constants.file_path + 'training_set_' + str(Constants.start_y) + '_' + str(Constants.end_y) + str(Constants.end_m).zfill(2) + '_' + str(Constants.start_time) + '_' + str(Constants.end_time) + '_Scaled.txt'
            else:
                self.out_file = Constants.file_path + 'training_set_' + str(Constants.start_y) + '_' + str(Constants.end_y) + str(
                Constants.end_m).zfill(2) + '_' + str(Constants.start_time) + '_' + str(Constants.end_time) + '.txt'

        self.EURUSD = {}
        self.GBPUSD = {}
        self.USDCHF = {}

    def get_out_file_name(self):
        return self.out_file

    def transform_data_file(self):
        self.out = open(self.out_file, 'w')
        for year in range(self.Constants.start_y, self.Constants.end_y + 1):

            if year == self.Constants.end_y:
                for month in range(1, self.Constants.end_m + 1):
                    file_EURUSD = self.Constants.file_path + self.Constants.filename_EURUSD + str(year) + str(month).zfill(2) + self.Constants.file_ext
                    file_GBPUSD = self.Constants.file_path + self.Constants.filename_GBPUSD + str(year) + str(month).zfill(2) + self.Constants.file_ext
                    file_USDCHF = self.Constants.file_path + self.Constants.filename_USDCHF + str(year) + str(month).zfill(2) + self.Constants.file_ext
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

                        if self.Constants.start_time <= int((line1[0])[8:14]) <= self.Constants.end_time:
                            self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
                        if self.Constants.start_time <= int((line2[0])[8:14]) <= self.Constants.end_time:
                            self.GBPUSD.update({line2[0]: line2[1:(len(line2) - 1)]})
                        if self.Constants.start_time <= int((line3[0])[8:14]) <= self.Constants.end_time:
                            self.USDCHF.update({line3[0]: line3[1:(len(line3) - 1)]})

            else:
                file_EURUSD = self.Constants.file_path + self.Constants.filename_EURUSD + str(year) + self.Constants.file_ext
                file_GBPUSD = self.Constants.file_path + self.Constants.filename_GBPUSD + str(year) + self.Constants.file_ext
                file_USDCHF = self.Constants.file_path + self.Constants.filename_USDCHF + str(year) + self.Constants.file_ext
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

                    if self.Constants.start_time <= int((line1[0])[8:14]) <= self.Constants.end_time:
                        self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
                    if self.Constants.start_time <= int((line2[0])[8:14]) <= self.Constants.end_time:
                        self.GBPUSD.update({line2[0]: line2[1:(len(line2) - 1)]})
                    if self.Constants.start_time <= int((line3[0])[8:14]) <= self.Constants.end_time:
                        self.USDCHF.update({line3[0]: line3[1:(len(line3) - 1)]})

        raw = []
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
# ----------------------------------------------------------------------------------------------------------------------
class OneDayRawData:
    def __init__(self):
        self.data = []
    def add(self,record):
        self.data.append(record)
    def get(self):
        return self.data
#FUNCTIONS--------------------------------------------------------------------------------------------------------------
def read_data(file):
    raw_data = []
    f = open(file, 'r')
    raw_date_time = []
    raw_prices = []
    raw_prices_tmp = []
    if Constants.scale_min_max is True:
        for line in f:
            raw = line.split(',')
            raw_date_time_tmp = raw[0]
#            raw_prices_tmp = raw[1:]
            for item in raw[1:]:
                raw_prices_tmp.append(float(item))
            raw_date_time.append(raw_date_time_tmp)
            if len(raw_prices_tmp) != 12:
                print(" WE HAVE A PROBLEM: " + raw)
            raw_prices.append(list(raw_prices_tmp))
            del raw_prices_tmp[:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        raw_prices_scaled = scaler.fit_transform(np.array(raw_prices)).tolist()
        prevday = ''
        oneDay = None
        for rdt, rps in zip(raw_date_time, raw_prices_scaled):

            line = str(rdt) + ',' + ''.join(str(rps))
            line = line.replace('[','')
            line = line.replace(']', '')
            line = line.replace(' ', '')
            day = line[:8]
            record = line[8:]
            tmp = day + ',' + record
            if day == prevday:
                oneDay.add(tmp.split(","))
            else:
                if not oneDay == None:
                    raw_data.append(oneDay)
                oneDay = OneDayRawData()
                oneDay.add(tmp.split(","))
            prevday = day
        return raw_data
    else:
        prevday = ''
        oneDay = None
        for line in f:
            day = line[:8]
            record = line[8:]
            tmp = day + ',' + record
            if day == prevday:
                oneDay.add(tmp.split(","))
            else:
                if not oneDay == None:
                    raw_data.append(oneDay)
                oneDay = OneDayRawData()
                oneDay.add(tmp.split(","))
            prevday = day
        return raw_data
#-----------------------------------------------------------------------------------------------------------------------
def extractCasesfromDay(oneDayRawData, lookback, lookahead):
    data_x = []
    data_y = []
    tmp_x = []
    tmp_y = []
    if len(oneDayRawData)-lookback - lookahead < 0:
        return None, None

    for i in range(0,len(oneDayRawData)-lookback-lookahead+1,1):
        tmp_x = oneDayRawData[i:i+lookback]
        tmp_y = calcTarget(tmp_x[-1][5], oneDayRawData[i+lookback:i+lookback+lookahead])
        if not (data_x is None or data_y is None):
            data_x.append(tmp_x)
            data_y.append(tmp_y)
    return data_x, data_y
#-----------------------------------------------------------------------------------------------------------------------
def calcTarget(price_t, future_series):
    price_tplus10 = float(future_series[-1][5])
    target = float(price_tplus10) - float(price_t)
    if target == 0:
        return target

    for price in future_series:
        if target > 0:
            if float(price[5]) < float(price_t):
                target = 0
                break
        elif target < 0:
            if float(price[5]) > float(price_t):
                target = 0
                break
    return target
#-----------------------------------------------------------------------------------------------------------------------
#MAIN-------------------------------------------------------------------------------------------------------------------
data = Data()
if Constants.create_file == True:
    data.transform_data_file()
print("File:" + data.get_out_file_name())
daily_data = read_data(data.get_out_file_name())
print("Number of days in file: " + str(len(daily_data)))
data_x = []
data_y = []

for day in daily_data:
    tmp_data_x , tmp_data_y = extractCasesfromDay(day.data,Constants.lookback,Constants.lookahead)
    if not (tmp_data_x is None or tmp_data_y is None):
        data_x.append(tmp_data_x)
        data_y.append(tmp_data_y)
daily_data = None
pass