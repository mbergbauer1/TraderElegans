import sys
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
    FILE_PATH = 'C:\\Users\\mbergbauer\\Desktop\\NN\\TraderElegans\\CSV_M1\\'
#    FILE_PATH = 'C:\\Users\\bergbmi\Desktop\\NN\\TraderElegans\\Data\\M1_Raw\\'
    FILE_EXT = '.csv'
    FILENAME_EURUSD = 'DAT_ASCII_EURUSD_M1_'
    FILENAME_GBPUSD = 'DAT_ASCII_GBPUSD_M1_'
    FILENAME_USDCHF = 'DAT_ASCII_USDCHF_M1_'
    START_Y = 2016
    END_Y = 2017
    END_M = 3
    START_TIME = 80000
    END_TIME = 120000
    CREATE_FILE = False
    CREATE_SCALED = False
    SCALE_MIN_MAX = False
    SCALE_MIN = 0
    SCALE_MAX = 1
    LOOKBACK = 20
    LOOKAHEAD = 10
    FEATURES = 12
    BIP_LONG_SPREAD = 3
    PIP_SHORT_SPREAD = 3
    TRAIN_SIZE = 0.8
    ONE_PIP = 10000
    RANDOM_SEED = 99
#-----------------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self):
        self.files = [Constants.FILENAME_EURUSD, Constants.FILENAME_GBPUSD, Constants.FILENAME_USDCHF]
        if Constants.END_M == 0:
            if Constants.CREATE_SCALED == True:
                self.out_file = Constants.FILE_PATH + 'training_set_' + str(Constants.START_Y) + '_' + str(Constants.END_Y) + '_' + str(
                Constants.START_TIME) + '_' + str(Constants.END_TIME) + '_Scaled.txt'
            else:
                self.out_file = Constants.FILE_PATH + 'training_set_' + str(Constants.START_Y) + '_' + str(Constants.END_Y) + '_' + str(
                Constants.START_TIME) + '_' + str(Constants.END_TIME) + '.txt'
        else:
            if Constants.CREATE_SCALED == True:
                self.out_file = Constants.FILE_PATH + 'training_set_' + str(Constants.START_Y) + '_' + str(Constants.END_Y) + str(Constants.END_M).zfill(2) + '_' + str(Constants.START_TIME) + '_' + str(Constants.END_TIME) + '_Scaled.txt'
            else:
                self.out_file = Constants.FILE_PATH + 'training_set_' + str(Constants.START_Y) + '_' + str(Constants.END_Y) + str(
                Constants.END_M).zfill(2) + '_' + str(Constants.START_TIME) + '_' + str(Constants.END_TIME) + '.txt'
        self.EURUSD = {}
        self.GBPUSD = {}
        self.USDCHF = {}
    def get_out_file_name(self):
        return self.out_file
    def transform_data_file(self):
        self.out = open(self.out_file, 'w')
        for year in range(Constants.START_Y, Constants.END_Y + 1):
            if year == Constants.END_Y:
                if Constants.END_M != 0:
                    for month in range(1, Constants.END_M + 1):
                        file_EURUSD = Constants.FILE_PATH + Constants.FILENAME_EURUSD + str(year) + str(month).zfill(2) + Constants.FILE_EXT
                        file_GBPUSD = Constants.FILE_PATH + Constants.FILENAME_GBPUSD + str(year) + str(month).zfill(2) + Constants.FILE_EXT
                        file_USDCHF = Constants.FILE_PATH + Constants.FILENAME_USDCHF + str(year) + str(month).zfill(2) + Constants.FILE_EXT
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
                            if Constants.START_TIME <= int((line1[0])[8:14]) <= Constants.END_TIME:
                                self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
                            if Constants.START_TIME <= int((line2[0])[8:14]) <= Constants.END_TIME:
                                self.GBPUSD.update({line2[0]: line2[1:(len(line2) - 1)]})
                            if Constants.START_TIME <= int((line3[0])[8:14]) <= Constants.END_TIME:
                                self.USDCHF.update({line3[0]: line3[1:(len(line3) - 1)]})
                else:
                    file_EURUSD = Constants.FILE_PATH + Constants.FILENAME_EURUSD + str(
                        year) + Constants.FILE_EXT
                    file_GBPUSD = Constants.FILE_PATH + Constants.FILENAME_GBPUSD + str(
                        year) + Constants.FILE_EXT
                    file_USDCHF = Constants.FILE_PATH + Constants.FILENAME_USDCHF + str(
                        year) + Constants.FILE_EXT
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
                        if Constants.START_TIME <= int((line1[0])[8:14]) <= Constants.END_TIME:
                            self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
                        if Constants.START_TIME <= int((line2[0])[8:14]) <= Constants.END_TIME:
                            self.GBPUSD.update({line2[0]: line2[1:(len(line2) - 1)]})
                        if Constants.START_TIME <= int((line3[0])[8:14]) <= Constants.END_TIME:
                            self.USDCHF.update({line3[0]: line3[1:(len(line3) - 1)]})
            else:
                file_EURUSD = Constants.FILE_PATH + Constants.FILENAME_EURUSD + str(year) + Constants.FILE_EXT
                file_GBPUSD = Constants.FILE_PATH + Constants.FILENAME_GBPUSD + str(year) + Constants.FILE_EXT
                file_USDCHF = Constants.FILE_PATH + Constants.FILENAME_USDCHF + str(year) + Constants.FILE_EXT
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
                    if Constants.START_TIME <= int((line1[0])[8:14]) <= Constants.END_TIME:
                        self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
                    if Constants.START_TIME <= int((line2[0])[8:14]) <= Constants.END_TIME:
                        self.GBPUSD.update({line2[0]: line2[1:(len(line2) - 1)]})
                    if Constants.START_TIME <= int((line3[0])[8:14]) <= Constants.END_TIME:
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
    if Constants.SCALE_MIN_MAX is True:
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
        scaler = MinMaxScaler(feature_range=(Constants.SCALE_MIN, Constants.SCALE_MAX))
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
def get_train_test_data(all_days):
    all_cases_x = []
    all_cases_y = []
    all_cases_x_tmp = []
    all_cases_y_tmp = []
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    no_train = 0
    no_test = 0
    for day in all_days:
        all_cases_x_tmp, all_cases_y_tmp = extractCasesfromDay(day.data)
        if not (all_cases_x_tmp is None or all_cases_y_tmp is None):
            for case_x, case_y in zip(all_cases_x_tmp, all_cases_y_tmp):
                all_cases_x.append(case_x)
                all_cases_y.append(case_y)
    no_train = int(Constants.TRAIN_SIZE * len(all_cases_x))
    no_test = len(all_cases_x) - no_train
    train_x = all_cases_x[:no_train]
    train_y = all_cases_y[:no_train]
    test_x = all_cases_x[no_train:]
    test_y = all_cases_y[no_train:]
    return train_x, train_y, test_x, test_y
#-----------------------------------------------------------------------------------------------------------------------
def extractCasesfromDay(oneDayRawData):
    data_x = []
    data_y = []
    tmp_x = []
    tmp_y = []
    if len(oneDayRawData)-Constants.LOOKBACK - Constants.LOOKAHEAD < 0:
        return None, None
    for i in range(0,len(oneDayRawData)-Constants.LOOKBACK-Constants.LOOKAHEAD+1,1):
        tmp_x = oneDayRawData[i:i+Constants.LOOKBACK]
        tmp_y = calcTarget(tmp_x[-1][5], oneDayRawData[i+Constants.LOOKBACK:i+Constants.LOOKBACK+Constants.LOOKAHEAD])
        if not (data_x is None or data_y is None):
            data_x.append(tmp_x)
            data_y.append(tmp_y)
    if (len(data_x) != len(data_y)):
        print("numbe of cases not equal for data_x and data_y for one day!")
        exit()
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
    if not (Constants.SCALE_MIN_MAX):
        return round((target*Constants.ONE_PIP),1)
    else:
        return target
#-----------------------------------------------------------------------------------------------------------------------
def numpy_reshape(cases):
    tmp_all_cases = []
    tmp_float = []
    count = 0
    for case in cases:
#        count+=1
#        print("Convert cases: "+str(count)+'/'+str(len(cases)))
        tmp_case = []
        for timestep in case:
            tmp_float = [float(item) for item in timestep[2:]]
            tmp_case.append(list(tmp_float))
        tmp_all_cases.append(list(tmp_case))
    tmp = np.array(tmp_all_cases, np.float16)
    tmp.reshape(len(tmp_all_cases), len(tmp_all_cases[0]), len(tmp_all_cases[0][0]))
    return tmp
#-----------------------------------------------------------------------------------------------------------------------
#MAIN-------------------------------------------------------------------------------------------------------------------
data = Data()
if Constants.CREATE_FILE == True:
    data.transform_data_file()
train_x = []
train_y = []
test_x = []
test_y = []
print("Reading raw data...")
train_x, train_y, test_x, test_y = get_train_test_data(read_data(data.get_out_file_name()))
print('train_x: ' + str(len(train_x)))
print('train_y: ' + str(len(train_y)))
print('test_x: ' + str(len(test_x)))
print('test_y: ' + str(len(test_y)))
print("Reshaping train_x...")
train_x = numpy_reshape(train_x)
print("Reshaping test_x...")
test_x = numpy_reshape(test_x)

np.random.seed(Constants.RANDOM_SEED)
model = Sequential()

model.add(LSTM(units = 24 , return_sequences=True, input_shape=(Constants.LOOKBACK, Constants.FEATURES)))
model.add(LSTM(units = 48, return_sequences=True))
model.add(LSTM(units = 12, return_sequences=False))
model.add(Dense(output_dim=1))

model.compile(loss='mean_squared_error', optimizer='Adam',metrics=['acc'])
print(model.summary())
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=10, verbose=1)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

pass