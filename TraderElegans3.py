import sys
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.layers import TimeDistributed
#CLASSES----------------------------------------------------------------------------------------------------------------
class Constants:
    #**********FILE**********
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
    CREATE_FILE = True
    #**********SCALE**********
    SCALE_MIN_MAX = False
    SCALE_MIN = 0
    SCALE_MAX = 1
    #**********DATA**********
    LOOKBACK = 20
    LOOKAHEAD = 20
    FEATURES = 1
    TRAIN_SIZE = 0.8
    VALID_SIZE = 0.1
    PREDI_SIZE = 0.1
    ONE_PIP = 10000
    RANDOM_SEED = 99
    #**********MODEL**********
    BATCH_SIZE = 20
    EPOCHS = 1
    TREND_TRESHOLD = 4
    LABEL_LONGTREND = 2
    LABEL_SHORTTREND = 1
    LABEL_NOTREND = 0
    CHECKP = False
    MODEL_PATH = 'C:\\Users\\mbergbauer\\Desktop\\NN\\TraderElegans\\Checkpoint\\checkpoint_model.hdf5'
    CHECKP_PATH = 'C:\\Users\\mbergbauer\\Desktop\\NN\\TraderElegans\\Checkpoint\\checkpoint_weights.hdf5'
    LOAD_CHECKP = False

#-----------------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self):
        self.files = [Constants.FILENAME_EURUSD]
        if Constants.END_M == 0:
            self.out_file = Constants.FILE_PATH + 'training_set_' + str(Constants.START_Y) + '_' + str(Constants.END_Y) + '_' + str(
            Constants.START_TIME) + '_' + str(Constants.END_TIME) + '.txt'
        else:
            self.out_file = Constants.FILE_PATH + 'training_set_' + str(Constants.START_Y) + '_' + str(Constants.END_Y) + str(
            Constants.END_M).zfill(2) + '_' + str(Constants.START_TIME) + '_' + str(Constants.END_TIME) + '.txt'
        self.EURUSD = {}
    def get_out_file_name(self):
        return self.out_file
    def transform_data_file(self):
        self.out = open(self.out_file, 'w')
        for year in range(Constants.START_Y, Constants.END_Y + 1):
            if year == Constants.END_Y:
                if Constants.END_M != 0:
                    for month in range(1, Constants.END_M + 1):
                        file_EURUSD = Constants.FILE_PATH + Constants.FILENAME_EURUSD + str(year) + str(month).zfill(2) + Constants.FILE_EXT
                        f1 = open(file_EURUSD, 'r')
                        for line1 in f1:
                            line1 = line1.replace(" ", "")
                            line1 = line1.split(';')
                            if Constants.START_TIME <= int((line1[0])[8:14]) <= Constants.END_TIME:
                                self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
                else:
                    file_EURUSD = Constants.FILE_PATH + Constants.FILENAME_EURUSD + str(
                        year) + Constants.FILE_EXT
                    f1 = open(file_EURUSD, 'r')
                    for line1 in f1:
                        line1 = line1.replace(" ", "")
                        line1 = line1.split(';')
                        if Constants.START_TIME <= int((line1[0])[8:14]) <= Constants.END_TIME:
                            self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
            else:
                file_EURUSD = Constants.FILE_PATH + Constants.FILENAME_EURUSD + str(year) + Constants.FILE_EXT
                f1 = open(file_EURUSD, 'r')
                for line1 in f1:
                    line1 = line1.replace(" ", "")
                    line1 = line1.split(';')
                    if Constants.START_TIME <= int((line1[0])[8:14]) <= Constants.END_TIME:
                        self.EURUSD.update({line1[0]: line1[1:(len(line1) - 1)]})
        raw = []
        for key in sorted(self.EURUSD):
            tmp = key + ',' + str(self.EURUSD.get(key)).rstrip()
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
    no_pred = 0
    for day in all_days:
        all_cases_x_tmp, all_cases_y_tmp = extractCasesfromDay(day.data)
        if not (all_cases_x_tmp is None or all_cases_y_tmp is None):
            for case_x, case_y in zip(all_cases_x_tmp, all_cases_y_tmp):
                all_cases_x.append(case_x)
                all_cases_y.append(case_y)
    no_train = int(Constants.TRAIN_SIZE * len(all_cases_x))
    no_test = int(Constants.VALID_SIZE * len(all_cases_x))
    no_pred =  len(all_cases_x) - no_train - no_test
    train_x = all_cases_x[:no_train]
    train_y = all_cases_y[:no_train]
    test_x = all_cases_x[no_train:(no_train+no_test)]
    test_y = all_cases_y[no_train:(no_train+no_test)]
    pred_x = all_cases_x[(no_train+no_test):]
    pred_y = all_cases_y[(no_train+no_test):]
    return train_x, train_y, test_x, test_y, pred_x, pred_y
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
        tmp_y = oneDayRawData[i+Constants.LOOKBACK:i+Constants.LOOKBACK+Constants.LOOKAHEAD]
        if not (data_x is None or data_y is None):
            data_x.append(tmp_x)
            data_y.append(tmp_y)
    if (len(data_x) != len(data_y)):
        print("numbe of cases not equal for data_x and data_y for one day!")
        exit()
    return data_x, data_y
#-----------------------------------------------------------------------------------------------------------------------
def numpy_reshape(cases,x_y):
    tmp_all_cases = []
    tmp_float = []
    for case in cases:
        tmp_case = []
        for timestep in case:
            tmp_case.append(float(timestep[5]))
        tmp_all_cases.append(list(tmp_case))
    if x_y == 'x':
        return np.array(tmp_all_cases).reshape(len(cases),Constants.LOOKBACK,Constants.FEATURES)
    elif x_y == 'y':
        return np.array(tmp_all_cases).reshape(len(cases),Constants.LOOKAHEAD,Constants.FEATURES)
    else:
        return None
#-----------------------------------------------------------------------------------------------------------------------
def scale_input_data(unscaled_input):
    tmp = []
    for example in unscaled_input:
        for timestep in example:
            tmp.append(timestep)


    scaler = MinMaxScaler(feature_range=(Constants.SCALE_MIN, Constants.SCALE_MAX))
    prices_scaled = scaler.fit_transform(tmp)

    tmp1 = np.array(prices_scaled).reshape((len(unscaled_input),Constants.LOOKBACK,Constants.FEATURES))
    return tmp1
#-----------------------------------------------------------------------------------------------------------------------

#MAIN-------------------------------------------------------------------------------------------------------------------
data = Data()
if Constants.CREATE_FILE:
    data.transform_data_file()
train_x = []
train_y = []
test_x = []
test_y = []
pred_x = []
pred_y = []

print("Reading raw data...")
train_x, train_y, test_x, test_y, pred_x, pred_y = get_train_test_data(read_data(data.get_out_file_name()))
print('Training cases: ' + str(len(train_x)))
print('Validation cases: ' + str(len(test_x)))
print('Prediction cases: ' + str(len(test_x)))
print("Reshaping Training data...")
train_x = numpy_reshape(train_x,'x')
train_y = numpy_reshape(train_y,'y')
print("Reshaping Test data...")
test_x = numpy_reshape(test_x,'x')
test_y = numpy_reshape(test_y,'y')
print("Reshaping Prediction data...")
pred_x = numpy_reshape(pred_x,'x')
pred_y = numpy_reshape(pred_y,'y')
pass
if Constants.SCALE_MIN_MAX:
    train_x = scale_input_data(train_x)
    test_x = scale_input_data(test_x)
    pred_x = scale_input_data(pred_x)

print("Seeding Random number generator")
np.random.seed(Constants.RANDOM_SEED)

#model = Sequential()
#model.add(LSTM(units = 12 , return_sequences=False, batch_input_shape=(Constants.BATCH_SIZE,Constants.LOOKBACK, Constants.FEATURES), stateful=False))
#model.add(Dense(3,activation='sigmoid'))

model = Sequential()
model.add(LSTM(units = 120 , return_sequences=True, batch_input_shape=(len(train_x), Constants.LOOKBACK,Constants.FEATURES), stateful=False))
#model.add(Dropout(0.2))
#model.add(LSTM(units = 60, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(units = 12, return_sequences=False))
#model.add(Dropout(0.2))
#model.add(Dense(3,activation='softmax'))
model.add(TimeDistributed(Dense(1)))

print("Compiling model...")
model.compile(loss='mse', optimizer='adam',metrics=['acc'])
print(model.summary())

print("Start training...")
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=Constants.EPOCHS, batch_size=Constants.BATCH_SIZE, verbose=1, shuffle=False)

#for i in range(Constants.EPOCHS):
#    print("Epoch: " + str(i+1) + '\n')
#    if Constants.CHECKP:
#        model.fit(train_x, train_y, validation_data=(test_x, test_y), callbacks=callbacks_list, epochs=1, batch_size=Constants.BATCH_SIZE, verbose=1, shuffle=False)
#    else:
#        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1,batch_size=Constants.BATCH_SIZE, verbose=1, shuffle=False)
#    model.reset_states()



