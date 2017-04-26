import sys
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
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
    CREATE_FILE = False
    #**********SCALE**********
    CREATE_SCALED = False
    SCALE_MIN_MAX = False
    SCALE_MIN = 0
    SCALE_MAX = 1
    #**********DATA**********
    LOOKBACK = 30
    LOOKAHEAD = 5
    FEATURES = 12
    PREDICT_CLASS = False
    TRAIN_SIZE = 0.8
    VALID_SIZE = 0.1
    PREDI_SIZE = 0.1
    ONE_PIP = 10000
    RANDOM_SEED = 99
    #**********MODEL**********
    BATCH_SIZE = 10
    EPOCHS = 10
    TREND_TRESHOLD = 2
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
        tmp_y = calcTarget(tmp_x[-1][5], oneDayRawData[i+Constants.LOOKBACK:i+Constants.LOOKBACK+Constants.LOOKAHEAD])
        if not (data_x is None or data_y is None):
            data_x.append(tmp_x)
            data_y.append(get_y_categories(tmp_y))
    if (len(data_x) != len(data_y)):
        print("numbe of cases not equal for data_x and data_y for one day!")
        exit()
    return data_x, data_y
#-----------------------------------------------------------------------------------------------------------------------
def get_y_categories(y):
    return_y = []
    a = y * Constants.ONE_PIP
    if abs(a) >= Constants.TREND_TRESHOLD:
        if a > 0:
            return [1,0,0]
        else:
            return [0,1,0]
    else:
        return [0,0,1]
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
                return target
        elif target < 0:
            if float(price[5]) > float(price_t):
                target = 0
                return target
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
    tmp = np.array(tmp_all_cases)
#    tmp.reshape(len(tmp_all_cases), len(tmp_all_cases[0]), len(tmp_all_cases[0][0]))
    return tmp
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
if Constants.CREATE_FILE == True:
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
train_x = numpy_reshape(train_x)
train_y = np.array(train_y)
print("Reshaping Test data...")
test_x = numpy_reshape(test_x)
test_y = np.array(test_y)
print("Reshaping Prediction data...")
pred_x = numpy_reshape(pred_x)
pred_y = np.array(pred_y)

if Constants.SCALE_MIN_MAX:
    train_x = scale_input_data(train_x)
    test_x = scale_input_data(test_x)
    pred_x = scale_input_data(pred_x)

print("Seeding Random number generator")
np.random.seed(Constants.RANDOM_SEED)

model = Sequential()
model.add(LSTM(units = 12 , return_sequences=False, batch_input_shape=(Constants.BATCH_SIZE,Constants.LOOKBACK, Constants.FEATURES), stateful=False))
model.add(Dense(3,activation='softmax'))

#model = Sequential()
#model.add(LSTM(units = 60 , return_sequences=True, batch_input_shape=(Constants.BATCH_SIZE, Constants.LOOKBACK,Constants.FEATURES), stateful=True, activation='relu', kernel_initializer='random_uniform'))
#model.add(Dropout(0.2))
#model.add(LSTM(units = 24, return_sequences=False))
#model.add(Dropout(0.2))
#model.add(LSTM(units = 24, return_sequences=False))
#model.add(Dropout(0.2))
#model.add(Dense(3,activation='softmax'))

#if Constants.LOAD_CHECKP:
#    Print("Loading weights from checkpoint file...")
#    model.load_weights(Constants.CHECKP_PATH)

print("Compiling model...")
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['acc'])
print(model.summary())

#print("Initializing checkpoint...")
#if Constants.CHECKP:
#    checkpoint = ModelCheckpoint(Constants.CHECKP_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#    callbacks_list = [checkpoint]
print("Start training...")

model.fit(train_x, train_y, epochs=Constants.EPOCHS, batch_size=Constants.BATCH_SIZE, verbose=2, shuffle=False)

#for i in range(Constants.EPOCHS):
#    print("Epoch: " + str(i+1) + '\n')
#    if Constants.CHECKP:
#        model.fit(train_x, train_y, validation_data=(test_x, test_y), callbacks=callbacks_list, epochs=1, batch_size=Constants.BATCH_SIZE, verbose=1, shuffle=False)
#    else:
#        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1,batch_size=Constants.BATCH_SIZE, verbose=1, shuffle=False)
#    model.reset_states()

#if Constants.CHECKP:
#    model_json = model.to_json()
#    with open(Constants.MODEL_PATH, "w") as json_file:
#        json_file.write(model_json)


val_score = model.evaluate(test_x,test_y, batch_size=1, verbose=2)
pred_score = model.predict_classes(test_x, batch_size=1,verbose=2)

for prediction in pred_score:
    if prediction != 2:
        print(prediction)

