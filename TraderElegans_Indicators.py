import sys
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
import talib
#CLASSES----------------------------------------------------------------------------------------------------------------
class Constants:
    #**********FILE**********
    FILE_PATH = 'C:\\Users\\mbergbauer\\Desktop\\NN\\TraderElegans\\CSV_M1\\'
    #FILE_PATH = 'C:\\Users\\bergbmi\\Desktop\\NN\\TraderElegans\\Data\\M1_Raw\\'
    FILE_EXT = '.csv'
    FILENAME_EURUSD = 'DAT_ASCII_EURUSD_M1_'
    START_Y = 2017
    END_Y = 2017
    END_M = 4
    START_TIME = 80000
    END_TIME = 120000
    CREATE_FILE = True
    #**********MODEL PARAMS****
    BATCH_SIZE = 1
    EPOCHS = 5
    LOOKBACK = 34
    TIMESTEP = 1
    LOOKAHEAD = 20
    FEATURES = 11
    SCALE_MIN_MAX = True
    SCALE_MIN = 0
    SCALE_MAX = 1
    TRAIN_SIZE = 0.8
    VALID_SIZE = 0.1
    PREDI_SIZE = 0.1
    ONE_PIP = 10000
    RANDOM_SEED = 99
    TREND_TRESHOLD = 1
    LABEL_LONGTREND = [1,0,0]
    LABEL_SHORTTREND = [0,1,0]
    LABEL_NOTREND = [0,0,1]
    #**********CHECKPOINTING***
    CHECKP = False
    LOAD_CHECKP = False
    MODEL_PATH = 'C:\\Users\\mbergbauer\\Desktop\\NN\\TraderElegans\\Checkpoint\\checkpoint_model.hdf5'
    CHECKP_PATH = 'C:\\Users\\mbergbauer\\Desktop\\NN\\TraderElegans\\Checkpoint\\checkpoint_weights.hdf5'
    PLOT = False
    #**********INDICATORS*****
    EMA5 = 5
    EMA8 = 8
    EMA13 = 13
    MACD = (12,26,9)
    RSI = 13
    BB = (20,2,0)
    ADX = 14


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
# ----------------------------------------------------------------------------------------------------------------------
class Scaler:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(Constants.SCALE_MIN, Constants.SCALE_MAX))
    def scale_input_data(self,unscaled_input):
        tmp = []
        for example in unscaled_input:
            for timestep in example:
                tmp.append(timestep)
        self.scaler = MinMaxScaler(feature_range=(Constants.SCALE_MIN, Constants.SCALE_MAX))
        prices_scaled = self.scaler.fit_transform(tmp)
        tmp1 = np.array(prices_scaled).reshape((len(unscaled_input), Constants.LOOKBACK, Constants.FEATURES))
        return tmp1
    def descale_input_data(self,scaled_input):
        tmp = []
        for example in scaled_input:
            for timestep in example:
                tmp.append(timestep)
        prices_descaled = self.scaler.inverse_transform(tmp)
        tmp1 = np.array(prices_descaled).reshape((len(scaled_input), Constants.LOOKBACK, Constants.FEATURES))
        return tmp1
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
        tmp = day + ',' + record.rstrip()
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
    a = y * Constants.ONE_PIP
    if abs(a) >= Constants.TREND_TRESHOLD:
        if a > 0:
            return Constants.LABEL_LONGTREND
        else:
            return Constants.LABEL_SHORTTREND
    else:
        return Constants.LABEL_NOTREND
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
def predict(model, data):
    predicted_output = model.predict(data, batch_size=Constants.BATCH_SIZE)
    return predicted_output
#-----------------------------------------------------------------------------------------------------------------------
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------
def calc_indicators(data):
    tmp = []
    for example in data:
        open = (np.array(example)[:,2]).astype(float)
        high = np.array(example)[:,3].astype(float)
        low = np.array(example)[:,4].astype(float)
        close = np.array(example)[:,5].astype(float)

        ema5 = talib.EMA(close,timeperiod = Constants.EMA5)[-1]
        ema8 = talib.EMA(close,timeperiod = Constants.EMA8)[-1]
        ema13 = talib.EMA(close,timeperiod = Constants.EMA13)[-1]
        tmp_macd, tmp_macdsignal, tmp_macdhist = talib.MACD(close, fastperiod=Constants.MACD[0], slowperiod=Constants.MACD[1], signalperiod=Constants.MACD[2])
        macd = tmp_macd[-1]
        macdsignal = tmp_macdsignal[-1]
        macdhist = tmp_macdhist[-1]
        rsi = talib.RSI(close, timeperiod = Constants.RSI)[-1]
        tmp_upper, tmp_middle, tmp_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        upper = tmp_upper[-1]
        middle = tmp_middle[-1]
        lower = tmp_lower[-1]
        adx = talib.ADX(high,low,close,Constants.ADX)[-1]
        last_record = example[-1]
        date = last_record[0]
        time = last_record[1]
        record = str(ema5)+';'+str(ema8)+';'+str(ema13)+';'+str(macd)+';'+str(macdsignal)+';'+str(macdhist)+';'+str(rsi)+';'+str(upper)+';'+str(middle)+';'+str(lower)+';'+str(adx)
        tmp.append(record.split(';'))
    return tmp
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
all_data = read_data(data.get_out_file_name())
train_x, train_y, test_x, test_y, pred_x, pred_y = get_train_test_data(all_data)

print("Calculating indicators...")
train_x_indi = calc_indicators(train_x)
test_x_indi = calc_indicators(test_x)
pred_x_indi = calc_indicators(pred_x)
del train_x, test_x, pred_x, all_data

print("Scaling and reformating...")
if Constants.SCALE_MIN_MAX:
    scaler = StandardScaler()
    train_x_indi = scaler.fit(train_x_indi).transform(train_x_indi)
    test_x_indi = scaler.fit(test_x_indi).transform(test_x_indi)
    pred_x_indi = scaler.fit(pred_x_indi).transform(pred_x_indi)

train_x_indi = np.array(train_x_indi).astype(float).reshape(len(train_x_indi),1,Constants.FEATURES)
test_x_indi = np.array(test_x_indi).astype(float).reshape(len(test_x_indi),1,Constants.FEATURES)
pred_x_indi = np.array(pred_x_indi).astype(float).reshape(len(pred_x_indi),1,Constants.FEATURES)
train_y = np.array(train_y)
test_y = np.array(test_y)
pred_y = np.array(pred_y)



np.random.seed(Constants.RANDOM_SEED)
print('Assembling model...')
model = Sequential()
model.add(LSTM(units = 60, return_sequences=True, batch_input_shape=(Constants.BATCH_SIZE, Constants.TIMESTEP,Constants.FEATURES), stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 40, return_sequences=True, batch_input_shape=(Constants.BATCH_SIZE, Constants.TIMESTEP,Constants.FEATURES), stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 20, return_sequences=False, batch_input_shape=(Constants.BATCH_SIZE, Constants.TIMESTEP,Constants.FEATURES), stateful=True))
model.add(Dense(3,activation='softmax'))

if Constants.LOAD_CHECKP:
    print("Loading weights from checkpoint file...")
    model.load_weights(Constants.CHECKP_PATH)

print("Compiling model...")
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['acc'])
print(model.summary())
print('Training cases: ' + str(len(train_x_indi)))
print('Validation cases: ' + str(len(test_x_indi)))
print('Prediction cases: ' + str(len(pred_x_indi)))

print("Initializing checkpoint...")
if Constants.CHECKP:
    checkpoint = ModelCheckpoint(Constants.CHECKP_PATH, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

if not Constants.LOAD_CHECKP:
    print("Start training...")
    for i in range(Constants.EPOCHS):
        print("Epoch: " + str(i+1) + '\n')
        if Constants.CHECKP:
            model.fit(train_x_indi, train_y, validation_data=(test_x_indi, test_y), callbacks=callbacks_list, epochs=1, batch_size=Constants.BATCH_SIZE, verbose=2, shuffle=False)
        else:
            model.fit(train_x_indi, train_y, validation_data=(test_x_indi, test_y),epochs=1, batch_size=Constants.BATCH_SIZE, verbose=2, shuffle=False)
        model.reset_states()


prediction = predict(model,pred_x_indi)
for x in prediction:
    print(str(x))
#if Constants.PLOT:
#    plot_pred = scaler.descale_input_data(prediction)
#    pred_y=pred_y[140].reshape(1,20,1)
#    plot_actual = scaler.descale_input_data(pred_y)
#    plot_pred = plot_pred.reshape(plot_pred.size,Constants.FEATURES)
#    plot_actual = np.array(plot_actual).reshape(pred_y.size,Constants.FEATURES)
#    plot_results(plot_pred,plot_actual)

