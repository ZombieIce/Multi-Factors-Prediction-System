from flask import Flask, request, render_template, jsonify
import pandas as pd

app = Flask(__name__)

data = pd.DataFrame()

def runModel(factors, paras):

    import pandas as pd
    from talib import MACD,SMA,LINEARREG_SLOPE,RSI,BBANDS,KAMA
    import numpy as np

    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import Dense,Activation,LSTM
    from keras.optimizers import SGD,RMSprop,Adagrad,Adam,Adamax, Nadam
    from keras.callbacks import EarlyStopping


    close=data['adjclose']
    date = list(data['date'])
    ret=close.diff()/close.shift(1)
    #2: rise, 1:stay, 0:drop
    ret_dir=1*(ret>ret.quantile(0.66))+1*(ret>ret.quantile(0.33))
    #ret_dir=ret>ret.quantile(0.6)
    MA_diff1=SMA(close, timeperiod=5)-close
    MA_diff2=SMA(close, timeperiod=5)-SMA(close, timeperiod=15)
    macd, macdsignal, macdhist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    SLOPE5=LINEARREG_SLOPE(close, timeperiod=5)
    SLOPE15=LINEARREG_SLOPE(close, timeperiod=15)
    SLOPE_diff=LINEARREG_SLOPE(close, timeperiod=5)-LINEARREG_SLOPE(close, timeperiod=10)
    SKEW=ret.rolling(window=10,center=False).skew()
    RSI20=RSI(close,timeperiod=20)
    RSI_SIG=1*(RSI20<40)-(RSI20>60)
    upper, middle, lower = BBANDS(close,timeperiod=15)
    BOLL_SIG=1*(close<lower)-(close>upper)
    MACD_RATIO=macd/SMA(macd, timeperiod=5)
    Kalman=KAMA(close, timeperiod=5)
    STD=close.rolling(window=10,center=False).std()
    KALMAN_SIG=1*(close<Kalman-0.1*STD)-(close>Kalman+0.1*STD)


    train_ratio=paras[0]
    trans_fee=paras[1]
    stop_loss=paras[2]
    take_profit=paras[3]

    originalDataSet=pd.DataFrame({'dir':ret_dir.shift(-1),0:MA_diff1,1:MA_diff2,2:macd,3:SLOPE5,
                    4:SLOPE15,5:SLOPE_diff,6:SKEW,7:RSI20,
                    8:RSI_SIG,9:BOLL_SIG,10:KALMAN_SIG,
                    11:MACD_RATIO})
    originalDataSet=originalDataSet.dropna(axis=0)
    del_num=[]
    for i in range(len(factors)):
        if factors[i]==0:
            del_num.append(i)
    originalDataSet=originalDataSet.drop(labels=del_num,axis=1)


    def normalize(train,flag=0):
        if flag==0:
            train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        else:
            y=train.iloc[:,0]
            train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            train_norm.iloc[:,0]=y
        return train_norm


    win_size=5
    def windows(da, window_size):
        start = 0
        while start < len(da):
            yield start, start + window_size
            start += 1
    #        start += (window_size // 2)
            
    def extract_segments(da, window_size = 30):
        segments = np.empty((0,(window_size)*(da.shape[1]-1)))
        labels = np.empty((0))
        for (start,end) in windows(da,window_size):
            if end>=len(originalDataSet):
                break
            if(len(da.iloc[start:end]) == (window_size)):
                signal = np.array(da.iloc[start:end,1:]).reshape(1,-1)[0]
                segments = np.vstack([segments, signal])
                labels = np.append(labels,da.iloc[end,0])
        
        return segments, labels


    segments,labels = extract_segments(originalDataSet,win_size)

    num_train = int(train_ratio*labels.shape[0])
    num_test = labels.shape[0]-num_train

    dummy_labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    reshaped_segments = segments.reshape([labels.shape[0],win_size,
                                        originalDataSet.shape[1]-1])
    train_x = reshaped_segments[:num_train]
    train_y = dummy_labels[:num_train]
    test_x = reshaped_segments[num_train:]
    test_y = dummy_labels[num_train:]


    batch_size = 32
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, 
                input_shape=[train_x.shape[1],
                                train_x.shape[2]], dropout=0.2))
    model.add(Activation('relu'))
    model.add(LSTM(units=50, return_sequences=True, 
                input_shape=[train_x.shape[1],
                                train_x.shape[2]], dropout=0.2))
    model.add(Activation('relu'))
    model.add(LSTM(units=24, dropout=0.2, recurrent_dropout=0.2))
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=0.0002, rho=0.9, epsilon=1e-06)
    adagrad = Adagrad(lr=0.001, epsilon=1e-06)
    adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    nadam = Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)



    model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

    model.fit(train_x, train_y, 
                        epochs=50, batch_size=batch_size, 
                        validation_data=(test_x, test_y), 
                        verbose=2, shuffle=False,
                        callbacks=[EarlyStopping(patience=40)])


    initial_captial=10000000
    pred_y=model.predict(test_x)

    norm_y = np.diff(np.log(pred_y),axis=0)
    norm_y=np.array(norm_y, dtype=float)

    price=list(close[len(close)-1-len(pred_y):len(close)-1])


    signal=[0 for i in range(len(pred_y))]
    buySell=[0 for i in range(len(pred_y))]
    buySell_size=[0 for i in range(len(pred_y))]
    position=[0 for i in range(len(pred_y))]
    Notional=[0 for i in range(len(pred_y))]
    CumNotional=[0 for i in range(len(pred_y))]
    PnL=[0 for i in range(len(pred_y))]
    cost=1
    station=0
    for i in range(1,len(pred_y)):
        temp=list(norm_y[i-1])
        signal[i]=temp.index(max(temp))-1
        if i==0:
            buySell[i]=signal[i]
            buySell_size[i]=(initial_captial+PnL[-1])//price[i]*buySell[i]
            position[i]=buySell_size[i]
            Notional[i]=-buySell_size[i]*price[i]
            CumNotional[i]=Notional[i]
            PnL[i]=CumNotional[i]+position[i]*price[i]
        else:
            station=np.sign(position[i-1])
            if signal[i]!=station:

                if station==1:
                    if signal[i]==0:
                        buySell_size[i]=-position[i-1]
                    if signal[i]==-1:
                        buySell_size[i]=-position[i-1]-(initial_captial+PnL[-1])//price[i]
                elif station==0:
                    if signal[i]==1:
                        buySell_size[i]=(initial_captial+PnL[-1])//price[i]
                    if signal[i]==-1:
                        buySell_size[i]=-(initial_captial+PnL[-1])//price[i]
                else:
                    if signal[i]==0:
                        buySell_size[i]=-position[i-1]
                    if signal[i]==1:
                        buySell_size[i]=-position[i-1]+(initial_captial+PnL[-1])//price[i]
            
            #stop loss and take profit
            if i!=0 and buySell_size[i]==0 and position[i-1]!=0:
                
                if position[i-1]>0:
                    if price[i]/cost>1+take_profit or price[i]/cost<1-stop_loss:
                        buySell_size[i]=-position[i-1]
                else:
                    if price[i]/cost<1-take_profit or price[i]/cost>1+stop_loss:
                        buySell_size[i]=-position[i-1]
            position[i]=position[i-1]+buySell_size[i]
            Notional[i]=-buySell_size[i]*price[i]
            CumNotional[i]=CumNotional[i-1]+Notional[i]
            PnL[i]=CumNotional[i]+position[i]*price[i]

            if buySell_size[i]!=0 and position[i]!=0:
                cost=price[i]

    oneMinPnL= np.diff(PnL)
    oneMinPnL[0]=0
    from math import sqrt
    Sharpe=oneMinPnL.mean()/oneMinPnL.std()*sqrt(252)

    pnl=list(PnL)
    price = [round(x, 2) for x in price]
    pnl = [round(x, 2) for x in pnl]
    date = date[len(close)-1-len(pred_y):len(close)-1]
    srp= round(Sharpe,2)
    totalrtn= round(PnL[-1]/initial_captial, 2)

    return date, pnl, price, srp, totalrtn



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/backTest', methods = ["POST"])
def backTest():
    resp_dict = {}
    if len(data) == 0:
        resp_dict['status_code'] = 202
    else:
        resp_dict['status_code'] = 200
        facotrs = request.form.getlist("facotrs[]")
        facotrs = [int(x) for x in facotrs]
        paras = request.form.getlist("paras[]")
        paras = [float(x) for x in paras]


        resp_dict['date'], resp_dict['pnl'], resp_dict['price'], resp_dict['spr'], resp_dict['totalrtn'] = runModel(facotrs, paras)
    
    return jsonify(resp_dict)

@app.route('/up_file', methods = ["POST"])
def upfile():
    global data
    data = pd.read_csv(request.files['file'])
    title = ['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume']
    show_data = []
    for i in range(7):
        curr_data = list(data.loc[i, :])
        curr_value_dict = {}
        for j in range(7):
            curr_value = curr_data[j]
            if j != 0:
                curr_value = str(round(curr_value, 2))
            curr_value_dict[title[j]] = curr_value
        show_data.append(curr_value_dict)

    return jsonify(show_data)

app.run()