import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from talib.abstract import EMA
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

def chart_candle(data, addp=[], jpg_name=None):
    mcolor = mpf.make_marketcolors(up='r', down='g', inherit=True)
    mstyle = mpf.make_mpf_style(base_mpf_style="yahoo", marketcolors=mcolor)
    mpf.plot(data, addplot=addp, style=mstyle, type="candle", volume=False, savefig=jpg_name)

def relation_heatmap(data, company_list):
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', linewidths=.5, xticklabels=company_list, yticklabels=company_list)
    plt.title('Stocks Daily Returns Relation')
    plt.savefig('relation.jpg')
    plt.close()

def train_lstm(data, company):
    # 正規化訓練資料集
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # 處理訓練資料集
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 建立模型
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 訓練模型
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.1)
    model.save(f'lstm_model_{company}.h5')
    
def test_lstm(data, company):
    # 正規化測試資料集
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # 載入之前訓練好的模型
    model = load_model(f'lstm_model_{company}.h5')

    # 處理測試資料集
    x_test, y_test = [], []
    for i in range(60, len(scaled_data)):
        x_test.append(scaled_data[i-60:i, 0])
        y_test.append(scaled_data[i, 0])
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # 測試模型
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)

    # 評估模型
    test_dates = pd.to_datetime(data.index[60:])
    rmse = np.sqrt(np.mean(((prediction - y_test) ** 2)))

    # 可視化預測結果
    plt.figure(figsize=(16,6))
    plt.plot(test_dates, data['Close'].values[60:], label=f'Actual Close Price ({company})')
    plt.plot(test_dates, prediction, label=f'Predicted Close Price ({company})')
    plt.title(f'Stock Price Prediction with LSTM ({company})')
    plt.xlabel('Days')
    plt.ylabel('Close Price') 
    plt.legend()
    plt.savefig(f'{company}_pred.jpg')
    plt.close()

if __name__ == '__main__':
    company_list = ['2330.tw', '2454.tw', '2317.tw', '2412.tw', '2308.tw'] # 台積電 聯發科 鴻海 中華電信 台達電
    #company_list = ['2330.tw']
    all_data = pd.DataFrame()

    # 下載 2008/01-現在 的股價資料
    for company in company_list:
        data = yf.download(company, period='16y', interval='1d')
        data['Company'] = company
        all_data = pd.concat([all_data, data])

    # 切割資料集
    train_data = all_data.loc['2008-01-01':'2023-06-30']
    test_data = all_data.loc['2023-07-01':]
    train_data.to_csv("train.csv")
    test_data.to_csv("test.csv")
        
    # 5間公司的K線圖 & 120天移動平均線
    for company in company_list:
        k_data = train_data[train_data['Company'] == company]
        k_data['ema'] = EMA(k_data['Close'], timeperiod=120)
        addp=[]
        addp.append(mpf.make_addplot(k_data['ema']))
        chart_candle(k_data, addp=addp, jpg_name=f"{company[:4]}.jpg")

    # 不同股票的相關性分析
    closing_dff = train_data.pivot(columns='Company', values='Adj Close')
    tech_rets = closing_dff.pct_change()
    relation_heatmap(tech_rets, company_list)

    # 訓練LSTM預測股票
    for company in company_list:
        train_lstm(train_data[train_data['Company'] == company], company[:4])
        test_lstm(test_data[test_data['Company'] == company], company[:4])
        print(f"------- {company} Completed -------")